"""KV Reuse, for temporal redundancy: per-stream cache state, change detection and the refresh rule.

Consecutive screenshots (every 5 s) are often identical; only small regions change (typing, cursor). Across frames, a block counts as
changed if any pixel differs by more than delta (8 grey levels = 8/255) from when it was LAST computed (not the previous frame, so slow
drifts cannot accumulate unseen). Changed kept blocks are recomputed; the ViT reuses the cached K/V of all other kept blocks (vit.py),
and the LLM reuses the cached K/V of their visual tokens (llm.py).

Full refresh (recompute all kept blocks) when the stream starts, when more than rho of the kept blocks changed (a new screen), or every
K frames since the last refresh (bounds the drift of the approximate reuse).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .config import MethodConfig
from .model import Grid, Model


def to_device(idx: np.ndarray, m: Model) -> torch.Tensor:
    # changed: index sets are built on the host and uploaded (MPS index tensors from nonzero() were corrupted under memory pressure)
    return torch.from_numpy(np.ascontiguousarray(idx, dtype=np.int64)).to(m.device)


class StreamState:
    """Everything one stream carries from frame to frame (one per Engine; reset when a new stream starts)."""

    def __init__(self, m: Model, g: Grid, approx_llm: bool):
        nl, n, nh, hd = len(m.vit.blocks), g.n_patches, m.vit_heads, m.vit_head_dim
        out = m.vcfg.out_hidden_size
        self.key = g.key
        # ViT: post-RoPE K/V of every patch in every ViT layer, so changed blocks attend over unchanged ones without recomputing them
        self.vit_k = torch.empty(nl, n, nh, hd, dtype=m.dtype, device=m.device)
        self.vit_v = torch.empty_like(self.vit_k)
        # KV Reuse (ViT): per-block ViT outputs (merger = LLM visual token, DeepStack = features added to early LLM layers)
        self.merged = torch.empty(g.n_blocks, out, dtype=m.dtype, device=m.device)
        self.deepstack = torch.empty(len(m.deepstack_layers), g.n_blocks, out, dtype=m.dtype, device=m.device)
        # KV Reuse (ViT): pixels of each block at its last recompute (the change-detection reference)
        self.ref = torch.empty(g.n_blocks, 4 * m.patch_w.shape[1], dtype=torch.float32, device=m.device)
        # LLM: post-RoPE K/V of every LLM layer per FULL-grid slot (slot = position in the prompt with all blocks), so a block keeps
        # its slot even when Token Pruning keeps a different set of blocks next frame (~100 MB fp16)
        self.llm_k = self.llm_v = None
        if approx_llm:
            tc = m.tcfg
            self.llm_k = torch.empty(tc.num_hidden_layers, g.length, tc.num_key_value_heads, tc.head_dim, dtype=m.dtype, device=m.device)
            self.llm_v = torch.empty_like(self.llm_k)
        self.reset(m, g)

    def reset(self, m: Model, g: Grid):
        self.valid = torch.zeros(g.n_blocks, dtype=torch.bool, device=m.device)   # block has cached ViT state
        self.since_refresh = 0
        self.prev_kept: Optional[np.ndarray] = None     # kept blocks of the previous frame (LLM prefix reuse)
        self.prev_emb: Optional[torch.Tensor] = None    # previous embedding (returned unchanged when nothing changed)
        self.exact_kv = None                            # exact LLM prefix reuse: HF DynamicCache of the previous frame
        self.llm_valid = np.zeros(g.length, bool) if self.llm_k is not None else None   # KV Reuse (LLM): slot has cached K/V


@dataclass
class FramePlan:
    """What to compute for one frame."""
    kept: np.ndarray                    # block ids fed to ViT/LLM after Token Pruning (sorted)
    recompute: np.ndarray               # block ids that go through the ViT this frame (subset of kept; = kept on a refresh)
    refresh: bool
    changed_frac: float                 # fraction of kept blocks that changed (-1 at stream start / without KV Reuse (ViT))
    since_refresh: int                  # frames since the last refresh, after this frame
    state: Optional[StreamState]
    kept_t: torch.Tensor = None         # device copies of the index sets
    recompute_t: torch.Tensor = None
    # LLM part, filled by llm.plan_llm()
    prefix: int = 0                     # P: positions [0, P) are bit-identical to the previous frame
    skip_llm: bool = False              # nothing changed at all: reuse the previous embedding
    llm_rows: Optional[np.ndarray] = None   # KV Reuse (LLM): positions recomputed in the LLM (R)


@torch.inference_mode()
def plan_frame(cfg: MethodConfig, m: Model, g: Grid, state: Optional[StreamState], block_pixels: torch.Tensor, first: bool,
               keep_mask: Optional[np.ndarray]) -> FramePlan:
    """Kept blocks (Token Pruning) and recomputed blocks (KV Reuse) of one frame. block_pixels: (n_blocks, 4*1536) float32 on the
    device."""
    kept = np.flatnonzero(keep_mask) if keep_mask is not None else np.arange(g.n_blocks)
    kept_t = to_device(kept, m)
    if state is None:                                   # no KV Reuse (ViT): every kept block goes through the ViT
        return FramePlan(kept, kept, True, -1.0, 0, None, kept_t, kept_t)
    refresh = first or state.since_refresh + 1 >= cfg.refresh_every
    frac = -1.0
    changed = None
    if not first:
        changed = ((block_pixels - state.ref).abs().amax(1) > cfg.delta) | ~state.valid
        frac = float(changed[kept_t].float().mean())    # float32 mean on the device (rounding decides exact ties at rho)
        refresh = refresh or frac > cfg.refresh_rho
    if refresh:
        recompute, recompute_t = kept, kept_t
    else:
        recompute = kept[changed.cpu().numpy()[kept]]
        recompute_t = to_device(recompute, m)
    return FramePlan(kept, recompute, refresh, frac, 0 if refresh else state.since_refresh + 1, state, kept_t, recompute_t)


def commit(plan: FramePlan, emb: torch.Tensor):
    """After the frame: remember what the next frame's prefix reuse / skip needs."""
    st = plan.state
    if st is not None:
        st.prev_kept = plan.kept
        st.prev_emb = emb
        st.since_refresh = plan.since_refresh
