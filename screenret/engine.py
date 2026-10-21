"""Engine: one screenshot in, one embedding out, carrying per-stream state between frames.

Engine.step:  S keep set + T_vit plan (change detection, refresh rule)  ->  LLM plan (prefix P, skip, T_llm rows R)
              ->  ViT (full or partial)  ->  sync  ->  LLM (exact prefix reuse or T_llm)  ->  stats / MAC row  ->  commit state
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from . import llm
from .config import MethodConfig
from .macs import llm_macs_rows, llm_macs_suffix, vit_macs
from .model import Grid, Model, grid
from .spatial import keep_mask
from .temporal import FramePlan, StreamState, commit, plan_frame
from .vit import run_vit


@dataclass
class FrameInputs:
    inputs: dict                        # processor output (CPU tensors)
    keep_mask: Optional[np.ndarray]     # S keep mask (None = all blocks)


def preprocess(m: Model, image_path: str, cfg: MethodConfig) -> FrameInputs:
    """Image file -> processor tensors (+ S keep mask). CPU only, so it can run in DataLoader workers."""
    emb = m.embedder
    inputs = emb.preprocess([emb.format_model_input(image=image_path)])
    return FrameInputs(inputs, keep_mask(cfg.spatial, inputs["pixel_values"], inputs["image_grid_thw"], cfg.tau))


class Engine:
    """One engine per stream and config (own caches)."""

    def __init__(self, m: Model, cfg: MethodConfig):
        self.m, self.cfg = m, cfg
        self.state: Optional[StreamState] = None

    @torch.inference_mode()
    def step(self, frame: FrameInputs, first: bool, timings: Optional[dict] = None) -> tuple[torch.Tensor, dict]:
        """Embed one frame. first=True starts a new stream (drops all reuse state). Returns (embedding (2048,), stats row)."""
        m, cfg = self.m, self.cfg
        inputs = {k: v.to(m.device) for k, v in frame.inputs.items()}
        g = grid(m, inputs)
        pixels = inputs["pixel_values"].float()
        block_pixels = pixels.reshape(g.n_blocks, -1)

        state = None
        if cfg.temporal_vit:
            if self.state is None or self.state.key != g.key:   # new resolution / prompt: fresh caches
                self.state = StreamState(m, g, approx_llm=cfg.approx_llm)
                first = True
            state = self.state
            if first:
                state.reset(m, g)

        plan = plan_frame(cfg, m, g, state, block_pixels, first, frame.keep_mask)
        llm.plan_llm(g, plan, cfg.approx_llm)
        visual_tokens, deepstack = run_vit(m, g, plan, pixels, block_pixels, cfg.linear_patch_embed)
        m.sync()
        if timings is not None:
            timings["t_vit_done"] = time.perf_counter()
        run_llm = llm.run_approx if cfg.approx_llm else llm.run_exact
        emb = run_llm(m, g, plan, visual_tokens, deepstack)
        row = stats_row(m, g, plan, cfg.approx_llm)
        commit(plan, emb)
        return emb, row


def stats_row(m: Model, g: Grid, plan: FramePlan, approx_llm: bool) -> dict:
    """Per-frame work counters and MACs."""
    nk, nc = len(plan.kept), len(plan.recompute)
    L = g.llm_len(nk)
    P = L if plan.skip_llm else plan.prefix
    if plan.skip_llm:
        n_rows, lm = 0, 0
    elif approx_llm:
        n_rows, lm = len(plan.llm_rows), llm_macs_rows(m, plan.llm_rows)
    else:
        n_rows, lm = L - P, llm_macs_suffix(m, P, L)
    return dict(n_kept=nk, n_comp_blocks=nc, full=bool(plan.refresh), changed_frac=plan.changed_frac,
                vit_q_patches=4 * nc, vit_kv_patches=4 * nk, vit_macs=vit_macs(m, 4 * nc, 4 * nk, nc) if nc else 0,
                llm_comp_tokens=n_rows, llm_len=L, llm_macs=lm, since_full=plan.since_refresh, llm_skip=bool(plan.skip_llm),
                llm_P=int(P))
