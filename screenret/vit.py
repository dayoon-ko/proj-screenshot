"""ViT forward over a subset of blocks: plain (FULL / S) or partial with cached K/V (T_vit).

Re-implements the HF Qwen3-VL vision forward layer by layer (same modules and weights) so that only some patches are queries:
  * S: only the patches of kept blocks enter (changed: stock HF always runs every patch).
  * T_vit: only patches of recomputed blocks are queries in EVERY layer; their fresh post-RoPE K/V are written into the stream's cache and
    they attend over the cached K/V of all kept blocks. Unchanged blocks are never run; their merger / DeepStack outputs come from a
    per-block cache. Attention in the Qwen3-VL ViT is full (non-causal) over the image, so unchanged blocks do not see the change: this is
    the approximation of T_vit.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl import modeling_qwen3_vl as MQ

from .model import Grid, Model, embed_patches
from .temporal import FramePlan, to_device


def patch_rows(blocks: np.ndarray) -> np.ndarray:
    """Patch row ids of blocks (each block = 4 consecutive patch rows in the processor's order)."""
    return (blocks[:, None] * 4 + np.arange(4)).reshape(-1)


@torch.inference_mode()
def run_vit(m: Model, g: Grid, plan: FramePlan, pixels: torch.Tensor, block_pixels: torch.Tensor, linear_patch_embed: bool):
    """Returns (visual tokens of the kept blocks, list of DeepStack features of the kept blocks), both in kept order."""
    st = plan.state
    if len(plan.recompute) == 0:                        # nothing changed: everything from the cache
        return _from_cache(m, plan)
    V = m.vit
    q_rows = to_device(patch_rows(plan.recompute), m)
    partial = st is not None and len(plan.recompute) < len(plan.kept)
    kv_rows = to_device(patch_rows(plan.kept), m) if partial else None
    x = embed_patches(m, pixels[q_rows].to(m.dtype), linear_patch_embed)
    x = x + g.vit_pos[q_rows].to(x.dtype)
    cos, sin = g.vit_cos[q_rows], g.vit_sin[q_rows]
    n = x.shape[0]
    deepstack = []
    for layer, blk in enumerate(V.blocks):
        q, k, v = KERNELS["vit_qkv"](blk, x, cos, sin, m.vit_heads, m.vit_head_dim)
        if st is not None:                              # changed: keep fresh K/V for later frames (also on a refresh)
            st.vit_k[layer][q_rows] = k
            st.vit_v[layer][q_rows] = v
            if partial:                                 # changed: attend over all kept blocks, unchanged ones from the cache
                k, v = st.vit_k[layer][kv_rows], st.vit_v[layer][kv_rows]
        # changed: one SDPA call over a single image (no cu_seqlens varlen path); identical for batch-1 inputs
        o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                                           dropout_p=0.0, scale=blk.attn.scaling, is_causal=False)
        x = KERNELS["vit_out"](blk, x, o, n)
        if layer in m.deepstack_layers:
            deepstack.append(V.deepstack_merger_list[m.deepstack_layers.index(layer)](x))
    merged = V.merger(x)
    if st is None:
        return merged, deepstack
    c = plan.recompute_t
    st.merged[c] = merged
    for i in range(len(deepstack)):
        st.deepstack[i][c] = deepstack[i]
    st.valid[c] = True
    st.ref[c] = block_pixels[c]
    return _from_cache(m, plan)


def _vit_qkv(blk, x, cos, sin, heads: int, head_dim: int):
    """norm1 -> qkv projection -> 2-D RoPE (the dense part before attention)."""
    h = blk.norm1(x)
    q, k, v = blk.attn.qkv(h).reshape(-1, 3, heads, head_dim).permute(1, 0, 2, 3).unbind(0)
    q, k = MQ.apply_rotary_pos_emb_vision(q, k, cos, sin)
    return q, k, v


def _vit_out(blk, x, o, n: int):
    """attention output projection + residual, then norm2 -> MLP + residual (the dense part after attention)."""
    x = x + blk.attn.proj(o[0].transpose(0, 1).reshape(n, -1))
    return x + blk.mlp(blk.norm2(x))


# changed: the dense per-block parts are looked up here so cpu_compile.enable() can swap in torch.compile'd versions (same ops, same order)
KERNELS = {"vit_qkv": _vit_qkv, "vit_out": _vit_out}


def _from_cache(m: Model, plan: FramePlan):
    st, kept = plan.state, plan.kept_t
    return st.merged[kept], [st.deepstack[i][kept] for i in range(len(m.deepstack_layers))]
