"""Multiply-accumulate (MAC) count of the work actually done per frame (weights and attention; norms / activations ignored).

All results are Python ints (no overflow). MAC fraction = (vit_macs + llm_macs) / full_frame_macs(m, grid).
"""
from __future__ import annotations

import numpy as np

from .model import Grid, Model


def vit_macs(m: Model, n_query: int, n_kv: int, n_merge: int) -> int:
    """ViT with n_query query patches attending over n_kv patches in every layer; n_merge blocks through merger + DeepStack mergers."""
    cv = m.vcfg
    d, inter = cv.hidden_size, cv.intermediate_size
    mh = d * cv.spatial_merge_size ** 2
    per_layer = n_query * (4 * d * d + 2 * d * inter) + 2 * n_query * n_kv * d
    patch_embed = n_query * cv.in_channels * cv.temporal_patch_size * cv.patch_size ** 2 * d
    mergers = n_merge * (len(m.deepstack_layers) + 1) * (mh * mh + mh * cv.out_hidden_size)
    return len(m.vit.blocks) * per_layer + patch_embed + mergers


def _llm_linear_per_token(m: Model) -> int:
    tc = m.tcfg
    H, nh, nk, hd, inter = tc.hidden_size, tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim, tc.intermediate_size
    return 2 * H * nh * hd + 2 * H * nk * hd + 3 * H * inter       # q,o + k,v + gate,up,down


def llm_macs_suffix(m: Model, P: int, L: int) -> int:
    """Exact path: prefill positions [P, L); position i attends over i+1 positions."""
    tc = m.tcfg
    att = 2 * tc.num_attention_heads * tc.head_dim * (L * (L + 1) - P * (P + 1)) // 2
    return tc.num_hidden_layers * ((L - P) * _llm_linear_per_token(m) + att)


def llm_macs_rows(m: Model, rows: np.ndarray) -> int:
    """KV Reuse (LLM) path: only positions `rows` (the host indices that were executed); row i attends over i+1 positions."""
    tc = m.tcfg
    rows = np.asarray(rows, dtype=np.int64)
    att = 2 * tc.num_attention_heads * tc.head_dim * int((rows + 1).sum())
    return tc.num_hidden_layers * (len(rows) * _llm_linear_per_token(m) + att)


def full_frame_macs(m: Model, g: Grid) -> int:
    """FULL config on this resolution (3.084e12 for 1280x704 / 880 blocks)."""
    return vit_macs(m, g.n_patches, g.n_patches, g.n_blocks) + llm_macs_suffix(m, 0, g.length)
