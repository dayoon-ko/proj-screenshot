"""LLM part: token sequence, exact prefix reuse (with KV Reuse (ViT)) and KV Reuse (LLM).

Sequence = [prompt head (first image token at `first`)] + [one visual token per kept block] + [text tail incl. the final token].
The embedding is the normalised final hidden state of the last token.

Exact prefix reuse (on with KV Reuse): the LLM is causal, so positions before the first changed position P are bit-identical to the
previous frame. Crop the previous frame's KV cache to P and prefill only [P, L). If nothing changed, return the previous embedding.

KV Reuse (LLM) (approximate): recompute only R = visual tokens of changed blocks + every text token after the first change + the final token
(+ any slot without cached K/V). Unchanged visual tokens after the first change reuse their cached per-layer K/V although their causal
context changed: this is the approximation. Text after the change is always recomputed because it summarises the image.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_vl import modeling_qwen3_vl as MQ

from .model import Grid, Model
from .temporal import FramePlan, to_device


def full_grid_slots(g: Grid, kept: np.ndarray) -> np.ndarray:
    """Full-grid slot of each position of the (pruned) sequence."""
    return np.concatenate([np.arange(g.first), g.first + kept, np.arange(g.last + 1, g.length)])


def first_changed_position(g: Grid, plan: FramePlan) -> int:
    """P: length of the prefix identical to the previous frame's sequence (0 without history)."""
    st = plan.state
    if st is None or st.prev_kept is None:
        return 0
    kept, prev = plan.kept, st.prev_kept
    recomputed = np.zeros(g.n_blocks, bool)
    recomputed[plan.recompute] = True
    n = min(len(kept), len(prev))
    same = (kept[:n] == prev[:n]) & ~recomputed[kept[:n]]   # same block in the same slot and not recomputed
    j = int(np.argmin(same)) if not same.all() else n
    if j == n and len(kept) == len(prev):
        return g.llm_len(len(kept))                         # nothing changed
    return g.first + j


def plan_llm(g: Grid, plan: FramePlan, approx_llm: bool):
    """Fill plan.prefix, plan.skip_llm and (KV Reuse (LLM)) plan.llm_rows, on the host."""
    st = plan.state
    L = g.llm_len(len(plan.kept))
    plan.prefix = first_changed_position(g, plan)
    plan.skip_llm = st is not None and plan.prefix >= L and st.prev_emb is not None
    if approx_llm and not plan.skip_llm:
        plan.llm_rows = recompute_rows(g, plan)


def recompute_rows(g: Grid, plan: FramePlan) -> np.ndarray:
    """KV Reuse (LLM): R, the sorted positions to run through the LLM (always contains the final position L-1)."""
    f, kept, P = g.first, plan.kept, plan.prefix
    nk = len(kept)
    L = g.llm_len(nk)
    r = np.zeros(L, bool)
    if plan.refresh or P == 0:
        r[P:] = True
    else:
        changed = np.zeros(g.n_blocks, bool)
        changed[plan.recompute] = True
        r[f:f + nk] = changed[kept]                         # visual tokens of recomputed blocks
        r[max(P, f + nk):] = True                           # text after the first change
    r |= ~plan.state.llm_valid[full_grid_slots(g, kept)]   # slots never computed in this stream
    r[L - 1] = True
    return np.flatnonzero(r)


def llm_inputs(g: Grid, kept: np.ndarray, kept_t: torch.Tensor, visual_tokens: torch.Tensor, m: Model):
    """Input embeddings and mRoPE positions of the (pruned) sequence.
    changed: dropped blocks are removed from the sequence; the rest keep their full-grid mRoPE positions."""
    seq = torch.cat([torch.arange(g.first, device=m.device), g.first + kept_t, torch.arange(g.last + 1, g.length, device=m.device)])
    x = g.token_emb[seq].clone()
    x[g.first:g.first + len(kept)] = visual_tokens.to(x.dtype)
    return x, g.llm_pos[:, seq]


@torch.inference_mode()
def run_exact(m: Model, g: Grid, plan: FramePlan, visual_tokens: torch.Tensor, deepstack: list) -> torch.Tensor:
    """Stock HF LLM forward; with KV Reuse (ViT) it continues from the previous frame's KV cache cropped to P (exact)."""
    st = plan.state
    if plan.skip_llm:
        return st.prev_emb
    x, pos = llm_inputs(g, plan.kept, plan.kept_t, visual_tokens, m)
    L, f, nk = x.shape[0], g.first, len(plan.kept)
    P = plan.prefix
    if st is None or st.exact_kv is None or P == 0:
        kv, P = DynamicCache(config=m.tcfg), 0
    else:
        kv = st.exact_kv                                    # changed: reuse the previous frame's cache, cropped to P
        cur = kv.get_seq_length()
        if cur > P:
            kv.crop(-(cur - P))
    visual = torch.zeros(L, dtype=torch.bool, device=m.device)
    visual[f:f + nk] = True
    n_vis_in_prefix = min(max(P - f, 0), nk)
    hs = m.llm(input_ids=None, inputs_embeds=x[None, P:], position_ids=pos[:, None, P:], past_key_values=kv, use_cache=True,
               visual_pos_masks=visual[None, P:],
               deepstack_visual_embeds=[d[n_vis_in_prefix:].to(x.dtype) for d in deepstack]).last_hidden_state
    if st is not None:
        st.exact_kv = kv
    return F.normalize(hs[0, -1].float(), p=2, dim=-1)


@torch.inference_mode()
def run_approx(m: Model, g: Grid, plan: FramePlan, visual_tokens: torch.Tensor, deepstack: list) -> torch.Tensor:
    """Run only rows R through every LLM layer. Per layer, write their fresh K/V into the full-grid slot cache and attend causally
    (by sequence index) over the cached K/V of all current positions.
    changed: re-implements the Qwen3 decoder layer (same modules) with a row subset and a slot-indexed K/V cache."""
    st = plan.state
    if plan.skip_llm:
        return st.prev_emb
    tc = m.tcfg
    nh, nkv, hd = tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim
    f, nk = g.first, len(plan.kept)
    x, pos = llm_inputs(g, plan.kept, plan.kept_t, visual_tokens, m)
    rows_np = plan.llm_rows
    slots_np = full_grid_slots(g, plan.kept)
    is_vis = (rows_np >= f) & (rows_np < f + nk)
    rows, slots = to_device(rows_np, m), to_device(slots_np, m)
    row_slots = to_device(slots_np[rows_np], m)
    vis_rows = to_device(np.flatnonzero(is_vis), m)                         # rows (within R) that are visual tokens
    ds_rows = to_device(rows_np[is_vis] - f, m)                             # their index among the kept blocks
    h = x[rows]
    rpos = pos[:, rows][:, None]
    if rpos.shape[0] == 4:                                                  # (text, t, h, w) layout -> the 3 mRoPE rows
        rpos = rpos[1:]
    ds = [d[ds_rows].to(h.dtype) for d in deepstack]
    cos, sin = m.llm.rotary_emb(h, rpos)
    cos, sin = cos[0], sin[0]
    mask = (torch.arange(len(slots_np), device=m.device)[None, :] <= rows[:, None])[None, None]   # causal over sequence index
    for layer_id, layer in enumerate(m.llm.layers):
        at = layer.self_attn
        q, k, v = KERNELS["llm_qkv"](layer, h, cos, sin, nh, nkv, hd)
        st.llm_k[layer_id][row_slots] = k
        st.llm_v[layer_id][row_slots] = v
        kf, vf = st.llm_k[layer_id][slots], st.llm_v[layer_id][slots]
        o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], kf.transpose(0, 1)[None], vf.transpose(0, 1)[None],
                                           attn_mask=mask, dropout_p=0.0, scale=at.scaling, enable_gqa=True)
        h = KERNELS["llm_out"](layer, h, o, len(rows_np))
        if layer_id < len(ds) and len(vis_rows):                             # DeepStack: add ViT features after early layers
            h[vis_rows] = h[vis_rows] + ds[layer_id]
    st.llm_valid[slots_np[rows_np]] = True
    return F.normalize(m.llm.norm(h[-1]).float(), p=2, dim=-1)              # R is sorted and ends with the final position


def _llm_qkv(layer, h, cos, sin, nh: int, nkv: int, hd: int):
    """input RMSNorm -> q/k/v projections -> q/k RMSNorm -> mRoPE (the dense part before attention)."""
    at = layer.self_attn
    a = layer.input_layernorm(h)
    q = at.q_norm(at.q_proj(a).view(-1, nh, hd))
    k = at.k_norm(at.k_proj(a).view(-1, nkv, hd))
    v = at.v_proj(a).view(-1, nkv, hd)
    q, k = MQ.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
    return q, k, v


def _llm_out(layer, h, o, n: int):
    """o_proj + residual, then post-attention RMSNorm -> MLP + residual (the dense part after attention)."""
    h = h + layer.self_attn.o_proj(o[0].transpose(0, 1).reshape(n, -1))
    return h + layer.mlp(layer.post_attention_layernorm(h))


# changed: the dense per-layer parts of KV Reuse (LLM) are looked up here so cpu_compile.enable() can swap in torch.compile'd versions
KERNELS = {"llm_qkv": _llm_qkv, "llm_out": _llm_out}
