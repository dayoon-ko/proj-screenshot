"""S: spatial pruning of 32x32-px blocks before the ViT. Dropped blocks enter neither the ViT nor the LLM.

Screens are full of flat or repeated regions (backgrounds, margins, table fills). Three rules over the processor's pixel values:

  lt          ("LT8") drop a block iff it is within tau (max abs, every pixel) of its LEFT or its UPPER neighbour block, comparing
              original pixels. Every block is decided independently -> a few vectorised array ops.
  lt-grid     (paper default, "LT8t2" = lt + Grid Keep) lt, then one block per flat region per 2x2-block tile is kept again, so a large
              flat region keeps a sparse grid of blocks instead of only its first block. Kept blocks are a superset of lt's.
  pixelprune  PixelPrune Pred-2D: raster-order scan with a lossless-coding predictor from the reconstructed left / up / up-left
              neighbours (kept for comparison; sequential, slower on the CPU).

Masks are computed on the host (numpy) from the CPU processor output.
"""
from __future__ import annotations

import numpy as np
import torch


def _blocks(pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> np.ndarray:
    """(blocks_h, blocks_w, 4*1536) view of the processor output (patches are ordered block by block, 4 per block)."""
    bh, bw = int(grid_thw[0, 1]) // 2, int(grid_thw[0, 2]) // 2
    return np.asarray(pixel_values, dtype=np.float32).reshape(bh, bw, -1)


def lt_keep_mask(pixel_values: torch.Tensor, grid_thw: torch.Tensor, tau: float) -> np.ndarray:
    """Keep mask (raster order) of the order-free left-or-upper rule. At least one block is always kept."""
    blocks = _blocks(pixel_values, grid_thw)
    drop = np.zeros(blocks.shape[:2], bool)
    drop[:, 1:] |= np.abs(blocks[:, 1:] - blocks[:, :-1]).max(-1) <= tau
    drop[1:, :] |= np.abs(blocks[1:] - blocks[:-1]).max(-1) <= tau
    keep = ~drop
    if not keep.any():          # a perfectly uniform screen: keep the top-left block so the LLM still sees one visual token
        keep[0, 0] = True
    return keep.reshape(-1)


def pixelprune_keep_mask(pixel_values: torch.Tensor, grid_thw: torch.Tensor, tau: float) -> np.ndarray:
    """Keep mask (raster order) of PixelPrune Pred-2D. tau == 0 means bit-exact equality. The first block is always kept.

    Prediction per block (a = left, b = up, c = up-left, all reconstructed):
        if c == b and a != b: predict a    elif c == a and a != b: predict b    else: predict a
    First row predicts from the left, first column from above. A dropped block's reconstruction is its prediction.
    """
    blocks = _blocks(pixel_values, grid_thw)
    bh, bw = blocks.shape[:2]
    rec = blocks.copy()
    keep = np.ones((bh, bw), bool)
    if tau == 0:
        eq = np.array_equal
    else:
        def eq(x, y):
            return float(np.abs(x - y).max()) <= tau
    for r in range(bh):
        for c in range(bw):
            if r == 0 and c == 0:
                continue
            if r == 0:
                pred = rec[0, c - 1]
            elif c == 0:
                pred = rec[r - 1, 0]
            else:
                a, b, cc = rec[r, c - 1], rec[r - 1, c], rec[r - 1, c - 1]
                cb, ca, ab = eq(cc, b), eq(cc, a), eq(a, b)
                pred = a if (cb and not ab) else b if (ca and not ab) else a
            if eq(blocks[r, c], pred):
                keep[r, c] = False
                rec[r, c] = pred
    return keep.reshape(-1)


def _flat_components(eq_left: np.ndarray, eq_up: np.ndarray) -> np.ndarray:
    """Connected components of the lt equality graph (edge to the left block where eq_left, to the upper block where eq_up).
    One label per block = the smallest raster index in its component (min-label propagation along the edges + pointer jumping)."""
    bh, bw = eq_left.shape
    lab = np.arange(bh * bw).reshape(bh, bw)
    hl, vu = eq_left[:, 1:], eq_up[1:]
    while True:
        old = lab.copy()
        m = np.minimum(lab[:, 1:], lab[:, :-1])
        lab[:, 1:] = np.where(hl, m, lab[:, 1:])
        lab[:, :-1] = np.where(hl, np.minimum(lab[:, :-1], m), lab[:, :-1])
        m = np.minimum(lab[1:], lab[:-1])
        lab[1:] = np.where(vu, m, lab[1:])
        lab[:-1] = np.where(vu, np.minimum(lab[:-1], m), lab[:-1])
        flat = lab.reshape(-1)
        lab = flat[flat].reshape(bh, bw)
        if np.array_equal(lab, old):
            return lab


def gridkeep_keep_mask(pixel_values: torch.Tensor, grid_thw: torch.Tensor, tau: float, tile: int = 2) -> np.ndarray:
    """Keep mask (raster order) of lt + Grid Keep. The flat components of lt (connected through its left / upper equalities) are
    cut into tile x tile block tiles; every (component, tile) group without a kept block keeps its raster-first (top-left) block,
    which is encoded normally at its own position."""
    blocks = _blocks(pixel_values, grid_thw)
    bh, bw = blocks.shape[:2]
    eq_left = np.zeros((bh, bw), bool)
    eq_up = np.zeros((bh, bw), bool)
    eq_left[:, 1:] = np.abs(blocks[:, 1:] - blocks[:, :-1]).max(-1) <= tau
    eq_up[1:, :] = np.abs(blocks[1:] - blocks[:-1]).max(-1) <= tau
    keep = (~(eq_left | eq_up)).reshape(-1)              # = lt_keep_mask; block (0, 0) is always kept
    lab = _flat_components(eq_left, eq_up)
    rr, cc = np.meshgrid(np.arange(bh), np.arange(bw), indexing="ij")
    ntc = -(-bw // tile)
    group = (lab * (-(-bh // tile) * ntc) + (rr // tile) * ntc + cc // tile).reshape(-1)   # (component, tile) id
    orphan = np.flatnonzero(~keep & ~np.isin(group, group[keep]))   # dropped blocks of groups without a kept block, raster order
    if len(orphan):
        _, first = np.unique(group[orphan], return_index=True)      # raster-first member of each such group
        keep[orphan[first]] = True
    return keep


RULES = {"lt": lt_keep_mask, "lt-grid": gridkeep_keep_mask, "pixelprune": pixelprune_keep_mask}


def keep_mask(rule: str, pixel_values: torch.Tensor, grid_thw: torch.Tensor, tau: float):
    """Keep mask of `rule`, or None for rule "none" (all blocks kept)."""
    return None if rule == "none" else RULES[rule](pixel_values, grid_thw, tau)
