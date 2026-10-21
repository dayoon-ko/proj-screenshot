"""S rules on synthetic block grids (no model needed)."""
import numpy as np
import torch

from screenret.config import LEVEL
from screenret.spatial import gridkeep_keep_mask, lt_keep_mask, pixelprune_keep_mask

BH, BW, D = 4, 5, 4 * 1536
THW = torch.tensor([[1, 2 * BH, 2 * BW]])


def grid_pixels(blocks: np.ndarray) -> torch.Tensor:
    """(BH, BW, D) block values -> processor-shaped (n_patches, 1536) tensor (4 patch rows per block)."""
    return torch.from_numpy(blocks.reshape(-1, 1536).astype(np.float32))


def lt_reference(blocks: np.ndarray, tau: float) -> np.ndarray:
    """Loop version of the left-or-upper rule."""
    keep = np.ones((BH, BW), bool)
    for r in range(BH):
        for c in range(BW):
            left = c > 0 and np.abs(blocks[r, c] - blocks[r, c - 1]).max() <= tau
            up = r > 0 and np.abs(blocks[r, c] - blocks[r - 1, c]).max() <= tau
            keep[r, c] = not (left or up)
    if not keep.any():
        keep[0, 0] = True
    return keep.reshape(-1)


def test_lt_matches_loop_reference():
    rng = np.random.default_rng(0)
    for _ in range(50):
        base = rng.integers(0, 3, (BH, BW, 1)).astype(np.float32) * 20 * LEVEL       # few distinct flat blocks
        blocks = base + rng.integers(-1, 2, (BH, BW, D)) * rng.choice([0, 6, 12]) * LEVEL
        px = grid_pixels(blocks)
        assert np.array_equal(lt_keep_mask(px, THW, 8 * LEVEL), lt_reference(blocks, 8 * LEVEL))


def test_uniform_screen_keeps_one_block():
    px = grid_pixels(np.zeros((BH, BW, D)))
    for rule in (lt_keep_mask, pixelprune_keep_mask):
        keep = rule(px, THW, 8 * LEVEL)
        assert keep.sum() == 1 and keep[0]


def test_distinct_blocks_are_all_kept():
    blocks = np.arange(BH * BW, dtype=np.float32).reshape(BH, BW, 1).repeat(D, 2) * 20 * LEVEL
    px = grid_pixels(blocks)
    assert lt_keep_mask(px, THW, 8 * LEVEL).all() and pixelprune_keep_mask(px, THW, 8 * LEVEL).all()


def test_tau_is_inclusive_in_grey_levels():
    blocks = np.zeros((BH, BW, D), np.float32)
    blocks[:, 1::2] += 8 * LEVEL                                                  # columns alternate by exactly 8 levels
    px = grid_pixels(blocks)
    assert lt_keep_mask(px, THW, 8 * LEVEL).sum() == 1                           # every block within tau of a neighbour
    assert lt_keep_mask(px, THW, 7 * LEVEL).sum() == BW                          # only the first row survives


def gridkeep_reference(blocks: np.ndarray, tau: float, tile: int = 2) -> np.ndarray:
    """Loop version of lt + Grid Keep: flat components by BFS over the left / upper equalities, then per (component, tile) group
    without a kept block keep its raster-first block."""
    eq = lambda a, b: np.abs(blocks[a] - blocks[b]).max() <= tau
    keep = lt_reference(blocks, tau).reshape(BH, BW)
    comp = -np.ones((BH, BW), int)
    for r0 in range(BH):
        for c0 in range(BW):
            if comp[r0, c0] >= 0:
                continue
            comp[r0, c0], todo = r0 * BW + c0, [(r0, c0)]
            while todo:
                r, c = todo.pop()
                for rr, cc, linked in ((r, c - 1, c > 0 and eq((r, c), (r, c - 1))), (r, c + 1, c + 1 < BW and eq((r, c + 1), (r, c))),
                                       (r - 1, c, r > 0 and eq((r, c), (r - 1, c))), (r + 1, c, r + 1 < BH and eq((r + 1, c), (r, c)))):
                    if linked and comp[rr, cc] < 0:
                        comp[rr, cc] = comp[r0, c0]
                        todo.append((rr, cc))
    out = keep.copy()
    groups = {}
    for r in range(BH):
        for c in range(BW):
            groups.setdefault((comp[r, c], r // tile, c // tile), []).append((r, c))
    for members in groups.values():
        if not any(keep[m] for m in members):
            out[members[0]] = True
    return out.reshape(-1)


def test_gridkeep_matches_loop_reference_and_is_lt_superset():
    rng = np.random.default_rng(1)
    for _ in range(50):
        base = rng.integers(0, 3, (BH, BW, 1)).astype(np.float32) * 20 * LEVEL
        blocks = base + rng.integers(-1, 2, (BH, BW, D)) * rng.choice([0, 6, 12]) * LEVEL
        px = grid_pixels(blocks)
        g, lt = gridkeep_keep_mask(px, THW, 8 * LEVEL), lt_keep_mask(px, THW, 8 * LEVEL)
        assert np.array_equal(g, gridkeep_reference(blocks, 8 * LEVEL))
        assert not (lt & ~g).any()


def test_gridkeep_uniform_screen_keeps_one_block_per_tile():
    keep = gridkeep_keep_mask(grid_pixels(np.zeros((BH, BW, D))), THW, 8 * LEVEL).reshape(BH, BW)
    assert keep.sum() == -(-BH // 2) * -(-BW // 2) and keep[::2, ::2].all()
