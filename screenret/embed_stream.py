"""Accuracy-side embedding: run the method over a screen stream in time order and store one vector per frame.

The stream is CONTINUOUS: no reset at recording boundaries; only the first frame (of the stream or of a segment) is a forced full compute;
afterwards refreshes come from the change detector (rho), the period K, or a new frame resolution. For parallel jobs the stream is cut
into --segment i/n contiguous pieces (the paper's stored runs used 4); each segment starts with a full compute.

--frames-list: text file, one frame per line in stream order, "<id>\t<path>" or "<path>" (id = line number). Paths are relative to
--frames-root unless absolute. Build it with scripts/make_stream_order.py.

Output (--out X.npz): emb (n, 2048) fp16, ids (n,), stats (n, k) + stat column names, method / model metadata. Merge segments into one
id-indexed matrix with `python -m screenret.merge_segments`.

example: python -m screenret.embed_stream --device cuda --preset S+G+T+L --lora adapter.pt --frames-list order.tsv --frames-root FRAMES \
             --segment 0/4 --out runs/LT8t2TL_seg0.npz
"""
from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

from .cli import add_cpu_speed_args, add_method_args, add_model_args, load_from_args, method_from_args, model_meta
from .engine import Engine, preprocess
from .macs import full_frame_macs

STAT_COLS = ["n_kept", "n_comp_blocks", "full", "changed_frac", "vit_q_patches", "vit_kv_patches", "vit_macs", "llm_comp_tokens",
             "llm_len", "llm_macs", "since_full", "llm_skip", "llm_P"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m screenret.embed_stream", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_method_args(p)
    add_model_args(p)
    add_cpu_speed_args(p)
    g = p.add_argument_group("stream")
    g.add_argument("--frames-list", required=True, help="frames in stream order: '<id>\\t<path>' or '<path>' per line")
    g.add_argument("--frames-root", default=os.environ.get("SCREENRET_FRAMES", ""), help="root of relative paths ($SCREENRET_FRAMES)")
    g.add_argument("--segment", default="0/1", help="i/n: embed only the i-th of n contiguous pieces of the stream (default 0/1)")
    g.add_argument("--limit", type=int, default=0, help="embed only the first n frames of the segment (quick checks)")
    g.add_argument("--workers", type=int, default=3, help="DataLoader workers for image decoding + processor + Token Pruning mask")
    g.add_argument("--out", required=True, help="output .npz")
    return p.parse_args(argv)


def read_frames_list(path: str, root: str) -> tuple[np.ndarray, list[str]]:
    ids, paths = [], []
    for i, line in enumerate(open(path)):
        line = line.rstrip("\n")
        if not line:
            continue
        fid, p = line.split("\t", 1) if "\t" in line else (i, line)
        ids.append(int(fid))
        paths.append(p if os.path.isabs(p) else os.path.join(root, p))
    return np.array(ids, dtype=np.int64), paths


def segment_range(n: int, seg: str) -> tuple[int, int]:
    i, k = (int(x) for x in seg.split("/"))
    assert 0 <= i < k, seg
    return n * i // k, n * (i + 1) // k


class _Frames(torch.utils.data.Dataset):
    """CPU preprocessing in DataLoader workers; holds only the processor (no model weights)."""

    def __init__(self, m, cfg, paths):
        self.m, self.cfg, self.paths = SimpleNamespace(embedder=m.embedder.without_model()), cfg, paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, j):
        return preprocess(self.m, self.paths[j], self.cfg)


def frame_loader(m, cfg, paths, workers: int):
    return torch.utils.data.DataLoader(_Frames(m, cfg, paths), batch_size=None, num_workers=workers,
                                       prefetch_factor=16 if workers else None)


def stream_summary(stats: np.ndarray, full_macs: int, n_blocks: int, full_len: int) -> dict:
    col = {k: stats[:, i] for i, k in enumerate(STAT_COLS)}
    mean = lambda x: float(np.mean(x)) if len(x) else None   # noqa: E731
    return dict(n=len(stats), kept_block_frac=mean(col["n_kept"] / n_blocks), vit_patch_frac=mean(col["vit_q_patches"] / (4 * n_blocks)),
                llm_token_frac=mean(col["llm_comp_tokens"] / full_len), mac_frac=mean((col["vit_macs"] + col["llm_macs"]) / full_macs),
                mac_frac_vit=mean(col["vit_macs"] / full_macs), mac_frac_llm=mean(col["llm_macs"] / full_macs),
                refresh_rate=mean(col["full"]), llm_skip_rate=mean(col["llm_skip"]))


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = method_from_args(args)
    ids, paths = read_frames_list(args.frames_list, args.frames_root)
    lo, hi = segment_range(len(ids), args.segment)
    if args.limit:
        hi = min(hi, lo + args.limit)
    m = load_from_args(args)
    print(f"[{cfg.name}] segment {args.segment}: stream positions {lo}..{hi} of {len(ids)} | {model_meta(args, m)}", flush=True)
    eng = Engine(m, cfg)
    embs, rows = [], []
    t0 = time.time()
    for k, frame in enumerate(frame_loader(m, cfg, paths[lo:hi], args.workers)):
        e, row = eng.step(frame, first=(k == 0))
        embs.append(e.float().cpu().numpy().astype(np.float16))
        rows.append([float(row[c]) for c in STAT_COLS])
        if (k + 1) % 500 == 0:
            fps = (k + 1) / (time.time() - t0)
            print(f"  {k + 1}/{hi - lo} frames | {fps:.2f} frames/s | ETA {(hi - lo - k - 1) / fps / 3600:.2f} h", flush=True)
    stats = np.array(rows, dtype=np.float64).reshape(-1, len(STAT_COLS))
    g = next(iter(m.grids.values()))
    fm = full_frame_macs(m, g)
    meta = dict(method=vars(cfg), name=cfg.name, segment=args.segment, positions=[lo, hi], n_stream=len(ids), stat_cols=STAT_COLS,
                full_frame_macs=fm, n_blocks=g.n_blocks, full_len=g.length, model=model_meta(args, m),
                summary=stream_summary(stats, fm, g.n_blocks, g.length),
                wall_s=time.time() - t0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, emb=np.stack(embs), ids=ids[lo:hi], stats=stats, meta=json.dumps(meta))
    print(f"DONE {cfg.name} {args.segment}: {json.dumps(meta['summary'])} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
