"""Merge embed_stream segment files into one id-indexed frame matrix (rows = frame ids, unused rows = 0) + a stats summary.

example: python -m screenret.merge_segments runs/LT8TL_seg*.npz --n-rows 80459 --out runs/LT8TL_frames.npy
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .embed_stream import stream_summary


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m screenret.merge_segments", description=__doc__)
    p.add_argument("parts", nargs="+", help="segment .npz files from embed_stream")
    p.add_argument("--n-rows", type=int, required=True, help="rows of the output matrix (> max frame id)")
    p.add_argument("--out", required=True, help="output .npy (fp16); the stats summary goes to <out>.json")
    a = p.parse_args(argv)
    zs = sorted((np.load(f) for f in a.parts), key=lambda z: json.loads(str(z["meta"]))["positions"][0])
    metas = [json.loads(str(z["meta"])) for z in zs]
    assert len({json.dumps(m["method"], sort_keys=True) for m in metas}) == 1, "segments of different methods"
    pos = [m["positions"] for m in metas]
    assert all(pos[i][1] == pos[i + 1][0] for i in range(len(pos) - 1)), f"segments not contiguous: {pos}"
    if pos[0][0] != 0 or pos[-1][1] != metas[0]["n_stream"]:
        print(f"WARNING: segments cover stream positions {pos[0][0]}..{pos[-1][1]} of {metas[0]['n_stream']}")
    ids = np.concatenate([z["ids"] for z in zs])
    assert len(set(ids.tolist())) == len(ids), "duplicate frame ids"
    X = np.zeros((a.n_rows, zs[0]["emb"].shape[1]), np.float16)
    X[ids] = np.concatenate([z["emb"] for z in zs])
    np.save(a.out, X)
    stats = np.concatenate([z["stats"] for z in zs])
    m0 = metas[0]
    summary = dict(method=m0["method"], name=m0["name"], n_segments=len(zs), n_frames=len(ids), model=m0["model"],
                   summary=stream_summary(stats, m0["full_frame_macs"], m0["n_blocks"], m0["full_len"]))
    json.dump(summary, open(a.out[:-4] + ".json", "w"), indent=1)
    print(json.dumps(summary["summary"]), "->", a.out)


if __name__ == "__main__":
    main()
