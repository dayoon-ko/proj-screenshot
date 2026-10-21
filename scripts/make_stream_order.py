"""Stream order for embed_stream: one person's frames as ONE continuous stream in time order -> "<fid>\t<rel_path>" lines.

Order = the split file's periods in order, each period's recordings in its listed (start-time) order, frames by index i. No frames are
dropped here (bad frames are excluded only from the evaluation pool), so the reuse caches see the real stream.

inputs:  --splits     JSON with "periods": [{"name", "uids": [...]}, ...] and "fids": {period name: [frame ids]}
         --frames-meta JSONL, one {"fid", "uid", "i", "rel_path"} per frame
example: python scripts/make_stream_order.py --splits splits.json --frames-meta frames.jsonl --periods T1,T2,T3,T4,T5 --out order.tsv
"""
import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", required=True)
    p.add_argument("--frames-meta", required=True)
    p.add_argument("--periods", default="T1,T2,T3,T4,T5", help="periods whose frames form the stream")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    s = json.load(open(a.splits))
    frames = {r["fid"]: r for r in map(json.loads, open(a.frames_meta))}
    names = a.periods.split(",")
    fids = sorted({f for k in names for f in s["fids"][k]})
    rec = {}
    for f in fids:
        rec.setdefault(frames[f]["uid"], []).append(f)
    uorder = [u for per in s["periods"] for u in per["uids"]]
    assert sorted(uorder) == sorted(rec), "periods do not cover exactly the recordings of the selected frames"
    order = [f for u in uorder for f in sorted(rec[u], key=lambda f: frames[f]["i"])]
    assert sorted(order) == fids
    with open(a.out, "w") as fo:
        fo.writelines(f"{f}\t{frames[f]['rel_path']}\n" for f in order)
    print(f"{len(order)} frames, {len(uorder)} recordings -> {a.out}")


if __name__ == "__main__":
    main()
