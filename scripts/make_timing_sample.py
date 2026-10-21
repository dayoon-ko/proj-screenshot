"""Timing sample for `python -m screenret.timing`: random windows of consecutive frames of ONE machine's continuous screen stream.

The machine's 5-s frames form one stream in time order (recordings concatenated by start time). The stream (or, with --test-fids, the
range from the first to the last test frame) is split into --n-windows equal segments with one random --len-frame window per segment;
with --test-fids every frame of a window must be a test frame. Each window carries the --ctx frames right before it (untimed history, so
the reuse caches start in the state the real stream has). Windows may cross recording boundaries, as the real stream does.

paper samples: --n-windows 100 --len 20 --ctx 10 --seed 20261013 (sample2000, all of PC A)
               --test-fids test_fids_T2T5.json --seed 20261015 (sample_test2000, test periods T2-T5)
example: python scripts/make_timing_sample.py --frames-meta frames.jsonl --tasks-meta tasks_meta.jsonl --node <machine id> --out sample.json
"""
import argparse
import json
import random


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames-meta", required=True, help='JSONL {"fid", "uid", "i", "rel_path"} per frame')
    p.add_argument("--tasks-meta", required=True, help='JSONL {"unique_data_id", "node", "start_time"} per recording')
    p.add_argument("--node", required=True, help="machine whose stream is sampled (tasks_meta 'node')")
    p.add_argument("--test-fids", default=None, help="optional JSON {period: [fid]}: windows only inside the test frames")
    p.add_argument("--n-windows", type=int, default=100)
    p.add_argument("--len", type=int, default=20)
    p.add_argument("--ctx", type=int, default=10)
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    meta = {r["unique_data_id"]: r for r in map(json.loads, open(a.tasks_meta))}
    frames = [f for f in map(json.loads, open(a.frames_meta)) if meta.get(f["uid"], {}).get("node") == a.node]
    frames.sort(key=lambda f: (meta[f["uid"]]["start_time"], f["uid"], f["i"]))
    if a.test_fids:
        test = {fid for v in json.load(open(a.test_fids)).values() for fid in v}
        ok = [f["fid"] in test for f in frames]
        pos = [i for i, t in enumerate(ok) if t]
        t0, seg = pos[0], (pos[-1] + 1 - pos[0]) // a.n_windows
    else:
        ok, t0, seg = [True] * len(frames), 0, len(frames) // a.n_windows
    rng = random.Random(a.seed)
    ref = lambda f: dict(fid=f["fid"], rel_path=f["rel_path"], uid=f["uid"])   # noqa: E731
    chunks = []
    for k in range(a.n_windows):
        while True:
            s = t0 + k * seg + rng.randrange(seg - a.len)
            if all(ok[s:s + a.len]):
                break
        chunks.append(dict(start=s, ctx=[ref(f) for f in frames[max(0, s - a.ctx):s]], frames=[ref(f) for f in frames[s:s + a.len]]))
    json.dump(dict(seed=a.seed, n_chunks=a.n_windows, chunk_len=a.len, ctx_len=a.ctx, node=a.node, test_only=bool(a.test_fids),
                   pool=f"{len(frames)}-frame continuous stream; one {a.len}-frame window per 1/{a.n_windows} segment", chunks=chunks),
              open(a.out, "w"), indent=1)
    print(f"{a.out}: {sum(len(c['frames']) for c in chunks)} timed + {sum(len(c['ctx']) for c in chunks)} context frames")


if __name__ == "__main__":
    main()
