"""Step 1: time splits of the test person's history and the answer-revealing ("bad") frames -> splits.json.

Person A (one machine) is cut into K = 5 periods with equal frame counts by recording start time (T1..T5). Recordings with an identical
(start, stop) window are byte-identical twins: one is kept, the twin is mapped to it. T1 = training period, its last 10 % of recordings
(by time) = T1-val (model selection); T2..T5 = test periods. Frames whose OCR text reproduces the task instruction (the recording tool's
task panel) or shows the recording portal give the answer away; they are listed in "bad_fids" and excluded from every pool.

example: python data_pipeline/s1_splits.py --meta psai_tasks_meta.jsonl --frames-meta frames.jsonl --ocr ocr_all.jsonl --out splits.json
"""
import argparse
import collections
import json
import re
from datetime import datetime

import numpy as np


def squash(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def gives_away(text, instr_sq):
    """The frame's OCR text shows the recording portal, or >= 40 % (and >= 2) of the instruction's 30-char windows (stride 15)."""
    if re.search(r"paradigm\s*shift", text, re.I) and re.search(r"submit\s*a\s*task|select\s*task|upload\s*files|select\s*video", text, re.I):
        return True
    if len(instr_sq) < 40:
        return False
    T = squash(text)
    w = range(0, len(instr_sq) - 30, 15)
    return sum(1 for a in w if instr_sq[a:a + 30] in T) >= max(2, len(w) * 0.4)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meta", required=True, help='PSAI recording metadata JSONL {"unique_data_id", "node", "start_time", "stop_time", "task_name", ...}')
    p.add_argument("--frames-meta", required=True, help='JSONL {"fid", "uid", ...} per 5-s frame')
    p.add_argument("--ocr", required=True, help='JSONL {"id": fid, "text"} (RapidOCR text per frame)')
    p.add_argument("--node", default="DESKTOP-GFL3B1N", help="machine of the test person (person A)")
    p.add_argument("--periods", type=int, default=5)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    frames = [json.loads(l) for l in open(a.frames_meta)]
    meta = [json.loads(l) for l in open(a.meta)]
    M = {m["unique_data_id"]: m for m in meta}
    fids_of = collections.defaultdict(list)
    for f in frames:
        fids_of[f["uid"]].append(f["fid"])

    instr = {u: squash(m["task_name"] or "") for u, m in M.items()}
    bad = np.zeros(len(frames), bool)
    for l in open(a.ocr):
        r = json.loads(l)
        bad[r["id"]] = gives_away(r["text"], instr.get(frames[r["id"]]["uid"], ""))

    t = lambda s: datetime.fromisoformat(s[:19])
    keep, twin, seen = [], {}, {}
    for r in sorted([m for m in meta if m.get("node") == a.node], key=lambda m: (m["start_time"], m["stop_time"], m["unique_data_id"])):
        k = (r["start_time"], r["stop_time"])
        if k in seen:
            twin[r["unique_data_id"]] = seen[k]
        else:
            seen[k] = r["unique_data_id"]
            keep.append(r["unique_data_id"])
    keep.sort(key=lambda u: (t(M[u]["start_time"]), u))
    cum = np.cumsum([len(fids_of[u]) for u in keep])
    K, tot = a.periods, int(cum[-1])
    period = {u: min(K - 1, int((cum[i] - 1) * K // tot)) for i, u in enumerate(keep)}
    t1 = [u for u in keep if period[u] == 0]
    nval = round(len(t1) * a.val_frac)
    t1_train, t1_val = t1[:-nval], t1[-nval:]

    fl = lambda us: [f for u in us for f in fids_of[u]]
    by_k = [[u for u in keep if period[u] == k] for k in range(K)]
    out = {"K": K, "val_frac": a.val_frac, "person_A": a.node,
           "periods": [{"name": f"T{k + 1}", "uids": by_k[k], "from": M[by_k[k][0]]["start_time"], "to": M[by_k[k][-1]]["start_time"]}
                       for k in range(K)],
           "t1_train_uids": t1_train, "t1_val_uids": t1_val,
           "fids": {**{f"T{k + 1}": fl(by_k[k]) for k in range(K)}, "T1-train": fl(t1_train), "T1-val": fl(t1_val)},
           "bad_fids": np.where(bad)[0].tolist(), "twins_A": twin}
    json.dump(out, open(a.out, "w"))
    print(f"A: {len(keep)} recordings, {tot:,} frames, {len(twin)} twins merged | answer-revealing frames in A: {int(bad[fl(keep)].sum())}")
    for k in range(K):
        print(f"  T{k + 1}: {len(by_k[k])} recordings, {len(out['fids'][f'T{k + 1}']):,} frames, {out['periods'][k]['from'][:10]} .. {out['periods'][k]['to'][:10]}")
    print(f"  T1-train {len(t1_train)} / T1-val {len(t1_val)} recordings")


if __name__ == "__main__":
    main()
