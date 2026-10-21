"""Step 9: the final train / test split of the paper -> questions_train.jsonl, t1_test_uids.json, questions_test_single.jsonl,
questions_test_parallel.jsonl.

train      the retrieval LoRA is trained on 1,000 of the 2,837 T1-train questions of s3: every web question (960) + desktop questions
           sampled to the test periods' desktop frame share (4 % -> 40; numpy default_rng(1), in s3 file order). Category = the diary
           category of the question's first gold recording.
T1 test    the T1-train recordings that are no question's first gold recording after this cut (433, all desktop) join the test set
           with the questions of the T1 question run (s5-s7 and p1-p4 on person A's T1 events; see the README): single-hop questions
           with a content cue (content / mixed), all gold recordings in T1-train and the first one in the 433 (690); parallel questions
           with every evidence recording in the 433 (431). Their qids get the prefix "T1:" (68 T1 parallel qids equal T2-T5 qids).
parallel   freq2 questions are dropped (T2-T5: 497 -> 415).
Output order: T2-T5 questions first, then the T1 ones. T1-val (79 recordings, 315 questions) stays validation; the pool is unchanged.

example: python data_pipeline/s9_final_split.py --train-all questions_train_all.jsonl --diary diary.jsonl --splits splits.json \
             --single questions_test_single_T2T5.jsonl --parallel questions_test_parallel_T2T5.jsonl \
             --t1-single t1/questions_single.jsonl --t1-parallel t1/questions_parallel.jsonl --train-out questions_train.jsonl \
             --t1-test-out t1_test_uids.json --single-out questions_test_single.jsonl --parallel-out questions_test_parallel.jsonl
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_jsonl, write_jsonl  # noqa: E402

DESK_SHARE = 0.04
DROP_TYPES = {"freq2"}


def uids(q):
    return [u for g in q["evidence_groups"] for u in g["uids"]]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-all", required=True, help="s3 T1-train questions (2,837)")
    p.add_argument("--diary", required=True, help="s2 diary.jsonl (recording category)")
    p.add_argument("--splits", required=True, help="s1 splits.json (t1_train_uids)")
    p.add_argument("--single", required=True, help="s7 single-hop test questions, T2-T5")
    p.add_argument("--parallel", required=True, help="p4 parallel test questions, T2-T5")
    p.add_argument("--t1-single", required=True, help="s7 output of the T1 question run")
    p.add_argument("--t1-parallel", required=True, help="p4 output of the T1 question run")
    p.add_argument("--train-out", required=True)
    p.add_argument("--t1-test-out", required=True)
    p.add_argument("--single-out", required=True)
    p.add_argument("--parallel-out", required=True)
    a = p.parse_args()

    cat = {u: d["category"] for d in read_jsonl(a.diary) for u in d["uids"]}
    rows = read_jsonl(a.train_all)
    web = [i for i, r in enumerate(rows) if cat.get(r["gold"][0]) == "web"]
    desk = [i for i, r in enumerate(rows) if cat.get(r["gold"][0]) != "web"]
    nd = int(round(DESK_SHARE / (1 - DESK_SHARE) * len(web)))
    keep = set(web) | {desk[i] for i in np.random.default_rng(1).choice(len(desk), nd, replace=False)}
    train = [r for i, r in enumerate(rows) if i in keep]
    write_jsonl(a.train_out, train)

    t1_train = set(json.load(open(a.splits))["t1_train_uids"])
    t1_test = sorted(t1_train - {r["gold"][0] for r in train})
    json.dump(t1_test, open(a.t1_test_out, "w"))
    T = set(t1_test)

    t1s = [q for q in read_jsonl(a.t1_single)
           if q["cue_type"] in ("content", "mixed") and set(uids(q)) <= t1_train and uids(q)[0] in T]
    t1p = [q for q in read_jsonl(a.t1_parallel) if q["type"] not in DROP_TYPES and set(uids(q)) <= T]
    single = read_jsonl(a.single) + [{**q, "qid": "T1:" + q["qid"]} for q in t1s]
    par = [q for q in read_jsonl(a.parallel) if q["type"] not in DROP_TYPES] + [{**q, "qid": "T1:" + q["qid"]} for q in t1p]
    assert len({q["qid"] for q in single}) == len(single) and len({q["qid"] for q in par}) == len(par)
    write_jsonl(a.single_out, single)
    write_jsonl(a.parallel_out, par)
    print(f"train {len(train)} of {len(rows)} (web {len(web)}, desktop {nd} of {len(desk)}) | T1 test recordings {len(t1_test)} "
          f"of {len(t1_train)}")
    print(f"single {len(single)} (T1 {len(t1s)} on {len({uids(q)[0] for q in t1s})} recordings) | parallel {len(par)} (T1 {len(t1p)}) |",
          dict(collections.Counter(q["type"] for q in par)))


if __name__ == "__main__":
    main()
