"""Parallel step 4: the released parallel multi-hop set -> questions_test_parallel.jsonl.

Concatenates the kept parallel questions of p2 and of the p3 rounds (in the given order) and adds "variant" (= type for p2 questions)
and "leak" = (# question content words found in the OCR text of the gold recordings) / (# question content words), content word =
>= 4 letters and not a stop word; the OCR words of all evidence recordings are pooled. The paper uses every question (no leak filter);
scripts/make_eval_protocol.py --leak-filter can drop high-leak ones.

example: python data_pipeline/p4_parallel_set.py --multihop questions_multihop.jsonl --extra questions_multihop_extra.jsonl \
             questions_multihop_extra2.jsonl questions_multihop_extra3.jsonl --frames-meta frames.jsonl --ocr ocr_all.jsonl \
             --person A --out questions_test_parallel.jsonl
"""
import argparse
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_jsonl, write_jsonl  # noqa: E402

STOP = set("""what which when where how did that this with from have your into about after before were they them then than their there
these those would could should also just like only some such very more most other over under while what's i've used using look looked
find found check checked page site website app application time first last next does done make made want need tool search searched open opened""".split())


def content_words(t):
    return {w for w in re.findall(r"[a-z0-9]+", (t or "").lower()) if len(w) >= 4 and w not in STOP}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--multihop", required=True, help="p2 kept questions (parallel)")
    p.add_argument("--extra", nargs="*", default=[], help="p3 kept questions, one file per round, in round order")
    p.add_argument("--frames-meta", required=True, help='JSONL {"fid", "uid", ...} per frame')
    p.add_argument("--ocr", required=True, help='JSONL {"id": fid, "text"} (RapidOCR text per frame)')
    p.add_argument("--person", default="", help="keep only this person's questions (release: A); default all")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    uid = {f["fid"]: f["uid"] for f in read_jsonl(a.frames_meta)}
    ocr = collections.defaultdict(set)
    for r in read_jsonl(a.ocr):
        ocr[uid[r["id"]]] |= content_words(r["text"])
    Q = [q for q in read_jsonl(a.multihop) if q["structure"] == "parallel"]
    for fp in a.extra:
        Q += read_jsonl(fp)
    out = []
    for q in Q:
        if a.person and q["person"] != a.person:
            continue
        qw = content_words(q["question"])
        gw = set().union(*[ocr[u] for g in q["evidence_groups"] for u in g["uids"]])
        out.append({**q, "variant": q.get("variant") or q["type"], "leak": round(len(qw & gw) / len(qw), 3) if qw else 0.0})
    write_jsonl(a.out, out)
    print(f"{len(out)} parallel questions -> {a.out} |", dict(collections.Counter((q["person"], q["type"]) for q in out)))


if __name__ == "__main__":
    main()
