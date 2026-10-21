"""Evaluation protocol JSON for `python -m screenret.evaluate` (the paper's protocol: single-hop + parallel multi-hop, unfiltered).

single-hop  rows of eval_input.json["new"] of --person whose qid starts with one of --single-prefix (paper: person A, all 4,265
            single-hop test questions = 3,575 of T2-T5 + 690 of the T1 test recordings, qids "T1:S-..."); gold = the question's
            recording(s) (groups[0]). --main-qids restricts them to a list (older 2,059-question table).
parallel    --parallel JSONL {"qid", "person", "question", "evidence_groups": [{"uids", "category"}], "leak"} of --person (paper:
            data/questions_test_parallel.jsonl, 846 person-A questions = 415 of T2-T5 + 431 of the T1 test recordings, no freq2);
            one group per evidence group.
            --leak-filter (off in the paper): drop questions with leak >= --leak-threshold unless every evidence group is a desktop
            recording and the person is in --desktop-exempt (default A).
pool        frames of the given periods (paper: T1-T5) minus splits.json "bad_fids" (paper: 63,538 frames); pool_group = recording id.

Query rows: with --queries-out, question i gets row i and its text is written to that JSONL ({"qid", "text"}; single-hop texts from
--texts, the row-indexed query texts of eval_input.json); embed it with `python -m screenret.embed_text queries --texts <queries-out>`.
Without --queries-out (single-hop only) "row" stays the eval_input row, for query vectors embedded in that order.

example: python scripts/make_eval_protocol.py --eval-input data/eval_input_test.json --texts data/texts_test.jsonl \
             --parallel data/questions_test_parallel.jsonl --splits data/splits.json --frames-meta frames.jsonl --out protocol.json \
             --queries-out protocol_queries.jsonl
"""
import argparse
import json


def keep_parallel(q: dict, threshold: float, exempt: set) -> bool:
    """Leak filter: keep a low-leak question, or a high-leak one whose evidence is all desktop recordings of an exempt person."""
    desktop = all(g.get("category") == "desktop" for g in q["evidence_groups"])
    return q.get("leak", 0.0) < threshold or (q["person"] in exempt and desktop)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-input", required=True, help='JSON with "new": [{"qid", "person", "row", "groups": [[uid, ...], ...]}]')
    p.add_argument("--texts", help='JSONL {"text"} per eval_input row (query texts); needed with --queries-out')
    p.add_argument("--parallel", help="JSONL of parallel multi-hop questions (omit: single-hop only)")
    p.add_argument("--main-qids", help="JSON list of qids: keep only these single-hop questions (default: all)")
    p.add_argument("--single-prefix", default="S-,T1:S-", help="qid prefixes of single-hop questions in eval_input, comma-separated "
                                                               "(default S-,T1:S-)")
    p.add_argument("--leak-filter", action="store_true", help="drop high-leak parallel questions (off = the paper's unfiltered set)")
    p.add_argument("--leak-threshold", type=float, default=0.65)
    p.add_argument("--desktop-exempt", default="A", help="persons whose all-desktop high-leak questions are kept (comma-separated)")
    p.add_argument("--splits", required=True, help='JSON with "fids": {period: [fid]} and "bad_fids"')
    p.add_argument("--frames-meta", required=True, help='JSONL {"fid", "uid", ...} per frame')
    p.add_argument("--person", default="A")
    p.add_argument("--periods", default="T1,T2,T3,T4,T5")
    p.add_argument("--out", required=True)
    p.add_argument("--queries-out", help="JSONL of the protocol's query texts (row i = question i)")
    a = p.parse_args()
    if a.parallel and not a.queries_out:
        p.error("--parallel needs --queries-out (parallel questions have no eval_input row)")
    if a.queries_out and not a.texts:
        p.error("--queries-out needs --texts for the single-hop query texts")

    s = json.load(open(a.splits))
    bad = set(s["bad_fids"])
    uid = {r["fid"]: r["uid"] for r in map(json.loads, open(a.frames_meta))}
    pool = sorted(f for k in a.periods.split(",") for f in s["fids"][k] if f not in bad)
    main_q = set(json.load(open(a.main_qids))) if a.main_qids else None
    texts = [json.loads(line)["text"] for line in open(a.texts)] if a.texts else None
    prefixes = tuple(a.single_prefix.split(","))

    qs, qtext = [], []
    for r in json.load(open(a.eval_input))["new"]:
        if r["person"] == a.person and r["qid"].startswith(prefixes) and (main_q is None or r["qid"] in main_q):
            qs.append(dict(qid=r["qid"], type="single", row=r["row"], groups=[r["groups"][0]]))
            qtext.append(texts[r["row"]] if texts else None)
    n_par_all = 0
    if a.parallel:
        exempt = set(a.desktop_exempt.split(","))
        for q in map(json.loads, open(a.parallel)):
            if q["person"] != a.person:
                continue
            n_par_all += 1
            if a.leak_filter and not keep_parallel(q, a.leak_threshold, exempt):
                continue
            qs.append(dict(qid=q["qid"], type="parallel", row=None, groups=[g["uids"] for g in q["evidence_groups"]]))
            qtext.append(q["question"])
    if a.queries_out:
        with open(a.queries_out, "w") as f:
            for i, (q, t) in enumerate(zip(qs, qtext)):
                q["row"] = i
                f.write(json.dumps(dict(qid=q["qid"], text=t)) + "\n")
    n_par = sum(q["type"] == "parallel" for q in qs)
    src = dict(person=a.person, periods=a.periods, n_bad_excluded=len(bad), main_qids=bool(main_q), parallel=bool(a.parallel),
               leak_filter=dict(threshold=a.leak_threshold, desktop_exempt=a.desktop_exempt) if a.leak_filter else None)
    json.dump(dict(pool=pool, pool_group=[uid[f] for f in pool], questions=qs, source=src), open(a.out, "w"))
    print(f"{len(qs) - n_par} single-hop + {n_par} parallel questions" + (f" (of {n_par_all})" if a.leak_filter else "")
          + f", pool {len(pool)} frames -> {a.out}")


if __name__ == "__main__":
    main()
