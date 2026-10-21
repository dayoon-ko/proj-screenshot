"""Step 8: evaluation input for scripts/make_eval_protocol.py from the single-hop test set -> eval_input_test.json + texts_test.jsonl.

eval_input_test.json = {"new": [{"qid", "person", "row", "groups"}]} with one group per evidence group (single-hop: one, the target
recording(s)); texts_test.jsonl = the question text of row i on line i. The parallel questions go to make_eval_protocol.py directly
(--parallel questions_test_parallel.jsonl).

example: python data_pipeline/s8_eval_input.py --single questions_test_single.jsonl --out-eval-input eval_input_test.json --out-texts texts_test.jsonl
then:    python scripts/make_eval_protocol.py --eval-input eval_input_test.json --texts texts_test.jsonl --parallel questions_test_parallel.jsonl \
             --splits splits.json --frames-meta frames.jsonl --out protocol.json --queries-out protocol_queries.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_jsonl, write_jsonl  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--single", required=True, help="questions_test_single.jsonl (s7)")
    p.add_argument("--out-eval-input", required=True)
    p.add_argument("--out-texts", required=True)
    a = p.parse_args()
    Q = read_jsonl(a.single)
    json.dump({"new": [{"qid": q["qid"], "person": q["person"], "row": i, "groups": [g["uids"] for g in q["evidence_groups"]]}
                       for i, q in enumerate(Q)]}, open(a.out_eval_input, "w"))
    write_jsonl(a.out_texts, [{"text": q["question"]} for q in Q])
    print(f"{len(Q)} single-hop questions -> {a.out_eval_input}, {a.out_texts}")


if __name__ == "__main__":
    main()
