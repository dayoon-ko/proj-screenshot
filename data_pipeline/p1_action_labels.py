"""Parallel step 1: one action-family label per test event ("verb object", e.g. "create trip plan") -> action_labels.jsonl.

The labels make aggregation predicates exact: aggregation groups are (person, site, action label, window). All test events of one
person on one site go into the same call (chunks of 40), so tasks of the same kind on that site get exactly the same label. The LLM
sees only the site/app and the instruction (paper run: Claude Opus 5.5 through the `claude` CLI, prompts/parallel_action_labels.txt).
Scope = the test events of s5 (person A in T2..T5, person B, each with >= 2 usable frames).

example: python data_pipeline/p1_action_labels.py --diary diary.jsonl --out action_labels.jsonl
"""
import argparse
import collections
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import COST, append_jsonl, claude_json, load_prompt, read_cache, read_jsonl, site_key  # noqa: E402

SYSTEM = load_prompt("parallel_action_labels.txt")


def test_scope(diary, persons, periods, min_frames=2):
    """Test events in diary order (the scope of s5)."""
    return [e["event_id"] for e in diary if e["n_frames_pool"] >= min_frames and e["person"] in persons
            and (e["person"] != "A" or e["period"] in periods)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--diary", required=True)
    p.add_argument("--persons", default="A,B")
    p.add_argument("--test-periods", default="T2,T3,T4,T5", help="person A's test periods (other persons: all events)")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out", required=True, help="resumable: labels already in the file are kept")
    a = p.parse_args()

    diary = read_jsonl(a.diary)
    EV = {e["event_id"]: e for e in diary}
    scope = test_scope(diary, set(a.persons.split(",")), set(a.test_periods.split(",")))
    by = collections.defaultdict(list)
    for eid in scope:
        by[(EV[eid]["person"], site_key(EV[eid]["site"]))].append(eid)
    chunks = [v[i:i + 40] for v in by.values() for i in range(0, len(v), 40)]
    done = set(read_cache(a.out, "event_id"))
    chunks = [c for c in chunks if not set(c) <= done]
    print(f"sites {len(by)} | calls {len(chunks)}", flush=True)

    def work(ch):
        items = [{"id": k, "site_or_app": EV[e]["site"], "instruction": EV[e]["instruction"][:300]} for k, e in enumerate(ch)]
        res = claude_json(SYSTEM, "Tasks:\n" + json.dumps(items, ensure_ascii=False), a.model)
        got = {o.get("id"): o for o in res or [] if isinstance(o, dict)}
        for k, e in enumerate(ch):
            if k in got and got[k].get("action"):
                append_jsonl(a.out, {"event_id": e, "action": re.sub(r"\s+", " ", got[k]["action"].strip().lower())})
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, chunks))
    L = {r["event_id"]: r["action"] for r in read_jsonl(a.out)}
    grp = collections.Counter((EV[e]["person"], site_key(EV[e]["site"]), x) for e, x in L.items())
    print(f"labelled {len(L)} / {len(scope)} | groups of size 2-5:", dict(collections.Counter(p for (p, s, x), k in grp.items() if 2 <= k <= 5)),
          f"| cost ${COST['usd']:.2f}")


if __name__ == "__main__":
    main()
