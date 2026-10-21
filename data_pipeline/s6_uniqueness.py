"""Step 6: uniqueness judge for the candidate test questions -> judged.jsonl.

A question is ambiguous if the person could just as plausibly mean ANOTHER task of their own history. One LLM call per event
(paper run: Claude Opus 5.5 through the `claude` CLI; prompts/judge_uniqueness.txt): the target task, all its where / when / how /
input / anchor candidates with their ground-truth answers, and up to 30 competitor tasks of the same person (common.Competitors:
repeats, same site on the same day, similar tasks). For every competitor the judge lists the questions it matches and whether the
competitor gives the SAME answer.
  delete : a leak flag from s5, span_not_in_source, a matching competitor with a DIFFERENT answer, an input question the leak judge
           (prompts/judge_input_leak.txt, chunks of 30) says gives its answer away, or an event the judge never answered;
  keep   : otherwise; competitors with the SAME answer add their recordings to the gold (anchors never: every match deletes them).
"what" questions (cue = site + time only) are judged by rule: ambiguous if another task of the same site on the same date in the same
part of day (generic sites: within 30 min) exists among the competitors.

example: python data_pipeline/s6_uniqueness.py --candidates candidates.jsonl --scope scope.json --diary diary.jsonl --task-emb task_emb.npy \
             --raw judge_raw.jsonl --raw-leak leak_raw.jsonl --out judged.jsonl
"""
import argparse
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (COST, Competitors, annotate, append_jsonl, claude_json, is_generic, load_prompt, read_cache, read_jsonl,  # noqa: E402
                    site_key, write_jsonl)

SYSTEM = load_prompt("judge_uniqueness.txt")
LEAK_SYSTEM = load_prompt("judge_input_leak.txt")
JUDGED = ("where", "when", "how", "input", "anchor")


def judge_event(e, qs, comp, EV, model):
    qd = [{"id": f"q{k}", "type": c["type"], "question": c["question"],
           "ground_truth": c["answer"] if c["type"] != "anchor" else "(identifies the target task)"} for k, c in enumerate(qs)]
    prompt = ("Target task:\n" + json.dumps(annotate(e, "T"), ensure_ascii=False) + "\n\nQuestions:\n" + json.dumps(qd, ensure_ascii=False)
              + "\n\nCandidates:\n" + json.dumps([annotate(EV[x["cid"]], j) for j, x in enumerate(comp)], ensure_ascii=False))
    res = claude_json(SYSTEM, prompt, model)
    if not isinstance(res, list):
        return None
    got = {o.get("cid"): o for o in res if isinstance(o, dict)}
    verdicts = []
    for j, x in enumerate(comp):
        o = got.get(j) if j in got else got.get(str(j))
        for m in (o or {}).get("matches", []):
            s = str(m.get("q", ""))
            k = int(s[1:]) if s.startswith("q") and s[1:].isdigit() else -1
            if 0 <= k < len(qs):
                verdicts.append({"qid": qs[k]["qid"], "cid": x["cid"], "why_found": x["why"], "sim": x["sim"],
                                 **{kk: m.get(kk) for kk in ("same", "value", "why")}})
    missing = [j for j in range(len(comp)) if j not in got and str(j) not in got]
    return {"event_id": e["event_id"], "verdicts": verdicts, "n_comp": len(comp), "missing_cids": missing}


def what_ambiguous(e, comp, EV):
    t0, out = datetime.fromisoformat(e["start"]), []
    for x in comp:
        o = EV[x["cid"]]
        if site_key(o["site"]) != site_key(e["site"]) or o["date"] != e["date"]:
            continue
        if is_generic(e["site"]):
            if abs((datetime.fromisoformat(o["start"]) - t0).total_seconds()) <= 1800:
                out.append(x["cid"])
        elif o["part_of_day"] == e["part_of_day"]:
            out.append(x["cid"])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", required=True, help="candidates.jsonl from s5")
    p.add_argument("--scope", required=True, help="scope.json from s5")
    p.add_argument("--diary", required=True)
    p.add_argument("--task-emb", required=True, help="(n_events, d) task-text vectors in diary order (s4)")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--raw", required=True, help="raw judge outputs (resumable)")
    p.add_argument("--raw-leak", required=True, help="raw input-leak outputs (resumable)")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    diary = read_jsonl(a.diary)
    EV = {e["event_id"]: e for e in diary}
    cands = read_jsonl(a.candidates)
    byev = collections.defaultdict(list)
    for c in cands:
        byev[c["event_id"]].append(c)
    events = [e for e in json.load(open(a.scope)) if e in byev]
    comp_of = Competitors(diary, np.load(a.task_emb))
    COMP = {eid: comp_of(EV[eid]) for eid in events}
    print(f"events {len(events)} | competitors mean {sum(map(len, COMP.values())) / max(1, len(COMP)):.1f}", flush=True)

    done = read_cache(a.raw, "event_id")

    def work(eid):
        qs = [c for c in byev[eid] if c["type"] in JUDGED]
        if not qs or not COMP[eid]:
            r = {"event_id": eid, "verdicts": [], "n_comp": len(COMP[eid])}
        else:
            r = judge_event(EV[eid], qs, COMP[eid], EV, a.model)
            if r is None:
                return
        append_jsonl(a.raw, r)
        done[eid] = r
    todo = [e for e in events if e not in done]
    print(f"judge: cached {len(done)} | to judge {len(todo)}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))

    leak = read_cache(a.raw_leak, "qid")
    inq = [c for e in events for c in byev[e] if c["type"] == "input" and c["qid"] not in leak]

    def lwork(ch):
        out = claude_json(LEAK_SYSTEM, "Items:\n" + json.dumps([{"id": k, "question": c["question"], "answer": c["answer"]}
                                                                for k, c in enumerate(ch)], ensure_ascii=False), a.model)
        by = {o.get("id"): o for o in out or [] if isinstance(o, dict)}
        for k, c in enumerate(ch):
            if k in by:
                r = {"qid": c["qid"], "leak": bool(by[k].get("leak")), "why": by[k].get("why")}
                append_jsonl(a.raw_leak, r)
                leak[c["qid"]] = r
    print(f"input leak check: cached {len(leak)} | to check {len(inq)}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lwork, [inq[b:b + 30] for b in range(0, len(inq), 30)]))

    V = collections.defaultdict(list)
    for r in done.values():
        for v in r["verdicts"]:
            V[v["qid"]].append(v)
    out = []
    for eid in events:
        for c in byev[eid]:
            c, reasons = dict(c), []
            if any(f.startswith("leak") for f in c["flags"]):
                reasons.append("leak_rule")
            if "span_not_in_source" in c["flags"]:
                reasons.append("span_not_in_source")
            if c["type"] == "what":
                amb = what_ambiguous(EV[eid], COMP[eid], EV)
                c["competitors_diff"], c["competitors_same"] = [{"cid": x} for x in amb], []
                if amb:
                    reasons.append("ambiguous")
            else:
                if eid not in done:
                    reasons.append("unjudged")
                vs = V.get(c["qid"], [])
                c["competitors_diff"] = [v for v in vs if not v.get("same")]
                c["competitors_same"] = [v for v in vs if v.get("same")]
                if c["type"] == "anchor":
                    c["competitors_diff"] += c["competitors_same"]
                    c["competitors_same"] = []
                if c["competitors_diff"]:
                    reasons.append("ambiguous")
                if c["type"] == "input" and leak.get(c["qid"], {}).get("leak"):
                    reasons.append("leak_llm")
                c["gold_uids"] = c["gold_uids"] + [u for v in c["competitors_same"] for u in EV[v["cid"]]["uids"]]
            c["n_competitors_checked"] = len(COMP[eid])
            c["status"], c["delete_reasons"] = ("deleted" if reasons else "kept"), reasons
            out.append(c)
    write_jsonl(a.out, out)
    tab = collections.Counter((c["type"], c["status"]) for c in out)
    for t in ("where", "when", "what", "how", "input", "anchor"):
        k, d = tab[(t, "kept")], tab[(t, "deleted")]
        print(f"{t:7s} kept {k:5d} deleted {d:5d} ({100 * d / max(1, k + d):.0f}%)")
    print("reasons", dict(collections.Counter(r for c in out for r in c["delete_reasons"])), f"| cost ${COST['usd']:.2f}")


if __name__ == "__main__":
    main()
