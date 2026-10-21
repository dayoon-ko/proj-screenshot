"""Parallel step 2: parallel multi-hop questions over several recordings (order, compare, aggregate) -> questions_multihop.jsonl.

Hops are verified single-hop facts (s7 single_all.jsonl: kept anchor cues and input values) or diary records; links are chosen by
rules; the LLM (paper run: Claude Opus 5.5 through the `claude` CLI) only words the question and checks it.
Specs, per person (A first, then B):
  order     : two anchored events on the same day, >= 15 min apart, different action labels; "which did I do first?" (<= 100)
  compare   : two anchored events with numeric input values of the same unit (different values), same day or same site; "which was higher?"
  aggregate : (site, action label, month) groups of 2-5 events, dropped if the person has an unlabelled event on that site in that
              month (the count must be exact); "how many times in <month> did I <action> on <site>?" (<= 120)
Sequential specs (the input value of the task right before / after an anchored task) are drawn first, as in the paper run: they fix
the random state and the spec ids ("mid") of the parallel specs. They are not worded, checked or released (multi-hop = parallel only);
--with-sequential keeps them.
Wording: one call per 20 specs (prompts/parallel_wording.txt): no answer, no input value, no date (aggregate: the month only), a site
  is named only in aggregate questions.
Checks (deletion reasons):
  leak                          : an input value appears in the question;
  compare_not_single_price      : compare values that are not single prices (ranges, mixed units);
  not_answerable_with_all_evidence / shortcut_drop_one : an LLM given all evidence + up to 6 same-day distractors must recover the
                                  answer, and must NOT recover it after any one hop is dropped (non-aggregate);
  recount_mismatch              : two independent LLM recounts over all the person's events on that site in that month must return
                                  exactly the group (aggregate);
  compare_event_reused          : at most 2 kept compare questions per event.

example: python data_pipeline/p2_multihop.py --diary diary.jsonl --single-all single_all.jsonl --action-labels action_labels.jsonl \
             --raw multihop_raw.jsonl --checks multihop_checks.jsonl --recount multihop_recount.jsonl \
             --out-specs multihop_specs.jsonl --out-all multihop_all.jsonl --out questions_multihop.jsonl
"""
import argparse
import collections
import json
import os
import random
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import COST, load_prompt, read_jsonl, site_key, write_jsonl  # noqa: E402
from parallel_lib import Facts, check, leaks, recount, recount_ok, start, word  # noqa: E402

WORD_SYSTEM = load_prompt("parallel_wording.txt")
RECOUNT_SYSTEM = load_prompt("parallel_recount.txt")
CAP = {"next": 150, "prev": 75, "order": 100, "aggregate": 120}     # per person


def month(d):
    return datetime.fromisoformat(d).strftime("%B %Y")


def number(ans):
    """(value, unit) of a numeric input value with a currency or a unit word; None otherwise."""
    m = re.search(r"(\$|€|£)?\s?(\d[\d,]*(?:\.\d+)?)\s*([a-zA-Z%]+)?", ans or "")
    if not m:
        return None
    unit = m.group(1) or (m.group(3) or "").lower().rstrip("s")
    return (float(m.group(2).replace(",", "")), unit) if unit else None


def make_specs(F, persons, cap):
    random.seed(0)
    specs = []

    def one_input(eid):
        xs = F.inputs.get(eid, [])
        sl = collections.Counter(x["slot"] for x in xs)
        xs = [x for x in xs if sl[x["slot"]] == 1]          # the slot type alone must pick one value
        return random.choice(xs) if xs else None

    for p in persons:
        full = sorted([e for e in F.E if e["person"] == p], key=lambda e: e["start"])
        # sequential (not released; drawn first because it consumes the random state)
        nx, pv = [], []
        for a, b in zip(full, full[1:]):
            if a["date"] != b["date"] or (start(b) - start(a, "stop")).total_seconds() > 600:
                continue
            ia, ib = one_input(a["event_id"]), one_input(b["event_id"])
            if a["event_id"] in F.anchor and ib:
                nx.append({"kind": "next", "cue_event": a["event_id"], "ask_event": b["event_id"], "ask": ib})
            if b["event_id"] in F.anchor and ia:
                pv.append({"kind": "prev", "cue_event": b["event_id"], "ask_event": a["event_id"], "ask": ia})
        for k, L in (("next", nx), ("prev", pv)):
            for s in random.sample(L, min(cap[k], len(L))):
                specs.append({**s, "person": p, "structure": "sequential", "events": [s["cue_event"], s["ask_event"]], "answer": s["ask"]["answer"]})
        labelled = [e for e in full if e["event_id"] in F.anchor]
        # order
        od, byday = [], collections.defaultdict(list)
        for e in labelled:
            byday[e["date"]].append(e)
        for L in byday.values():
            for i in range(len(L)):
                for j in range(i + 1, len(L)):
                    a, b = L[i], L[j]
                    if abs((start(b) - start(a)).total_seconds()) >= 900 and F.LAB.get(a["event_id"]) != F.LAB.get(b["event_id"]):
                        od.append((a, b))
        for a, b in random.sample(od, min(cap["order"], len(od))):
            x, y = random.sample([a, b], 2)
            first = x if x["start"] < y["start"] else y
            specs.append({"kind": "order", "person": p, "structure": "parallel", "events": [x["event_id"], y["event_id"]],
                          "answer": F.anchor[first["event_id"]], "first_event": first["event_id"]})
        # compare (numeric input values with the same unit)
        nums = [(e, x, number(x["answer"])) for e in labelled for x in F.inputs.get(e["event_id"], []) if x["slot"] == "number" and number(x["answer"])]
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                (a, xa, va), (b, xb, vb) = nums[i], nums[j]
                if a["event_id"] == b["event_id"] or va[1] != vb[1] or va[0] == vb[0]:
                    continue
                if a["date"] != b["date"] and site_key(a["site"]) != site_key(b["site"]):
                    continue
                hi = a if va[0] > vb[0] else b
                specs.append({"kind": "compare", "person": p, "structure": "parallel", "events": [a["event_id"], b["event_id"]], "asks": [xa, xb],
                              "answer": f"{F.anchor[hi['event_id']]} ({xa['answer']} vs {xb['answer']})", "higher_event": hi["event_id"]})
        # aggregate
        grp = collections.defaultdict(list)
        for e in full:
            if e["event_id"] in F.LAB:
                grp[(site_key(e["site"]), F.LAB[e["event_id"]], e["date"][:7])].append(e)
        unl = {(site_key(e["site"]), e["date"][:7]) for e in full if e["event_id"] not in F.LAB}
        ag = [(k, L) for k, L in grp.items() if 2 <= len(L) <= 5 and (k[0], k[2]) not in unl]
        for (sk, act, mon), L in random.sample(ag, min(cap["aggregate"], len(ag))):
            specs.append({"kind": "aggregate", "person": p, "structure": "parallel", "events": [e["event_id"] for e in L],
                          "action": act, "site": L[0]["site"], "month": month(L[0]["date"]), "answer": str(len(L))})
    for k, s in enumerate(specs):
        s["mid"] = f"M-{s['person']}-{s['kind']}-{k:04d}"
    return specs


def wording_item(F):
    def wspec(k, s):
        if s["kind"] in ("next", "prev"):
            return {"id": k, "kind": s["kind"], "cue": F.anchor[s["cue_event"]], "slot": s["ask"]["slot"],
                    "second_task_question_for_reference_only": s["ask"]["question"]}
        if s["kind"] == "order":
            return {"id": k, "kind": "order", "cues": [F.anchor[e] for e in s["events"]]}
        if s["kind"] == "compare":
            return {"id": k, "kind": "compare", "cues": [F.anchor[e] for e in s["events"]], "slot_questions": [x["question"] for x in s["asks"]]}
        return {"id": k, "kind": "aggregate", "action": s["action"], "site": s["site"], "month": s["month"],
                "example_tasks": [F.EV[e]["instruction"][:120] for e in s["events"][:3]]}
    return wspec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--diary", required=True)
    p.add_argument("--single-all", required=True, help="s7 single_all.jsonl (every single-hop candidate with its status)")
    p.add_argument("--action-labels", required=True, help="p1 output")
    p.add_argument("--persons", default="A,B")
    p.add_argument("--cap", default="{}", help='JSON overrides of the per-person caps, e.g. {"order": 600}')
    p.add_argument("--with-sequential", action="store_true", help="also word, check and output the sequential specs")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--raw", required=True, help="wording raw JSONL (resumable)")
    p.add_argument("--checks", required=True, help="answerability-check raw JSONL (resumable)")
    p.add_argument("--recount", required=True, help="recount raw JSONL (resumable)")
    p.add_argument("--out-specs", required=True)
    p.add_argument("--out-all", required=True, help="every worded question with status and delete reasons")
    p.add_argument("--out", required=True, help="kept questions")
    a = p.parse_args()

    F = Facts(a.single_all, a.action_labels, read_jsonl(a.diary))
    print(f"facts: anchors {len(F.anchor)} | events with input facts {len(F.inputs)}", flush=True)
    specs = make_specs(F, a.persons.split(","), {**CAP, **json.loads(a.cap)})
    write_jsonl(a.out_specs, specs)
    print("specs", dict(collections.Counter((s["person"], s["kind"]) for s in specs)), flush=True)
    specs = [s for s in specs if a.with_sequential or s["structure"] == "parallel"]

    R = word(specs, wording_item(F), WORD_SYSTEM, a.raw, a.model)
    rows = [R[s["mid"]] for s in specs if "question" in R.get(s["mid"], {})]
    CK = check([s for s in rows if s["kind"] != "aggregate"], F, a.checks, a.model)
    RC = recount([s for s in rows if s["kind"] == "aggregate"],
                 lambda s: [e for e in F.E if e["person"] == s["person"] and site_key(e["site"]) == site_key(s["site"])
                            and e["date"][:7] == F.EV[s["events"][0]]["date"][:7]],
                 F, RECOUNT_SYSTEM, a.recount, a.model)

    single_price = re.compile(r"(under |below |less than |up to |max(imum)? )?[$€£]\s?\d[\d,]*(\.\d+)?")
    out, reasons_ct, used_cmp = [], collections.Counter(), collections.Counter()
    for s in sorted(rows, key=lambda s: s["mid"]):
        reasons = []
        vals = [s["ask"]["answer"]] if s["kind"] in ("next", "prev") else [x["answer"] for x in s.get("asks", [])]
        if leaks(s["question"], vals):
            reasons.append("leak")
        if s["kind"] == "compare" and not all(single_price.fullmatch((x["answer"] or "").strip().lower()) for x in s["asks"]):
            reasons.append("compare_not_single_price")
        if s["kind"] == "aggregate":
            if not recount_ok(RC, s):
                reasons.append("recount_mismatch")
        else:
            c = CK.get((s["mid"], "all"))
            if not c or not c["match"]:
                reasons.append("not_answerable_with_all_evidence")
            if any(CK.get((s["mid"], f"drop:{h}"), {}).get("match") for h in s["events"]):
                reasons.append("shortcut_drop_one")
        if s["kind"] == "compare" and not reasons:
            if any(used_cmp[e] >= 2 for e in s["events"]):
                reasons.append("compare_event_reused")
            else:
                used_cmp.update(s["events"])
        reasons_ct.update(reasons)
        roles = ["anchor", "target"] if s["kind"] in ("next", "prev") else [f"hop{i + 1}" for i in range(len(s["events"]))]
        out.append({"qid": s["mid"], "hop": "multi", "structure": s["structure"], "type": s["kind"], "person": s["person"],
                    "question": s["question"], "answer": s["answer"], "evidence_groups": [F.group(r, e) for r, e in zip(roles, s["events"])],
                    "status": "deleted" if reasons else "kept", "delete_reasons": reasons})
    write_jsonl(a.out_all, out)
    kept = [{k: v for k, v in r.items() if k not in ("status", "delete_reasons")} for r in out if r["status"] == "kept"]
    write_jsonl(a.out, kept)
    print("worded", len(out), "| kept", dict(collections.Counter(f"{r['person']}/{r['type']}" for r in kept)),
          "| reasons", dict(reasons_ct), f"| cost ${COST['usd']:.2f}")


if __name__ == "__main__":
    main()
