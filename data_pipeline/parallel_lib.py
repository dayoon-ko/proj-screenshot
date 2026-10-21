"""Shared parts of the parallel multi-hop builders (p2_multihop.py, p3_parallel_extra.py): the facts the hops are built from, and the
three LLM stages (wording, answerability checks, aggregate recounts), each resumable through its raw JSONL file."""
import collections
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from common import append_jsonl, claude_json, load_prompt, nice_date, read_cache, read_jsonl

CHECK_SYSTEM = load_prompt("parallel_check_answer.txt")
JUDGE_SYSTEM = load_prompt("parallel_check_judge.txt")


class Facts:
    """Verified single-hop facts per recording (event), from the s7 single_all.jsonl: kept anchor cue, kept input values, gold recordings."""

    def __init__(self, single_all, action_labels, diary):
        self.E = diary
        self.EV = {e["event_id"]: e for e in diary}
        self.LAB = {r["event_id"]: r["action"] for r in read_jsonl(action_labels)}
        self.anchor, self.inputs, self.gold = {}, collections.defaultdict(list), {}
        for r in read_jsonl(single_all):
            eid = r["evidence_groups"][0]["event"]
            self.gold[eid] = r["evidence_groups"][0]["uids"]
            if r["status"] != "kept":
                continue
            if r["type"] == "anchor":
                self.anchor[eid] = r["question"]
            if r["type"] == "input":
                self.inputs[eid].append(r)

    def rec(self, eid, dur=False):
        """An activity record as the checking LLM sees it."""
        x = self.EV[eid]
        r = {"time": f"{nice_date(x['date'])} {x['start'][11:16]}", "site_or_app": x["site"], "instruction": x["instruction"][:300]}
        if dur:
            r.update(end_time=x["stop"][11:16], duration_min=round(x["dur_s"] / 60, 1))
        return r

    def distractors(self, ev_ids, n=6):
        """Up to n other events of the same person on the first evidence event's day, nearest in time."""
        d0 = self.EV[ev_ids[0]]
        pool = [e for e in self.E if e["person"] == d0["person"] and e["date"] == d0["date"] and e["event_id"] not in ev_ids]
        pool.sort(key=lambda e: abs((start(e) - start(d0)).total_seconds()))
        return [e["event_id"] for e in pool[:n]]

    def group(self, role, eid):
        e = self.EV[eid]
        return {"role": role, "event": eid, "uids": self.gold.get(eid, e["uids"]), "category": e["category"]}


def start(e, k="start"):
    return datetime.fromisoformat(e[k])


def norm(x):
    return re.sub(r"[^a-z0-9]", "", (x or "").lower())


def leaks(question, values):
    """An input value (>= 3 alphanumerics) appears verbatim in the question."""
    return any(len(norm(v)) >= 3 and norm(v) in norm(question) for v in values)


def word(specs, wspec, system, raw_path, model, batch=20, workers=4):
    """One wording call per `batch` specs; returns {mid: spec row + "question"} (the raw file is the source of truth)."""
    R = read_cache(raw_path, "mid")
    todo = [s for s in specs if s["mid"] not in R or "question" not in R[s["mid"]]]

    def work(ch):
        out = claude_json(system, "Specs:\n" + json.dumps([wspec(k, s) for k, s in enumerate(ch)], ensure_ascii=False), model)
        by = {o.get("id"): o for o in out or [] if isinstance(o, dict)}
        for k, s in enumerate(ch):
            if k in by and by[k].get("question"):
                r = {**s, "question": by[k]["question"]}
                append_jsonl(raw_path, r)
                R[r["mid"]] = r
    print(f"wording: {len(specs)} specs, {len(todo)} to word", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, [todo[b:b + batch] for b in range(0, len(todo), batch)]))
    return R


def check(rows, facts, chk_path, model, dur_kinds=(), batch=15, workers=4):
    """Answerability checks: with all evidence + same-day distractors the answer must be recovered ("all"); after dropping any one
    hop it must NOT be ("drop:<event>"). An answering LLM sees shuffled records, a judging LLM compares with the ground truth."""
    checks = []
    for s in rows:
        ds = facts.distractors(s["events"])
        checks.append((s["mid"], "all", s["events"] + ds))
        for h in s["events"]:
            checks.append((s["mid"], f"drop:{h}", [e for e in s["events"] if e != h] + ds))
    CK = {(r["mid"], r["cond"]): r for r in read_jsonl(chk_path)} if os.path.exists(chk_path) else {}
    todo = [c for c in checks if (c[0], c[1]) not in CK]
    by_mid = {s["mid"]: s for s in rows}

    def work(ch):
        items = []
        for k, (mid, cond, recs) in enumerate(ch):
            recs = list(recs)
            random.Random(k).shuffle(recs)
            items.append({"id": k, "question": by_mid[mid]["question"], "records": [facts.rec(e, by_mid[mid]["kind"] in dur_kinds) for e in recs]})
        out = claude_json(CHECK_SYSTEM, "Items:\n" + json.dumps(items, ensure_ascii=False), model)
        by = {o.get("id"): o.get("answer") for o in out or [] if isinstance(o, dict)}
        jitems = [{"id": k, "question": by_mid[mid]["question"], "ground_truth": by_mid[mid]["answer"], "predicted": by.get(k, "CANNOT")}
                  for k, (mid, cond, _) in enumerate(ch)]
        jout = claude_json(JUDGE_SYSTEM, "Items:\n" + json.dumps(jitems, ensure_ascii=False), model)
        jb = {o.get("id"): bool(o.get("match")) for o in jout or [] if isinstance(o, dict)}
        for k, (mid, cond, _) in enumerate(ch):
            if k in jb:
                r = {"mid": mid, "cond": cond, "pred": by.get(k), "match": jb[k]}
                append_jsonl(chk_path, r)
                CK[(mid, cond)] = r
    print(f"checks: {len(checks)}, {len(todo)} to run", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, [todo[b:b + batch] for b in range(0, len(todo), batch)]))
    return CK


def recount(rows, pool_of, facts, system, rc_path, model, workers=4):
    """Two independent recounts per aggregate (the pool order is reshuffled); the second is skipped if the first already disagrees."""
    RC = collections.defaultdict(list)
    for r in (read_jsonl(rc_path) if os.path.exists(rc_path) else []):
        RC[r["mid"]].append(r)
    todo = [s for s in rows if len(RC[s["mid"]]) < 2 and not (RC[s["mid"]] and RC[s["mid"]][-1]["ids"] != sorted(s["events"]))]

    def work(s):
        pool = pool_of(s)
        for rep in range(len(RC[s["mid"]]), 2):
            if RC[s["mid"]] and RC[s["mid"]][-1]["ids"] != sorted(s["events"]):
                break
            order = pool[:]
            random.Random(rep + 7).shuffle(order)
            out = claude_json(system, json.dumps({"question": s["question"], "tasks": [{"id": i, **facts.rec(e["event_id"])} for i, e in enumerate(order)]},
                                                 ensure_ascii=False), model)
            ids = sorted({order[i]["event_id"] for i in (out or {}).get("ids", []) if isinstance(i, int) and 0 <= i < len(order)}) \
                if isinstance(out, dict) else None
            r = {"mid": s["mid"], "rep": rep, "ids": ids}
            append_jsonl(rc_path, r)
            RC[s["mid"]].append(r)
    print(f"recounts: {len(rows)} aggregates, {len(todo)} to recount", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, todo))
    return RC


def recount_ok(RC, s):
    return len(RC[s["mid"]]) >= 2 and all(r["ids"] == sorted(s["events"]) for r in RC[s["mid"]])

