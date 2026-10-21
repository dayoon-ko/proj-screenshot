"""Step 5: candidate single-hop TEST questions per diary event -> candidates.jsonl.

Scope = every diary event of person A in T2..T5 (and of person B, if --persons includes it) with >= 2 usable frames.
Two prompts per batch of 12 events (paper run: Claude Opus 5.5 through the `claude` CLI); the LLM sees the event's site/app, kind,
date, start time (and part of day), instruction and steps, never the frames:
  prompts/test_task_questions.txt  : where / when / what / how questions + a short "cue" (anchor) referring to the task
                                     (no how without steps, no where for a generic site);
  prompts/test_input_questions.txt : 0-2 questions whose answer is an INPUT value the person typed, chose or set (answer + the verbatim
                                     instruction span that contains it).
Programmatic flags: leak_site (a where-question naming the site), leak_date (a when-question stating a date/time),
span_not_in_source (input answer span not in the instruction/steps), verbatim<n> (>= 6 words copied from the instruction; kept).
Gold = the event's recordings.

example: python data_pipeline/s5_test_generate.py --diary diary.jsonl --raw-task gen_task_raw.jsonl --raw-input gen_input_raw.jsonl \
             --scope-out scope.json --out candidates.jsonl
"""
import argparse
import collections
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (COST, MONTHS, append_jsonl, claude_json, is_generic, load_prompt, longest_common_run, nice_date, read_cache,  # noqa: E402
                    read_jsonl, site_key, write_jsonl)

TASK_SYSTEM = load_prompt("test_task_questions.txt")
INPUT_SYSTEM = load_prompt("test_input_questions.txt")
DATE_RE = re.compile(r"\b(" + "|".join(MONTHS + [m[:3] for m in MONTHS]) + r")\b|\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
                     r"|\b\d{1,2}(:\d\d)?\s*(am|pm)\b|\b\d{1,2}/\d{1,2}\b", re.I)
GENERIC_WORDS = {"microsoft", "google", "windows", "online", "settings", "setting", "tool", "tools", "apps", "account", "search", "player",
                 "media", "client", "desktops", "options", "system", "network", "privacy", "university", "news", "maps", "timer", "clock",
                 "email", "browser", "preferences", "macos", "virtual"}


def item(i, e, task):
    d = {"id": i, "app_or_website": e["site"], "kind": "website" if e["category"] == "web" else "desktop app",
         "date": nice_date(e["date"]), "start_time": e["start"][11:16], "instruction": e["instruction"][:700]}
    if task:
        d["part_of_day"] = e["part_of_day"]
    if e["steps"]:
        d["steps"] = e["steps"][:10]
    return d


def generate(scope, path, system, task, model, workers):
    done = read_cache(path, "event_id")
    todo = [(i, e) for i, e in enumerate(scope) if e["event_id"] not in done]
    print(f"{os.path.basename(path)}: cached {len(done)} | to generate {len(todo)}", flush=True)

    def work(batch):
        res = claude_json(system, "Diary entries:\n" + json.dumps([item(i, e, task) for i, e in batch], ensure_ascii=False), model)
        if not isinstance(res, list):
            return
        by = {o.get("id"): o for o in res if isinstance(o, dict)}
        for i, e in batch:
            if i in by:
                r = {"event_id": e["event_id"], **{k: v for k, v in by[i].items() if k != "id"}}
                append_jsonl(path, r)
                done[e["event_id"]] = r
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, [todo[b:b + 12] for b in range(0, len(todo), 12)]))
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def flatten(EV, task_raw, input_raw):
    cands = []
    for g in task_raw:
        e = EV[g["event_id"]]
        qs = g.get("questions") or {}
        for t in ("where", "when", "what", "how"):
            q = qs.get(t)
            if not q or (t == "how" and not e["steps"]) or (t == "where" and is_generic(e["site"])):
                continue
            flags = []
            if t == "where":
                sk, qk = site_key(e["site"]), re.sub(r"[^a-z0-9]", "", q.lower())
                toks = [w for w in re.findall(r"[a-z0-9]+", e["site"].lower()) if len(w) >= 4 and w not in GENERIC_WORDS]
                if (len(sk) >= 4 and sk in qk) or any(re.search(r"\b" + w + r"\b", q.lower()) for w in toks):
                    flags.append("leak_site")
            if t == "when" and DATE_RE.search(q):
                flags.append("leak_date")
            run_ = longest_common_run(q, e["instruction"])
            if run_ >= 6:
                flags.append(f"verbatim{run_}")
            gt = {"where": e["site"], "when": f"{e['date']} {e['start'][11:16]}-{e['stop'][11:16]} ({e['part_of_day']})",
                  "what": g.get("summary"), "how": g.get("how_answer")}[t]
            cands.append({"qid": f"{e['event_id']}#{t}", "event_id": e["event_id"], "type": t, "question": q, "answer": gt, "cue": g.get("cue"),
                          "flags": flags})
        if g.get("cue"):
            cands.append({"qid": f"{e['event_id']}#anchor", "event_id": e["event_id"], "type": "anchor", "question": g["cue"], "answer": None,
                          "cue": g["cue"], "flags": []})
    for g in input_raw:
        e = EV[g["event_id"]]
        for k, q in enumerate(g.get("questions") or []):
            span, flags = q.get("value_span") or "", []
            if span.lower() not in (e["instruction"] + " " + " ".join(e["steps"])).lower():
                flags.append("span_not_in_source")
            run_ = longest_common_run(q.get("question"), e["instruction"])
            if run_ >= 6:
                flags.append(f"verbatim{run_}")
            cands.append({"qid": f"{e['event_id']}#input{k}", "event_id": e["event_id"], "type": "input", "slot": q.get("slot"),
                          "question": q.get("question"), "answer": q.get("answer"), "value_span": span, "cue_parts": q.get("cue_parts"),
                          "flags": flags})
    for c in cands:
        e = EV[c["event_id"]]
        c.update(person=e["person"], period=e["period"], category=e["category"], site=e["site"], gold_uids=e["uids"])
    return cands


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--diary", required=True)
    p.add_argument("--persons", default="A,B", help="A = person A's T2..T5 events; B = all of person B's events (paper run: A,B)")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--raw-task", required=True)
    p.add_argument("--raw-input", required=True)
    p.add_argument("--scope-out", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    E = read_jsonl(a.diary)
    EV = {e["event_id"]: e for e in E}
    persons = set(a.persons.split(","))
    scope = [e for e in E if e["n_frames_pool"] >= 2 and e["person"] in persons
             and (e["person"] != "A" or e["period"] in ("T2", "T3", "T4", "T5"))]
    json.dump([e["event_id"] for e in scope], open(a.scope_out, "w"))
    print("scope", len(scope), dict(collections.Counter((e["person"], e["category"]) for e in scope)), flush=True)
    task_raw = generate(scope, a.raw_task, TASK_SYSTEM, True, a.model, a.workers)
    input_raw = generate(scope, a.raw_input, INPUT_SYSTEM, False, a.model, a.workers)
    cands = flatten(EV, task_raw, input_raw)
    write_jsonl(a.out, cands)
    print("candidates", len(cands), dict(collections.Counter(c["type"] for c in cands)),
          "| flags", dict(collections.Counter(f.rstrip("0123456789") for c in cands for f in c["flags"])), f"| cost ${COST['usd']:.2f}")


if __name__ == "__main__":
    main()
