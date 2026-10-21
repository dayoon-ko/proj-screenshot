"""Step 3: TRAINING questions per task definition (the "v1" slot recipe; the v1 retrieval LoRA was trained on T1-train and selected on
T1-val; the paper's LoRA trains on the 1,000-question subset of step 9) -> questions.jsonl (+ questions_train.jsonl / questions_val.jsonl).

For every unique task definition (taskId) an LLM sees only the instruction, the app/website and the optional step list (no frames, no
date) and writes four first-person recall questions: where (which app/site/tool), when, how (which control/menu/sequence) and which
(one detail that can only be read from the screen). Paper run: Claude Opus 4.8 through the `claude` CLI, batches of 20 tasks.
Gold = every recording of the taskId; questions with identical text across tasks share the union of their recordings.
leak = a where-question that names the app/site as a proper noun.
Person-A questions get a split: T1-train / T1-val / T2..T5 if all gold recordings (twins mapped, restricted to A) fall in one period
(T1 questions must lie entirely in T1-train or T1-val); leak questions get no split.

example: python data_pipeline/s3_train_questions.py --meta psai_tasks_meta.jsonl --splits splits.json --raw slot_raw.jsonl --out questions.jsonl \
             --train-out questions_train.jsonl --val-out questions_val.jsonl
"""
import argparse
import collections
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import COST, append_jsonl, claude_json, write_jsonl  # noqa: E402

TYPES = ("where", "when", "how", "which")
SYSTEM = """You turn definitions of computer tasks into recall questions that the person who did the task might ask later about their own screen history.
Use ONLY each task definition (instruction, app or website, optional step list). Never invent facts and never assume what was on the screen.
For each task:
1. Fill the slots: "what" (the action and its object), "where" (the app, website or tool), "how" (the way it is done; use the steps if given), "who", "why". Write "not stated" when the definition does not say.
2. Write four questions in the first person, past tense, natural wording, at most 25 words each:
   - "where": asks which app, website or tool it was. It must NOT contain the name or URL of that app, website or tool.
   - "when": asks when the person did it. It may mention what and where.
   - "how": asks how it was done (which control, menu, filter or sequence) without listing the steps.
   - "which": asks for ONE specific detail that is not in the definition and can only be read from the screen (the item chosen, a value found, a name, text that was typed).
Examples of good output for three tasks:
{"id": 0, "slots": {"what": "delete an unwanted promotional email", "where": "the Promotions tab of the webmail inbox", "how": "open the Promotions tab, hover over the email, click the trash icon", "who": "not stated", "why": "the email was unwanted"}, "questions": {"where": "In which part of my mailbox did I find the promotional email I deleted?", "when": "When did I delete a promotional email from my inbox?", "how": "How did I delete that promotional email: which tab did I open and what did I click?", "which": "Which promotional email did I delete?"}}
{"id": 1, "slots": {"what": "find a travel credit card with no foreign transaction fee that earns at least 2 points per dollar, and read its annual fee", "where": "NerdWallet, travel credit card section", "how": "open the comparison chart for no-foreign-transaction-fee cards and check the earn rate", "who": "not stated", "why": "not stated"}, "questions": {"where": "On which site did I compare travel credit cards that charge no foreign transaction fee?", "when": "When did I look up travel credit cards with no foreign transaction fee?", "how": "How did I find a card earning at least 2 points per dollar: which chart or filter did I use?", "which": "What annual fee did the travel card I picked have?"}}
{"id": 2, "slots": {"what": "clear the cookies of one website and reload it", "where": "browser developer tools", "how": "Inspect, Application tab, Storage, Cookies, select the domain, Clear All, reload", "who": "not stated", "why": "not stated"}, "questions": {"where": "Which browser tool did I use to clear one site's cookies?", "when": "When did I clear one site's cookies and reload the page?", "how": "How did I clear that site's cookies: which tab and which button?", "which": "For which website did I clear the cookies?"}}
Answer with a JSON array only, one object per task in the same order, each {"id": <the task's id>, "slots": {...}, "questions": {...}}. No prose, no code fences."""

BOILER = re.compile(r"stop the screen recording|paradigm shift ai portal", re.I)
GENERIC = {"search", "engine", "browser", "web", "microsoft", "google", "windows", "macos", "email", "app", "settings", "system", "preferences",
           "online", "site", "tool", "file", "manager", "client", "website"}


def task_definitions(meta):
    tasks = collections.OrderedDict()
    for r in meta:
        t = tasks.setdefault(r["taskId"], {"taskId": r["taskId"], "instruction": " ".join((r["task_name"] or "").split()),
                                           "app": r["application_website"], "category": r["category"], "steps": [], "uids": []})
        t["uids"].append(r["unique_data_id"])
        if not t["steps"] and r.get("reasoning_steps"):
            seen = []
            for s in r["reasoning_steps"]:
                s1 = " ".join(s.split())
                if s1 and not BOILER.search(s1) and s1 not in seen:
                    seen.append(s1)
            t["steps"] = seen
    return list(tasks.values())


def names_app(q, app):
    """v1 leak rule: the where-question names the app/site as a proper noun (case-sensitive: "weather site" for "Weather" is fine)."""
    full = re.sub(r"[^a-z0-9]", "", app.lower())
    rest = " ".join(q.split()[1:])
    names = [x for x in re.findall(r"[A-Za-z0-9]+", app) if len(x) >= 4 and x.lower() not in GENERIC and x != x.lower()]
    return bool(any(re.search(rf"\b{re.escape(x)}\b", rest) for x in names)
                or (len(full) >= 4 and full not in GENERIC and full in re.sub(r"[^a-z0-9]", "", q.lower()) and " " in app.strip()))


def generate(tasks, raw_paths, model, batch, workers):
    done = collections.OrderedDict()
    for path in raw_paths:                        # first file wins (the paper run: main run, then a re-run for its parse failures)
        if os.path.exists(path):
            for l in open(path):
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                if r.get("questions") and set(r["questions"]) >= set(TYPES):
                    done.setdefault(r["taskId"], r)
    todo = [(i, t) for i, t in enumerate(tasks) if t["taskId"] not in done]
    print(f"task definitions {len(tasks)} | cached {len(done)} | to generate {len(todo)}", flush=True)

    def work(b):
        payload = [{"id": i, "instruction": t["instruction"], "app_or_website": t["app"], **({"steps": t["steps"][:12]} if t["steps"] else {})}
                   for i, t in b]
        arr = claude_json(SYSTEM, "Task definitions:\n" + json.dumps(payload, ensure_ascii=False), model) or []
        by = {int(o["id"]): o for o in arr if isinstance(o, dict) and set(o.get("questions", {})) >= set(TYPES)}
        for i, t in b:
            if i in by:
                r = {"taskId": t["taskId"], "slots": by[i].get("slots", {}), "questions": by[i]["questions"], "generator": model}
                append_jsonl(raw_paths[0], r)
                done[t["taskId"]] = r
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, [todo[k:k + batch] for k in range(0, len(todo), batch)]))
    print(f"generated; cost ${COST['usd']:.2f}", flush=True)
    return done


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meta", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--raw", nargs="+", required=True, help="raw generation JSONL(s); new outputs are appended to the first")
    p.add_argument("--model", default="claude-opus-4-8")
    p.add_argument("--batch", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True, help="all questions with their split")
    p.add_argument("--train-out", help="T1-train questions {qid, split, asks, question, taskId, gold}")
    p.add_argument("--val-out", help="T1-val questions, same fields")
    a = p.parse_args()

    meta = [json.loads(l) for l in open(a.meta)]
    tasks = task_definitions(meta)
    by_tid = {t["taskId"]: t for t in tasks}
    gen = generate(tasks, a.raw, a.model, a.batch, a.workers)

    qs = []
    for tid, r in gen.items():
        t = by_tid[tid]
        for k in TYPES:
            q = " ".join(str(r["questions"][k]).split())
            qs.append({"qid": f"SL-{tid}-{k}", "asks": k, "question": q, "taskId": tid, "uids": list(t["uids"]), "app": t["app"],
                       "category": t["category"], "leak": k == "where" and names_app(q, t["app"])})
    grp = collections.defaultdict(list)
    for q in qs:
        grp[q["question"].lower()].append(q)
    for g in grp.values():                        # identical text = one information need
        if len({q["taskId"] for q in g}) > 1:
            u = sorted({x for q in g for x in q["uids"]})
            for q in g:
                q["uids"] = u

    S = json.load(open(a.splits))
    keepA = {u for per in S["periods"] for u in per["uids"]}
    period = {u: k for k, per in enumerate(S["periods"]) for u in per["uids"]}
    tr, va = set(S["t1_train_uids"]), set(S["t1_val_uids"])
    stat = collections.Counter()
    for q in qs:
        g = sorted({S["twins_A"].get(u, u) for u in q["uids"]} & keepA)
        q["gold"], q["split"] = g, None
        if not g:
            continue
        if q["leak"]:
            stat["A: leak"] += 1
            continue
        ps = {period[u] for u in g}
        if len(ps) > 1:
            stat["A: spans periods"] += 1
            continue
        k = ps.pop()
        if k > 0:
            q["split"] = f"T{k + 1}"
        elif all(u in tr for u in g):
            q["split"] = "T1-train"
        elif all(u in va for u in g):
            q["split"] = "T1-val"
        else:
            stat["A: T1 train/val boundary"] += 1
    stat.update(f"split {q['split']}" for q in qs if q["split"])
    write_jsonl(a.out, qs)
    for split, path in (("T1-train", a.train_out), ("T1-val", a.val_out)):
        if path:
            write_jsonl(path, [{"qid": q["qid"], "split": split, "asks": q["asks"], "question": q["question"], "taskId": q["taskId"],
                                "gold": q["gold"]} for q in qs if q["split"] == split])
    print(f"questions {len(qs)} | where-questions naming the app/site {sum(q['leak'] for q in qs)} |", dict(sorted(stat.items())))


if __name__ == "__main__":
    main()
