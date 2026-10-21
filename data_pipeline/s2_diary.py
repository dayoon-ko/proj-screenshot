"""Step 2: per-person "diary" of performed tasks from the recording annotations only -> diary.jsonl.

One row per event = recording (twins of the same machine with identical start/stop are merged into one event). The instruction is
cleaned of recording boilerplate ("Only use https://... to achieve this task", trailing URLs); steps drop the record start/stop steps.
The diary is what the answer step (s5) and the uniqueness judge (s6) see about a recording, besides its frames.

example: python data_pipeline/s2_diary.py --meta psai_tasks_meta.jsonl --frames-meta frames.jsonl --splits splits.json --out diary.jsonl
"""
import argparse
import collections
import json
import re
from datetime import datetime

BOILER_SENT = re.compile(r"^(The task is achievable with just navigation from this site\.?|\(https?://[^)]*\)\.?|Only use .*|Don'?t go to .*"
                         r"|Do not go to .*|Use only .*)$", re.I)
STEP_BOILER = re.compile(r"stop the screen recording|paradigm shift ai portal|start the screen recording", re.I)


def clean_instr(t):
    t = " ".join((t or "").split())
    sents = re.split(r"(?<=[.!?)])\s+", t)
    while sents and BOILER_SENT.match(sents[-1].strip()):
        sents.pop()
    t = " ".join(sents)
    t = re.sub(r"\s*\(https?://[^)]*\)\s*$", "", t)
    t = re.sub(r"\s*Only use https?://\S+ to achieve (this|the) task\.?", "", t, flags=re.I)
    t = re.sub(r"\s*(Don'?t|Do not) go to any other (web)?sites?[^.]*\.?", "", t, flags=re.I)
    return t.strip()


def clean_steps(steps):
    out = []
    for s in steps or []:
        s1 = " ".join((s or "").split())
        if s1 and not STEP_BOILER.search(s1) and s1 not in out:
            out.append(s1)
    return out


def part_of_day(h):
    return "night" if h < 5 else "morning" if h < 12 else "afternoon" if h < 17 else "evening" if h < 21 else "night"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meta", required=True)
    p.add_argument("--frames-meta", required=True)
    p.add_argument("--splits", required=True, help="splits.json from s1 (periods, bad_fids)")
    p.add_argument("--nodes", default="DESKTOP-GFL3B1N:A,Learno:B", help="machine:person pairs")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    nodes = dict(x.split(":") for x in a.nodes.split(","))
    S = json.load(open(a.splits))
    period = {u: q["name"] for q in S["periods"] for u in q["uids"]}
    bad = set(S["bad_fids"])
    nfr, nfr_ok = collections.Counter(), collections.Counter()
    for l in open(a.frames_meta):
        f = json.loads(l)
        nfr[f["uid"]] += 1
        nfr_ok[f["uid"]] += f["fid"] not in bad

    groups = collections.OrderedDict()
    for r in map(json.loads, open(a.meta)):
        if r["node"] in nodes:
            groups.setdefault((r["node"], r["start_time"], r["stop_time"]), []).append(r)
    ev = []
    for (node, st, sp), rs in groups.items():
        rs.sort(key=lambda r: -nfr[r["unique_data_id"]])
        r0, P = rs[0], nodes[node]
        uids = [r["unique_data_id"] for r in rs]
        steps = next((clean_steps(r["reasoning_steps"]) for r in rs if clean_steps(r["reasoning_steps"])), [])
        h = int(st[11:13])
        ev.append({"event_id": f"{P}:{r0['unique_data_id']}", "person": P, "uids": uids,
                   "period": next((period[u] for u in uids if u in period), None) if P == "A" else P,
                   "start": st, "stop": sp, "date": st[:10], "hour": h, "part_of_day": part_of_day(h),
                   "dur_s": int((datetime.fromisoformat(sp) - datetime.fromisoformat(st)).total_seconds()),
                   "site": r0["application_website"], "category": "web" if r0["category"] == "BROWSER_TASK" else "desktop",
                   "taskIds": sorted({r["taskId"] for r in rs}), "instruction": clean_instr(r0["task_name"]), "raw_task_name": r0["task_name"],
                   "steps": steps, "n_frames": sum(nfr[u] for u in uids), "n_frames_pool": max(nfr_ok[u] for u in uids)})
    ev.sort(key=lambda e: (e["person"], e["start"]))
    for P in set(nodes.values()):
        for i, e in enumerate(x for x in ev if x["person"] == P):
            e["idx"] = i
    with open(a.out, "w") as f:
        for e in ev:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    for P in sorted(set(nodes.values())):
        es = [e for e in ev if e["person"] == P]
        print(P, "events", len(es), "| recordings", sum(len(e["uids"]) for e in es), "| with steps", sum(bool(e["steps"]) for e in es))


if __name__ == "__main__":
    main()
