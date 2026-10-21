"""Step 7: the single-hop test set from the judged candidates -> questions_single.jsonl (+ single_all.jsonl for p2/p3, stats).

Kept = s6 status "kept", anchors excluded (they only serve as task references for the parallel questions). Each question records its
cue: cue_parts = which cues the wording uses (content = task gist, app = the site named, time = a date/part of day); cue_type =
content / mixed (gist + site and/or time) / app / time. single_all.jsonl keeps every candidate with its status and delete reasons
(the parallel builders read kept anchors and input values from it). --person filters questions_single.jsonl (release: A).

example: python data_pipeline/s7_build_single.py --judged judged.jsonl --single-all single_all.jsonl --out questions_single.jsonl
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MONTHS, is_generic, read_jsonl, site_key, write_jsonl  # noqa: E402

TIME_RE = re.compile(r"\b(" + "|".join(MONTHS + [m[:3] for m in MONTHS if m != "may"]) + r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
                     r"|morning|afternoon|evening|night|tonight|yesterday)\b|\b\d{1,2}(:\d\d)?\s*(am|pm)\b|\bmay \d", re.I)
GENERIC_WORDS = {"microsoft", "google", "windows", "online", "settings", "setting", "tool", "tools", "apps", "account", "search", "player",
                 "media", "client", "desktops", "options", "system", "network", "privacy", "university", "news", "maps", "timer", "clock",
                 "email", "browser", "preferences", "macos", "virtual", "engine", "dictionary", "web"}


def names_site(q, s):
    if is_generic(s):
        return False
    ql, sk = q.lower(), site_key(s)
    return (len(sk) >= 4 and sk in re.sub(r"[^a-z0-9]", "", ql)) or any(
        re.search(r"\b" + w + r"\b", ql) for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) >= 4 and w not in GENERIC_WORDS)


def cue(q, content, site):
    parts = (["content"] if content else []) + (["app"] if names_site(q, site) else []) + (["time"] if TIME_RE.search(q) else [])
    return parts, ("content" if parts == ["content"] else "mixed" if "content" in parts else "app" if "app" in parts else "time")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judged", required=True, help="judged.jsonl from s6")
    p.add_argument("--single-all", required=True, help="output: every candidate with status")
    p.add_argument("--person", default="", help="restrict questions_single.jsonl to these persons, e.g. A (default: all)")
    p.add_argument("--stats", help="output stats JSON")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    allq, kept = [], []
    for c in read_jsonl(a.judged):
        parts, t = cue(c["question"] or "", c["type"] != "what", c["site"])
        r = {"qid": "S-" + c["qid"].replace("#", "-"), "hop": "single", "type": c["type"], "slot": c.get("slot"), "person": c["person"],
             "period": c["period"], "category": c["category"], "site": c["site"], "question": c["question"], "answer": c["answer"],
             "evidence_groups": [{"role": "target", "event": c["event_id"], "uids": c["gold_uids"], "category": c["category"]}],
             "n_gold_recordings": len(c["gold_uids"]), "cue_parts": parts, "cue_type": t, "flags": c["flags"], "status": c["status"],
             "delete_reasons": c["delete_reasons"], "n_competitors_checked": c.get("n_competitors_checked"),
             "competitors_diff": [{k: v.get(k) for k in ("cid", "why_found", "value", "why")} for v in c.get("competitors_diff") or []],
             "competitors_same": [{k: v.get(k) for k in ("cid", "why_found", "value", "why")} for v in c.get("competitors_same") or []]}
        allq.append(r)
        if c["status"] == "kept" and c["type"] != "anchor":
            kept.append(r)
    write_jsonl(a.single_all, allq)
    persons = set(a.person.split(",")) if a.person else None
    rel = [r for r in kept if persons is None or r["person"] in persons]
    write_jsonl(a.out, [{k: v for k, v in r.items() if k not in ("status", "delete_reasons", "competitors_diff")} for r in rel])
    st = {"candidates": collections.Counter(r["type"] for r in allq), "kept": collections.Counter(r["type"] for r in allq if r["status"] == "kept"),
          "kept_by_person": collections.Counter(f"{r['person']}/{r['type']}" for r in kept),
          "kept_by_cue": collections.Counter(f"{r['type']}/{r['cue_type']}" for r in kept),
          "reasons": collections.Counter(x for r in allq for x in r["delete_reasons"]),
          "extra_gold": sum(r["n_gold_recordings"] > 1 for r in kept)}
    if a.stats:
        json.dump(st, open(a.stats, "w"), indent=1)
    for k, v in st.items():
        print(k, dict(v) if isinstance(v, collections.Counter) else v)
    print(f"questions_single: {len(rel)} ({a.person or 'all persons'})")


if __name__ == "__main__":
    main()
