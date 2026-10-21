"""Shared helpers for the question pipeline: JSONL I/O, site / time helpers, the two LLM callers, and the competitor search
used by the uniqueness judge (s6)."""
import collections
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime

import numpy as np

GENERIC_SITES = {"searchengine", "webbrowser", "other", "noapplication", "windowsos", "filemanager", "browser", "systemsettings",
                 "email", "calendar"}
ALIAS = {"googlemap": "googlemaps", "googlesearch": "google"}
MONTHS = "january february march april may june july august september october november december".split()


# ---------------------------------------------------------------- I/O
def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


_io_lock = threading.Lock()


def append_jsonl(path, row):
    with _io_lock, open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_cache(path, key):
    """Resumable raw-output files: {key value: last row with that key}."""
    return {r[key]: r for r in read_jsonl(path)} if os.path.exists(path) else {}


# ---------------------------------------------------------------- sites and times
def site_key(s):
    k = re.sub(r"[^a-z0-9]", "", (s or "").lower())
    return ALIAS.get(k, k)


def is_generic(s):
    return site_key(s) in GENERIC_SITES


def nice_date(d):
    x = datetime.fromisoformat(d)
    return x.strftime("%A, %B ") + str(x.day) + ", " + str(x.year)


def words(t):
    return re.findall(r"[a-z0-9$%.']+", (t or "").lower())


def longest_common_run(a, b):
    """Length of the longest run of consecutive words shared by two texts."""
    a, b = words(a), words(b)
    best, prev = 0, [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def load_prompt(name):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", name)) as f:
        return f.read()


# ---------------------------------------------------------------- LLM callers
COST = {"usd": 0.0, "calls": 0, "fail": 0}
_cost_lock = threading.Lock()


def claude_json(system, prompt, model, timeout=900, tries=3):
    """Run `claude -p` headless (no tools, no MCP) and parse the first JSON array/object of the answer; None on failure."""
    cwd = os.path.join(os.environ.get("TMPDIR", "/tmp"), "screenret_empty_cwd")
    os.makedirs(cwd, exist_ok=True)
    for attempt in range(tries):
        try:
            p = subprocess.run(["claude", "-p", prompt, "--model", model, "--system-prompt", system, "--tools", "", "--strict-mcp-config",
                                "--output-format", "json"], capture_output=True, text=True, timeout=timeout, cwd=cwd)
            d = json.loads(p.stdout)
            with _cost_lock:
                COST["usd"] += d.get("total_cost_usd") or 0
                COST["calls"] += 1
            txt = d.get("result") or ""
            m = re.search(r"[\[{].*[\]}]", txt, re.S)
            return json.loads(m.group(0))
        except Exception:
            time.sleep(5 * (attempt + 1))
    with _cost_lock:
        COST["fail"] += 1
    return None


# ---------------------------------------------------------------- competitors (uniqueness judge)
_STOP = {"microsoft", "google", "windows", "online", "app", "apps", "macos", "system", "preferences", "tool", "and", "the"}


def _site_words(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) >= 4 and w not in _STOP}


def _norm_instr(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())


class Competitors:
    """Other events of the SAME person that a question about event e could also describe:
    (a) repeated : same taskId, identical instruction, or task-text embedding cosine >= 0.93
    (b) same_day : same site (or a shared site word such as "excel") on the same date; for generic sites any event within 90 min
    (c) site_sim : same site, top-10 by task-text similarity
    (d) xsite_sim: other sites, top-8 by similarity with cosine >= 0.6 (matters for "where" questions)
    Ordered repeated > same_day > similarity, capped at 30."""

    def __init__(self, diary, task_emb):
        self.E = diary
        self.row = {e["event_id"]: i for i, e in enumerate(diary)}
        self.X = np.asarray(task_emb, np.float32)          # L2-normalised rows (screenret.embed_text output), used as stored

    def __call__(self, e, exclude=(), cap=30):
        excl = {e["event_id"], *exclude}
        P = [x for x in self.E if x["person"] == e["person"] and x["event_id"] not in excl]
        sim = self.X[[self.row[x["event_id"]] for x in P]] @ self.X[self.row[e["event_id"]]]
        c = collections.OrderedDict()

        def add(x, why, s):
            d = c.setdefault(x["event_id"], {"cid": x["event_id"], "why": [], "sim": round(float(s), 3)})
            if why not in d["why"]:
                d["why"].append(why)

        t0 = datetime.fromisoformat(e["start"])
        sk, sw = site_key(e["site"]), _site_words(e["site"])
        for x, s in zip(P, sim):
            if set(x["taskIds"]) & set(e["taskIds"]) or _norm_instr(x["instruction"]) == _norm_instr(e["instruction"]) or s >= 0.93:
                add(x, "repeated", s)
        for x, s in zip(P, sim):
            same_site = site_key(x["site"]) == sk or (sw & _site_words(x["site"]))
            near = abs((datetime.fromisoformat(x["start"]) - t0).total_seconds()) <= 5400
            if x["date"] == e["date"] and (same_site or (is_generic(e["site"]) and near)):
                add(x, "same_day", s)
        same = [(s, x) for x, s in zip(P, sim) if site_key(x["site"]) == sk]
        for s, x in sorted(same, key=lambda z: -z[0])[:10]:
            add(x, "site_sim", s)
        other = [(s, x) for x, s in zip(P, sim) if site_key(x["site"]) != sk and s >= 0.6]
        for s, x in sorted(other, key=lambda z: -z[0])[:8]:
            add(x, "xsite_sim", s)
        order = sorted(c.values(), key=lambda d: ("repeated" not in d["why"], "same_day" not in d["why"], -d["sim"]))
        return order[:cap]


def annotate(x, cid):
    """An event as the judge sees it."""
    d = {"cid": cid, "site_or_app": x["site"], "kind": "website" if x["category"] == "web" else "desktop app",
         "when": f"{nice_date(x['date'])}, {x['start'][11:16]}-{x['stop'][11:16]}", "instruction": x["instruction"][:400]}
    if x["steps"]:
        d["steps"] = [s[:100] for s in x["steps"][:4]]
    return d
