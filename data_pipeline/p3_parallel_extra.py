"""Parallel step 3: extra parallel questions (one "round" per call) -> questions_multihop_<round>.jsonl.

Hops are verified single-hop facts / diary records; the LLM (paper run: Claude Opus 5.5 through the `claude` CLI) only words and checks.
Evidence sets already used by earlier rounds (--exclude-specs: the p2 specs and earlier rounds' specs) are skipped; duplicates across
rules are kept once. Specs, per person (A first, then B), for every window kind in --windows (month | week | day):
  aggregate, label : (site, action label, window), 2-8 events; dropped if the person has an unlabelled event on that site in that
                     window (the count must be exact).                      "How many times in <window> did I <action> on <site>?"
  aggregate, site  : site only, non-generic site whose name has no "," or "/", (site, window), 2-8 events; dropped if an event in
                     that window is unlabelled on that site or names that site inside a compound site field.
                                                                            "How many times in <window> did I use <site>?"
  compare, qtype   : (unless --no-compare) two anchored events whose kept number facts have the same QUANTITY TYPE (price cap, minimum
                     reviews, nights, prep/cook-time limit, calorie cap, servings, timer length, star rating), different values, same
                     site or same day; <= 3 specs per event before checks, <= 2 kept per event (counting p2's kept compares).
  --kinds freq2    : "did I use site X or site Y more often in <month>?" over two exact site-only month groups (same diary category,
                     counts differ, 3-8 events in total, each site-month used <= 2 times).
  --kinds dur2     : "which of tasks X and Y took longer?" (anchored, same day or site, different labels, longer >= 2x shorter and
                     >= 60 s longer, <= 2 uses per event; the checker sees end times and durations).   <= --cap-new per kind and person.
Checks: compare / duration as in p2 (all evidence + distractors answerable, not answerable after dropping a hop); aggregate / freq2
  by two recounts that must return exactly the group (label: all the person's events on that site in that window; site / freq2: the
  group + the person's other events on the same dates, nearest 40).
Paper rounds (in this order; each excludes the specs of p2 and of the earlier rounds):
  extra  : --mid-prefix MX                                                   (month + week windows, compare)
  extra2 : --mid-prefix MX2 --windows day --no-compare
  extra3 : --mid-prefix MX3 --windows "" --no-compare --kinds freq2,dur2

example: python data_pipeline/p3_parallel_extra.py --diary diary.jsonl --single-all single_all.jsonl --action-labels action_labels.jsonl \
             --p2-all multihop_all.jsonl --exclude-specs multihop_specs.jsonl --mid-prefix MX --raw extra_raw.jsonl --checks extra_checks.jsonl \
             --recount extra_recount.jsonl --out-specs extra_specs.jsonl --out-all extra_all.jsonl --out questions_multihop_extra.jsonl
"""
import argparse
import collections
import itertools
import os
import random
import re
import sys
from datetime import date, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import COST, is_generic, load_prompt, nice_date, read_jsonl, site_key, write_jsonl  # noqa: E402
from parallel_lib import Facts, check, leaks, recount, recount_ok, start, word  # noqa: E402

WORD_SYSTEM = load_prompt("parallel_extra_wording.txt")
RECOUNT_SYSTEM = load_prompt("parallel_extra_recount.txt")
WIN = {"month": lambda d: d[:7], "week": lambda d: "%d-W%02d" % date.fromisoformat(d).isocalendar()[:2], "day": lambda d: d}


def win_phrase(kind, key):
    if kind == "month":
        return datetime.strptime(key, "%Y-%m").strftime("%B %Y")
    if kind == "day":
        return "on " + nice_date(key)
    y, w = map(int, key.split("-W"))
    return "the week of " + nice_date(date.fromisocalendar(y, w, 1).isoformat())


# ---------------------------------------------------------------- quantity typing for compare
W2N = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}
QNAME = {"price_cap": "price limit / budget cap", "min_reviews": "minimum number of reviews", "nights": "number of nights",
         "time_limit_min": "prep/cook-time limit", "calorie_cap": "calorie limit", "servings": "number of servings",
         "timer_len": "timer / recording length", "star_rating": "star rating given"}


def qtype(q, a):
    """Quantity type of a number fact from its slot question and answer (None = not comparable)."""
    q, a = q.lower(), a.lower()
    cur = re.search(r"[$£]|dollar|usd", a)
    if re.search(r"zip|year|date|version|what time|start time|age\b", q):
        return None
    if cur and re.search(r"cap|limit|budget|ceiling|maximum|max\b|spending", q) and not re.search(r"income|loan|debt|valuation|balance", q):
        return "price_cap"
    if re.search(r"review", q) and re.search(r"minimum|at least|more than|over|exceed|require|need|threshold|surpass|insist|want", q) \
            and not re.search(r"[$€£]|dollar|star", a):
        return "min_reviews"
    if re.search(r"nights?", q) and not re.search(r"day|price|\$", a):
        return "nights"
    if re.search(r"prep time|cook time", q) and re.search(r"minute|hour", a):
        return "time_limit_min"
    if "calorie" in q and "calorie" in a:
        return "calorie_cap"
    if re.search(r"serve|servings", q) and re.fullmatch(r"\d+", a.strip()):
        return "servings"
    if re.search(r"countdown|timer|voice memo|yoga routine", q) and re.search(r"minute|second", a):
        return "timer_len"
    if re.search(r"star", q) and re.search(r"star", a) and re.search(r"rate|rating|give|gave", q):
        return "star_rating"
    return None


def qval(a, qt):
    """The single numeric value of an answer (minutes for time quantities); None for ranges or several numbers."""
    a = a.lower().replace(",", "")
    if re.search(r"\d\s*(to|-|–)\s*\$?\d", a):
        return None
    ns = re.findall(r"\d+(?:\.\d+)?", a)
    if not ns:
        w = [W2N[x] for x in re.findall(r"[a-z]+", a) if x in W2N]
        return float(w[0]) if len(w) == 1 else None
    if len(ns) != 1:
        return None
    v = float(ns[0])
    if qt in ("time_limit_min", "timer_len") and "hour" in a:
        v *= 60
    if qt in ("time_limit_min", "timer_len") and "second" in a:
        v /= 60
    return v


def compound_parts(full, wf):
    """(site part key, window) -> events whose compound site field ("A, B" / "A/B") names that site."""
    comp = collections.defaultdict(set)
    for e in full:
        if re.search(r"[,/]", e["site"] or ""):
            for part in re.split(r"[,/]", e["site"]):
                comp[(site_key(part.strip()), wf(e["date"]))].add(e["event_id"])
    return comp


def make_specs(F, a, old_sets, old_cmp_use):
    specs, seen = [], set(old_sets)
    windows = [w for w in a.windows.split(",") if w]
    kinds = set(filter(None, a.kinds.split(",")))
    for p in a.persons.split(","):
        full = sorted([e for e in F.E if e["person"] == p], key=lambda e: e["start"])
        lab = [e for e in full if e["event_id"] in F.LAB]
        for wk in windows:
            wf = WIN[wk]
            unl = {(site_key(e["site"]), wf(e["date"])) for e in full if e["event_id"] not in F.LAB}
            comp = compound_parts(full, wf)
            g = collections.defaultdict(list)                                   # label aggregate
            for e in lab:
                g[(site_key(e["site"]), F.LAB[e["event_id"]], wf(e["date"]))].append(e)
            for (sk, act, wkey), L in sorted(g.items()):
                ev = frozenset(e["event_id"] for e in L)
                if not 2 <= len(L) <= 8 or (sk, wkey) in unl or ev in seen:
                    continue
                seen.add(ev)
                specs.append({"kind": "aggregate", "variant": f"label_{wk}", "person": p, "structure": "parallel", "events": [e["event_id"] for e in L],
                              "action": act, "site": L[0]["site"], "window": win_phrase(wk, wkey), "window_kind": wk, "window_key": wkey,
                              "answer": str(len(L))})
            g = collections.defaultdict(list)                                   # site-only aggregate
            for e in lab:
                if is_generic(e["site"]) or re.search(r"[,/]", e["site"] or ""):
                    continue
                g[(site_key(e["site"]), wf(e["date"]))].append(e)
            for (sk, wkey), L in sorted(g.items()):
                ev = frozenset(e["event_id"] for e in L)
                if not 2 <= len(L) <= 8 or (sk, wkey) in unl or comp.get((sk, wkey), set()) - ev or ev in seen:
                    continue
                if any(site_key(e["site"]) == sk and wf(e["date"]) == wkey and e["event_id"] not in ev for e in full):
                    continue                                                    # e.g. a generic-flagged twin
                seen.add(ev)
                specs.append({"kind": "aggregate", "variant": f"site_{wk}", "person": p, "structure": "parallel", "events": [e["event_id"] for e in L],
                              "site": L[0]["site"], "window": win_phrase(wk, wkey), "window_kind": wk, "window_key": wkey, "answer": str(len(L))})
        if "freq2" in kinds:                                                    # two-site frequency compare (month)
            wf = WIN["month"]
            unl = {(site_key(e["site"]), wf(e["date"])) for e in full if e["event_id"] not in F.LAB}
            comp = compound_parts(full, wf)
            g = collections.defaultdict(list)
            for e in lab:
                if is_generic(e["site"]) or re.search(r"[,/]", e["site"] or ""):
                    continue
                g[(site_key(e["site"]), wf(e["date"]))].append(e)
            exact = {}
            for (sk, wkey), L in g.items():
                ev = {e["event_id"] for e in L}
                if (sk, wkey) in unl or comp.get((sk, wkey), set()) - ev:
                    continue
                if any(site_key(e["site"]) == sk and wf(e["date"]) == wkey and e["event_id"] not in ev for e in full):
                    continue
                exact[(sk, wkey)] = L
            cand = [(k1, L1, k2, L2) for (k1, L1), (k2, L2) in itertools.combinations(sorted(exact.items()), 2)
                    if k1[1] == k2[1] and len(L1) != len(L2) and 3 <= len(L1) + len(L2) <= 8 and L1[0]["category"] == L2[0]["category"]]
            random.Random(1).shuffle(cand)
            used, n = collections.Counter(), 0
            for k1, L1, k2, L2 in cand:
                ev = frozenset(e["event_id"] for e in L1 + L2)
                if n >= a.cap_new or used[k1] >= 2 or used[k2] >= 2 or ev in seen:
                    continue
                used.update([k1, k2])
                seen.add(ev)
                n += 1
                more, less = (L1, L2) if len(L1) > len(L2) else (L2, L1)
                specs.append({"kind": "freq2", "variant": "freq2_month", "person": p, "structure": "parallel", "events": sorted(ev),
                              "sites": [L1[0]["site"], L2[0]["site"]], "window": win_phrase("month", k1[1]), "window_kind": "month",
                              "window_key": k1[1], "answer": f"{more[0]['site']} ({len(more)} vs {len(less)} times)"})
        if "dur2" in kinds:                                                     # duration compare
            an = [e for e in lab if e["event_id"] in F.anchor and e.get("dur_s")]
            cand = []
            for x, y in itertools.combinations(an, 2):
                if x["date"] != y["date"] and site_key(x["site"]) != site_key(y["site"]):
                    continue
                if F.LAB[x["event_id"]] == F.LAB[y["event_id"]]:
                    continue
                lo, hi = sorted([x, y], key=lambda e: e["dur_s"])
                if hi["dur_s"] < 2 * lo["dur_s"] or hi["dur_s"] - lo["dur_s"] < 60:
                    continue
                cand.append((x, y, hi))
            random.Random(2).shuffle(cand)
            used, n = collections.Counter(), 0
            for x, y, hi in cand:
                ev = frozenset([x["event_id"], y["event_id"]])
                if n >= a.cap_new or used[x["event_id"]] >= 2 or used[y["event_id"]] >= 2 or ev in seen:
                    continue
                used.update(ev)
                seen.add(ev)
                n += 1
                specs.append({"kind": "duration", "variant": "dur2", "person": p, "structure": "parallel", "events": [x["event_id"], y["event_id"]],
                              "answer": f"{F.anchor[hi['event_id']]} ({x['dur_s'] / 60:.1f} min vs {y['dur_s'] / 60:.1f} min)",
                              "longer_event": hi["event_id"]})
        if a.no_compare:                                                        # quantity-typed compare
            continue
        facts = [(e, x) for e in lab if e["event_id"] in F.anchor for x in F.inputs.get(e["event_id"], []) if x["slot"] == "number"]
        ok = [(e, x, qt, qval(x["answer"], qt)) for e, x in facts for qt in [qtype(x["question"], x["answer"])] if qt]
        ok = [r for r in ok if r[3] is not None]
        pairs = []
        for (x, xa, ta, va), (y, xb, tb, vb) in itertools.combinations(ok, 2):
            if x["event_id"] == y["event_id"] or ta != tb or va == vb:
                continue
            if site_key(x["site"]) != site_key(y["site"]) and x["date"] != y["date"]:
                continue
            if frozenset([x["event_id"], y["event_id"]]) in seen:
                continue
            pairs.append((x, xa, y, xb, ta, va, vb))
        random.Random(0).shuffle(pairs)
        use = collections.Counter(old_cmp_use)
        for x, xa, y, xb, qt, va, vb in pairs:
            if use[x["event_id"]] >= 3 or use[y["event_id"]] >= 3:
                continue
            use.update([x["event_id"], y["event_id"]])
            seen.add(frozenset([x["event_id"], y["event_id"]]))
            hi = x if va > vb else y
            specs.append({"kind": "compare", "variant": f"qtype:{qt}", "qtype": qt, "person": p, "structure": "parallel",
                          "events": [x["event_id"], y["event_id"]], "asks": [xa, xb],
                          "answer": f"{F.anchor[hi['event_id']]} ({xa['answer']} vs {xb['answer']})", "higher_event": hi["event_id"]})
    for k, s in enumerate(specs):
        s["mid"] = f"{a.mid_prefix}-{s['person']}-{s['kind']}-{k:04d}"
    return specs


def wording_item(F):
    def wspec(k, s):
        if s["kind"] == "freq2":
            return {"id": k, "kind": "compare_freq", "sites": s["sites"], "window": s["window"]}
        if s["kind"] == "duration":
            return {"id": k, "kind": "compare_duration", "cues": [F.anchor[e] for e in s["events"]]}
        if s["kind"] == "compare":
            return {"id": k, "kind": "compare", "quantity": QNAME[s["qtype"]], "cues": [F.anchor[e] for e in s["events"]],
                    "slot_questions": [x["question"] for x in s["asks"]]}
        if s["variant"].startswith("site"):
            return {"id": k, "kind": "aggregate_site", "site": s["site"], "window": s["window"]}
        return {"id": k, "kind": "aggregate_label", "action": s["action"], "site": s["site"], "window": s["window"],
                "example_tasks": [F.EV[e]["instruction"][:120] for e in s["events"][:3]]}
    return wspec


def recount_pool(F):
    def pool(s):
        p, wf = s["person"], WIN[s["window_kind"]]
        if s["variant"].startswith("label"):
            sk = site_key(s["site"])
            return [e for e in F.E if e["person"] == p and site_key(e["site"]) == sk and wf(e["date"]) == s["window_key"]]
        grp = set(s["events"])
        dates = {F.EV[x]["date"] for x in grp}
        others = [e for e in F.E if e["person"] == p and e["date"] in dates and e["event_id"] not in grp]
        others.sort(key=lambda e: min(abs((start(e) - start(F.EV[x])).total_seconds()) for x in grp))
        return [F.EV[x] for x in grp] + others[:max(0, 40 - len(grp))]
    return pool


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--diary", required=True)
    p.add_argument("--single-all", required=True)
    p.add_argument("--action-labels", required=True)
    p.add_argument("--p2-all", required=True, help="p2 multihop_all.jsonl (its kept compares count toward the per-event compare limit)")
    p.add_argument("--exclude-specs", nargs="*", default=[], help="spec files of p2 and earlier rounds; their evidence sets are skipped")
    p.add_argument("--mid-prefix", default="MX", help="question-id prefix of this round (paper: MX, MX2, MX3)")
    p.add_argument("--persons", default="A,B")
    p.add_argument("--windows", default="month,week", help='aggregate windows, comma-separated (month, week, day; "" = none)')
    p.add_argument("--no-compare", action="store_true", help="skip the quantity-typed compare specs")
    p.add_argument("--kinds", default="", help="extra kinds: freq2, dur2")
    p.add_argument("--cap-new", type=int, default=100, help="per kind and person, for freq2 / dur2")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--raw", required=True)
    p.add_argument("--checks", required=True)
    p.add_argument("--recount", required=True)
    p.add_argument("--out-specs", required=True)
    p.add_argument("--out-all", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    F = Facts(a.single_all, a.action_labels, read_jsonl(a.diary))
    old_sets = set()
    for fp in a.exclude_specs:
        old_sets |= {frozenset(s["events"]) for s in read_jsonl(fp) if s["kind"] in ("aggregate", "compare", "order", "freq2", "duration")}
    old_cmp_use = collections.Counter(g["event"] for r in read_jsonl(a.p2_all) if r["type"] == "compare" and r["status"] == "kept"
                                      for g in r["evidence_groups"])
    specs = make_specs(F, a, old_sets, old_cmp_use)
    write_jsonl(a.out_specs, specs)
    print("specs", dict(collections.Counter((s["person"], s["variant"].split(":")[0]) for s in specs)), flush=True)

    R = word(specs, wording_item(F), WORD_SYSTEM, a.raw, a.model)
    rows = [R[s["mid"]] for s in specs if "question" in R.get(s["mid"], {})]
    CK = check([s for s in rows if s["kind"] in ("compare", "duration")], F, a.checks, a.model, dur_kinds=("duration",))
    RC = recount([s for s in rows if s["kind"] in ("aggregate", "freq2")], recount_pool(F), F, RECOUNT_SYSTEM, a.recount, a.model)

    out, reasons_ct, used_cmp = [], collections.Counter(), collections.Counter(old_cmp_use)
    for s in sorted(rows, key=lambda s: s["mid"]):
        reasons = []
        if leaks(s["question"], [x["answer"] for x in s.get("asks", [])]):
            reasons.append("leak")
        if s["kind"] in ("aggregate", "freq2"):
            if not recount_ok(RC, s):
                reasons.append("recount_mismatch")
        else:
            c = CK.get((s["mid"], "all"))
            if not c or not c["match"]:
                reasons.append("not_answerable_with_all_evidence")
            if any(CK.get((s["mid"], f"drop:{h}"), {}).get("match") for h in s["events"]):
                reasons.append("shortcut_drop_one")
            if not reasons and s["kind"] == "compare":
                if any(used_cmp[e] >= 2 for e in s["events"]):
                    reasons.append("compare_event_reused")
                else:
                    used_cmp.update(s["events"])
        reasons_ct.update(reasons)
        out.append({"qid": s["mid"], "hop": "multi", "structure": "parallel", "type": s["kind"], "variant": s["variant"], "person": s["person"],
                    "question": s["question"], "answer": s["answer"],
                    "evidence_groups": [F.group(f"hop{i + 1}", e) for i, e in enumerate(s["events"])],
                    "status": "deleted" if reasons else "kept", "delete_reasons": reasons})
    write_jsonl(a.out_all, out)
    kept = [{k: v for k, v in r.items() if k not in ("status", "delete_reasons")} for r in out if r["status"] == "kept"]
    write_jsonl(a.out, kept)
    print("worded", len(out), "| kept", dict(collections.Counter(f"{r['person']}/{r['variant'].split(':')[0]}" for r in kept)),
          "| reasons", dict(reasons_ct), f"| cost ${COST['usd']:.2f}")


if __name__ == "__main__":
    main()
