"""Retrieval accuracy of frame-vector runs on a fixed question set: single-hop R@k, parallel multi-hop recall@k, paired bootstrap.

Protocol JSON (scripts/make_eval_protocol.py):
  pool        frame ids searched (paper: person A's T1-T5 frames minus bad frames = 63,538)
  pool_group  recording id of each pool frame
  questions   [{"qid", "row" (row in --queries), "type": "single" | "parallel", "groups": [[recording ids], ...]}]
              (paper: 4,265 single-hop + 846 parallel person-A questions; an old {"gold": [...]} entry = one single-hop group)
Single-hop: gold = every pool frame of the question's recording(s); rank = 1 + #(non-gold pool frames scoring above the best gold frame);
            R@k = mean(rank <= k).
Parallel (several evidence groups, each one or more recordings): per group, rank = 1 + #(pool frames scoring above the best frame of that
            group); ALL pool frames compete, including other groups' gold frames. Per question recall@k = mean over groups of [rank <= k];
            PR@k = its mean over questions (paper: "parallel R@10"); PA@k = strict, mean of [every group within k].
Scores are raw inner products of the stored vectors (primary) and, as a secondary number, of L2-normalised ones.
Deltas vs the reference run: paired bootstrap over questions (B draws, seed 0) and a recording-clustered bootstrap (clusters = first
recording of the first group), separately for the single-hop and the parallel set.

example: python -m screenret.evaluate --protocol protocol.json --queries q_lora.npy \
             --frames FULL=runs/full_frames.npy S+T+L=runs/LT8TL_frames.npy --reference FULL --out results.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

KS = (1, 5, 10)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m screenret.evaluate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol", required=True, help="protocol JSON (pool, pool_group, questions)")
    p.add_argument("--queries", required=True, help="query vectors .npy; question 'row' indexes it")
    p.add_argument("--frames", nargs="+", required=True, help="[NAME=]PATH of id-indexed frame-vector matrices (.npy)")
    p.add_argument("--reference", default=None, help="NAME (or PATH) of the reference run for deltas (default: the first --frames)")
    p.add_argument("--n-boot", type=int, default=10000, help="paired bootstrap draws (default 10000)")
    p.add_argument("--n-cluster-boot", type=int, default=2000, help="recording-clustered bootstrap draws (default 2000; 0 = off)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="results JSON (per-run metrics, deltas, CIs, ranks)")
    return p.parse_args(argv)


def question_groups(q: dict) -> list:
    """Evidence groups of a question (old protocols: "gold" = one group)."""
    return q["groups"] if "groups" in q else [q["gold"]]


def split_questions(protocol: dict) -> tuple[list, list]:
    qs = protocol["questions"]
    single = [q for q in qs if q.get("type", "single") == "single"]
    parallel = [q for q in qs if q.get("type", "single") == "parallel"]
    assert len(single) + len(parallel) == len(qs), "question type must be 'single' or 'parallel'"
    return single, parallel


def gold_matrix(pool_group: np.ndarray, questions: list) -> np.ndarray:
    """Single-hop: (Q, N) bool, gold = pool frames of any recording of the question."""
    return np.stack([np.isin(pool_group, sum(question_groups(q), [])) for q in questions])


def ranks(Q: np.ndarray, X: np.ndarray, G: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Single-hop: 1-based rank of the best gold frame among non-gold frames, per question (chunks of questions bound the memory)."""
    out = []
    for b in range(0, len(Q), chunk):
        S, g = Q[b:b + chunk] @ X.T, G[b:b + chunk]
        best = np.where(g, S, -np.inf).max(1)
        out.append(((S > best[:, None]) & ~g).sum(1) + 1)
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def group_index(pool_group: np.ndarray, questions: list) -> list:
    """Parallel: per question, per group, the pool positions of the group's recordings."""
    pos: dict = {}
    for i, u in enumerate(pool_group.tolist()):
        pos.setdefault(u, []).append(i)
    out = []
    for q in questions:
        gi = [np.array(sorted(i for u in g for i in pos.get(u, [])), dtype=np.int64) for g in question_groups(q)]
        assert all(len(g) for g in gi), f"{q['qid']}: an evidence group has no frame in the pool"
        out.append(gi)
    return out


def group_ranks(Q: np.ndarray, X: np.ndarray, gidx: list) -> list:
    """Parallel: per question, per group, 1 + #(all pool frames scoring above the group's best frame)."""
    S = Q @ X.T
    return [[int((s > s[g].max()).sum()) + 1 for g in gi] for s, gi in zip(S, gidx)]


def recall(rk: np.ndarray) -> dict:
    return {f"R{k}": float((rk <= k).mean() * 100) for k in KS}


def parallel_hits(grk: list) -> dict:
    """Per question: recall@k = fraction of groups within k ("PR"), strict = all groups within k ("PA")."""
    out = {}
    for k in KS:
        out[f"PR{k}"] = np.array([np.mean([r <= k for r in g]) for g in grk])
        out[f"PA{k}"] = np.array([float(max(g) <= k) for g in grk])
    return out


def bootstrap_index(n: int, clusters: np.ndarray, n_boot: int, n_cluster: int, seed: int):
    """Index matrices for the paired bootstrap (questions) and the clustered bootstrap (whole recordings), drawn in this order."""
    rng = np.random.default_rng(seed)
    bi = rng.integers(0, n, (n_boot, n))
    uc, ci = np.unique(clusters, return_inverse=True)
    cb = [np.concatenate([np.nonzero(ci == j)[0] for j in rng.integers(0, len(uc), len(uc))]) for _ in range(n_cluster)]
    return bi, cb


def delta(hit: np.ndarray, ref: np.ndarray, bi: np.ndarray, cb: list) -> dict:
    d = hit.astype(np.float64) - ref.astype(np.float64)
    bs = d[bi].mean(1) * 100
    o = dict(delta=float(d.mean() * 100), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
             win_loss=[int((d > 0).sum()), int((d < 0).sum())])
    if cb:
        cs = np.array([d[ix].mean() for ix in cb]) * 100
        o["ci_clustered"] = [float(np.percentile(cs, 2.5)), float(np.percentile(cs, 97.5))]
    return o


def named(spec: str) -> tuple[str, str]:
    return tuple(spec.split("=", 1)) if "=" in spec else (spec, spec)


def evaluate_run(X: np.ndarray, Qs: np.ndarray, Gs: np.ndarray, Qp: np.ndarray, gidx: list) -> tuple[dict, dict]:
    """Metrics of one frame-vector run (raw and L2-normalised scores) + per-question hit arrays for the bootstrap."""
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    o, hits = {}, {}
    if len(Qs):
        rk, rk_n = ranks(Qs, X, Gs), ranks(Qs, Xn, Gs)
        o["single"] = dict(**recall(rk), **{f"{k}_l2norm": v for k, v in recall(rk_n).items()}, ranks=rk.tolist())
        hits.update({f"R{k}": rk <= k for k in KS})
    if len(Qp):
        grk, grk_n = group_ranks(Qp, X, gidx), group_ranks(Qp, Xn, gidx)
        ph, ph_n = parallel_hits(grk), parallel_hits(grk_n)
        o["parallel"] = dict(**{k: float(v.mean() * 100) for k, v in ph.items()},
                             **{f"{k}_l2norm": float(v.mean() * 100) for k, v in ph_n.items()}, group_ranks=grk)
        hits.update(ph)
    return o, hits


def main(argv=None) -> None:
    a = parse_args(argv)
    proto = json.load(open(a.protocol))
    single, parallel = split_questions(proto)
    pool = np.asarray(proto["pool"], dtype=np.int64)
    pool_group = np.asarray(proto["pool_group"])
    Gs = gold_matrix(pool_group, single) if single else np.zeros((0, len(pool)), bool)
    assert Gs.any(1).all(), "a single-hop question has no gold frame in the pool"
    gidx = group_index(pool_group, parallel)
    Qall = np.load(a.queries)
    Qs = Qall[[q["row"] for q in single]].astype(np.float32)
    Qp = Qall[[q["row"] for q in parallel]].astype(np.float32)
    runs = dict(named(s) for s in a.frames)
    ref = a.reference or next(iter(runs))
    ref = next((n for n, p in runs.items() if ref in (n, p)), ref)
    boot = {}                                               # per question set: paired + clustered bootstrap indices
    for key, qs in (("single", single), ("parallel", parallel)):
        if qs:
            boot[key] = bootstrap_index(len(qs), np.array([question_groups(q)[0][0] for q in qs]), a.n_boot, a.n_cluster_boot, a.seed)
    out = dict(n_single=len(single), n_parallel=len(parallel), n_pool=len(pool), reference=ref, runs={})
    hits = {}
    for name, path in runs.items():
        X = np.asarray(np.load(path, mmap_mode="r")[pool], dtype=np.float32)
        nrm = np.linalg.norm(X, axis=1)
        assert (nrm > 0).all(), f"{path}: {(nrm == 0).sum()} pool rows are empty"
        o, hits[name] = evaluate_run(X, Qs, Gs, Qp, gidx)
        out["runs"][name] = dict(file=path, **o)
    print(f"{len(single)} single-hop + {len(parallel)} parallel questions x {len(pool):,} pool frames | reference {ref} | "
          f"bootstrap {a.n_boot} (+{a.n_cluster_boot} clustered)")
    for name, o in out["runs"].items():
        for key, metrics, shown in (("single", [f"R{k}" for k in KS], ("R1", "R10")),
                                    ("parallel", [f"PR{k}" for k in KS] + [f"PA{k}" for k in KS], ("PR10", "PA10"))):
            if key not in o:
                continue
            if name != ref and ref in hits:
                for mk in metrics:
                    o[key][f"d{mk}"] = delta(hits[name][mk], hits[ref][mk], *boot[key])
            line = f"{name:24s} {key:8s}" + "".join(f"  {mk} {o[key][mk]:6.2f}" for mk in metrics if mk != "PA1")
            for mk in shown:
                if f"d{mk}" in o[key]:
                    d = o[key][f"d{mk}"]
                    line += f"  d{mk} {d['delta']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]"
            print(line, flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
