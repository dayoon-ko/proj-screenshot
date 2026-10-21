"""Rank rules of screenret.evaluate on a toy pool (no model)."""
import json

import numpy as np

from screenret import evaluate as E

# 6 pool frames of recordings a, a, b, c, c, d; one query whose score of frame i is X[i, 0]
POOL_GROUP = np.array(["a", "a", "b", "c", "c", "d"])
X = np.array([[0.9], [0.1], [0.5], [0.7], [0.2], [0.8]], np.float32)
Q = np.array([[1.0]], np.float32)


def test_single_hop_rank_ignores_other_gold_frames():
    G = E.gold_matrix(POOL_GROUP, [dict(qid="s", groups=[["b", "c"]])])
    assert E.ranks(Q, X, G).tolist() == [3]           # best gold = c (0.7); non-gold above it: a 0.9, d 0.8


def test_parallel_group_rank_counts_every_pool_frame():
    q = dict(qid="p", type="parallel", groups=[["a"], ["c"], ["b"]])
    grk = E.group_ranks(Q, X, E.group_index(POOL_GROUP, [q]))
    assert grk == [[1, 3, 4]]                          # c: a, d above; b: a, d and gold c above (all frames compete)
    h = E.parallel_hits(grk)
    assert np.isclose(h["PR1"][0], 1 / 3) and h["PA1"][0] == 0
    assert np.isclose(h["PR5"][0], 1.0) and h["PA5"][0] == 1


def test_main_end_to_end(tmp_path):
    proto = dict(pool=list(range(6)), pool_group=POOL_GROUP.tolist(),
                 questions=[dict(qid="s", type="single", row=0, groups=[["c"]]),
                            dict(qid="old", row=0, gold=["d"]),                         # old protocol entry
                            dict(qid="p", type="parallel", row=0, groups=[["a"], ["c"], ["b"]])])
    (tmp_path / "p.json").write_text(json.dumps(proto))
    np.save(tmp_path / "q.npy", Q)
    np.save(tmp_path / "x.npy", X)
    np.save(tmp_path / "y.npy", X[::-1].copy())
    out = tmp_path / "r.json"
    E.main(["--protocol", str(tmp_path / "p.json"), "--queries", str(tmp_path / "q.npy"), "--frames", f"A={tmp_path / 'x.npy'}",
            f"B={tmp_path / 'y.npy'}", "--n-boot", "50", "--n-cluster-boot", "10", "--out", str(out)])
    r = json.load(open(out))
    assert r["n_single"] == 2 and r["n_parallel"] == 1
    a = r["runs"]["A"]
    assert a["single"]["ranks"] == [3, 2] and a["single"]["R1"] == 0 and a["single"]["R5"] == 100
    assert a["parallel"]["group_ranks"] == [[1, 3, 4]] and np.isclose(a["parallel"]["PR1"], 100 / 3)
    assert "dR1" in r["runs"]["B"]["single"] and "dPR10" in r["runs"]["B"]["parallel"]
