from __future__ import annotations

"""Tests for analysis/reversal_model.py on synthetic pair rows (no cluster files needed).

  PYTHONPATH=. python tests/test_reversal_model.py
"""

import copy
import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import reversal_model as rm  # noqa: E402

COLS = [c for g in rm.GROUPS.values() for c in g]
CANDS = [f"c{i}" for i in range(8)]


def rows(n_targets: int = 4, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for t in range(n_targets):
        for s in ("42",):
            for i, j in combinations(range(8), 2):
                x = rng.gauss(0, 1)
                lab = "reversal" if x + rng.gauss(0, 0.5) > 0.8 else (
                    "near_tie" if abs(x) < 0.2 else "stable")
                r = {c: str(rng.gauss(0, 1)) for c in COLS}
                r.update({"val_diff_e3": str(-x), "target": f"t{t}", "est_seed": s,
                          "cand_a": CANDS[i], "cand_b": CANDS[j], "frozen_rank_a": str(i + 1),
                          "frozen_rank_b": str(j + 1), "label_main": lab, "label_robust": lab,
                          "target_regret_practical": "0.001" if t == 0 else "0.0"})
                out.append(r)
    return out


def test_fit_recovers_a_separating_direction():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] + 0.2 * rng.normal(size=400) > 0).astype(float)
    m = rm.fit(X, y, 1.0)
    p = rm.predict(m, X)
    assert rm.auc(y, p) > 0.97 and m[0][0] > abs(m[0][1]) and m[0][0] > abs(m[0][2])


def test_held_out_labels_never_reach_their_own_predictions_or_penalty():
    base = rows()
    p1, l1 = rm.loto(base, COLS, "label_main")
    flipped = copy.deepcopy(base)
    for r in flipped:
        if r["target"] == "t1":
            r["label_main"] = {"stable": "reversal", "reversal": "stable"}.get(r["label_main"],
                                                                              r["label_main"])
    p2, l2 = rm.loto(flipped, COLS, "label_main")
    own = [k for k in p1 if k[0] == "t1"]
    assert all(p1[k] == p2[k] for k in own) and l1["t1"] == l2["t1"]
    assert any(p1[k] != p2[k] for k in p1 if k[0] == "t0")


def test_near_ties_are_predicted_but_never_fitted():
    base = rows()
    assert all(r["label_main"] in ("stable", "reversal") for r in rm.fit_rows(base, "label_main"))
    pred, _ = rm.loto(base, COLS, "label_main")
    assert len(pred) == len(base)


def test_recommendation_takes_the_top_pick_and_its_largest_pair_risk():
    base = rows(n_targets=1)
    pred = {rm.key(r): 0.1 for r in base}
    head = [r for r in base if r["frozen_rank_a"] == "1"]
    pred[rm.key(head[3])] = 0.7
    rec = rm.recommendations(base, pred, "label_main")[0]
    assert rec["selected"] == "c0" and abs(rec["reversal_probability"] - 0.7) < 1e-12
    assert rec["abstain"] == 1 and abs(rec["confidence"] - 0.3) < 1e-12


def test_a_near_tie_pick_is_not_counted_as_a_missed_reversal():
    recs = [{"abstain": 0, "regret": 0.0005, "reversed": 0, "ambiguous": 0,
             "reversal_probability": 0.1, "margin_to_runner_up_z": 1.0},
            {"abstain": 1, "regret": 0.003, "reversed": 1, "ambiguous": 0,
             "reversal_probability": 0.9, "margin_to_runner_up_z": 0.1}]
    s = rm.selective(recs)
    assert s["missed"] == 0 and s["caught"] == 1 and s["false_alarms"] == 0 and s["reversed"] == 1


def test_ambiguous_recommendations_leave_the_selective_counts():
    recs = [{"abstain": 1, "regret": 0.002, "reversed": 0, "ambiguous": 1,
             "reversal_probability": 0.9, "margin_to_runner_up_z": 0.1},
            {"abstain": 0, "regret": 0.0, "reversed": 0, "ambiguous": 0,
             "reversal_probability": 0.1, "margin_to_runner_up_z": 2.0}]
    s = rm.selective(recs)
    assert s["n"] == 1 and s["false_alarms"] == 0


def test_aurc_orders_by_risk():
    recs = [{"regret": 0.0, "r": 0.1}, {"regret": 0.01, "r": 0.9}]
    good = rm.aurc(recs, lambda r: r["r"])
    bad = rm.aurc(recs, lambda r: -r["r"])
    assert good < bad


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001 (report every failure, then exit non-zero)
            failed += 1
            print(f"FAIL  {name}: {exc!r}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
