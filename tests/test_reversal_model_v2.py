from __future__ import annotations

"""Tests for analysis/reversal_model_v2.py on synthetic pair rows (no cluster files needed).

  PYTHONPATH=. python tests/test_reversal_model_v2.py
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import reversal_model_v2 as v2  # noqa: E402
from tests.test_reversal_model import COLS, rows  # noqa: E402


def test_only_pairs_headed_by_the_pick_are_rows():
    rs = rows()
    head = [r for r in rs if v2.is_head(r)]
    assert len(head) == 4 * 7 and all(r["cand_a"] == "c0" for r in head)


def test_noisy_or_and_max():
    assert abs(v2.noisy_or([0.5, 0.5]) - 0.75) < 1e-12 and v2.noisy_or([]) == 0.0
    assert abs(v2.noisy_or([0.1, 0.2, 0.3]) - (1 - 0.9 * 0.8 * 0.7)) < 1e-12


def test_youden_threshold_and_ties_go_high():
    recs = [{"risk": x, "reversed": y, "ambiguous": 0}
            for x, y in ((0.1, 0), (0.2, 0), (0.6, 1), (0.9, 1))]
    tau = v2.youden(recs, "risk")
    assert 0.2 <= tau < 0.6
    flat = [{"risk": 0.3, "reversed": y, "ambiguous": 0} for y in (0, 1)]
    assert v2.youden(flat, "risk") == 1.0


def test_held_out_labels_never_reach_its_predictions_or_threshold():
    base = rows()
    cols = [c for c in COLS if c in v2.FEATURES[v2.PRIMARY]]
    p1, f1 = v2.loto(base, cols, "label_main", head_only=True)
    flipped = copy.deepcopy(base)
    for r in flipped:
        if r["target"] == "t2":
            r["label_main"] = {"stable": "reversal", "reversal": "stable"}.get(r["label_main"],
                                                                              r["label_main"])
    p2, f2 = v2.loto(flipped, cols, "label_main", head_only=True)
    assert all(p1[k] == p2[k] for k in p1 if k[0] == "t2") and f1["t2"] == f2["t2"]


def test_recommendation_counts_skip_near_ties_and_ambiguous():
    recs = [{"target": "t0", "risk_noisy_or": 0.9, "reversed": 1, "ambiguous": 0, "regret": 0.003},
            {"target": "t1", "risk_noisy_or": 0.9, "reversed": 0, "ambiguous": 0, "regret": 0.0005},
            {"target": "t2", "risk_noisy_or": 0.1, "reversed": 0, "ambiguous": 1, "regret": 0.002},
            {"target": "t3", "risk_noisy_or": 0.1, "reversed": 0, "ambiguous": 0, "regret": 0.0}]
    fold = {t: {"tau_noisy_or": 0.5} for t in ("t0", "t1", "t2", "t3")}
    e = v2.evaluate(recs, "risk_noisy_or", fold)
    s = e["fold-chosen"]
    assert e["n"] == 3 and s["followed"] == 1 and s["caught"] == 1 and s["false_alarms"] == 1


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
