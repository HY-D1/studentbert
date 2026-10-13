from __future__ import annotations

"""Tests for analysis/nlp_head_pairs.py and analysis/fixed_rule_eval.py (synthetic inputs).

  PYTHONPATH=. python tests/test_option1_steps.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import fixed_rule_eval as fr  # noqa: E402
from analysis import nlp_head_pairs as nh  # noqa: E402
from tests.test_reversal_model import rows  # noqa: E402

MODELS = [f"m{i}" for i in range(7)]


def toy(perf_means: list[float], sd: float = 0.1):
    gold, scores = {}, {}
    for t in nh.TASKS:
        for p in nh.POOLINGS:
            gold[(t, p)] = {m: {"logme": 0.0, "frozen": (50.0, sd), "tuned": (v, sd)}
                            for m, v in zip(MODELS, perf_means)}
            for est in ("hscore", "logme"):
                scores[(est, t, p)] = {s: {m: float(-i) for i, m in enumerate(MODELS)}
                                       for s in (1, 2, 42)}
    return scores, gold


def test_labels_follow_the_task4_tie_rule():
    scores, gold = toy([80.0, 90.0, 80.05, 70.0, 80.0, 79.0, 81.0])
    pairs, recs = nh.build(scores, gold)
    one = {p["alternative"]: p["label"] for p in pairs
           if (p["task"], p["pooling"], p["seed"]) == ("agnews", "cls", 1)}
    assert all(p["pick"] == "m0" for p in pairs)
    assert one == {"m1": "reversal", "m2": "near_tie", "m3": "stable", "m4": "near_tie",
                   "m5": "stable", "m6": "reversal"}
    assert len(pairs) == 30 * 6 and all(r["reversed"] == 1 for r in recs)


def test_a_pick_tied_with_the_best_is_not_reversed():
    scores, gold = toy([90.0, 90.05, 80.0, 80.0, 80.0, 80.0, 80.0])
    _, recs = nh.build(scores, gold)
    assert all(r["reversed"] == 0 and r["tied_with_best"] == 1 for r in recs)


def test_the_rule_abstains_exactly_when_the_pick_is_not_the_epoch3_leader():
    rs = rows(n_targets=2)
    for r in rs:
        r["rank_e3_a"] = "1" if r["target"] == "t0" else "2"
    risk = fr.rule_risk(rs)
    assert all(v == (0.0 if k[0] == "t0" else 1.0) for k, v in risk.items())
    assert len(risk) == 2 * 7


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
