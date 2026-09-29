"""Tests for analysis/task4_ceiling.py (no GPU, no transformers).

  PYTHONPATH=. python tests/test_task4_ceiling.py
"""

from __future__ import annotations

import copy
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis import task4_ceiling as tc  # noqa: E402

MODELS = [f"m{i}" for i in range(7)]


def toy_gold(frozen_order, tuned_order, tasks=("a", "b", "c")):
    gold = {}
    for t in tasks:
        g = {}
        for m in MODELS:
            fr = 60.0 + 2.0 * frozen_order.index(m)
            tu = 70.0 + 2.0 * tuned_order.index(m)
            g[m] = {"logme": fr / 100, "frozen": (fr, 0.1), "tuned": (tu, 0.1)}
        gold[(t, "cls")] = g
    return gold


def test_oracle_is_perfect_when_frozen_and_tuned_agree():
    rows = tc.cells(toy_gold(MODELS, MODELS))
    tuned = [r for r in rows if r["estimator"] == "frozen_oracle" and r["regime"] == "tuned"]
    assert tuned
    assert all(abs(r["rho"] - 1) < 1e-12 and r["top1"] and r["regret"] == 0 for r in tuned)


def test_oracle_is_reversed_when_tuned_reverses_frozen():
    rows = tc.cells(toy_gold(MODELS, MODELS[::-1]))
    tuned = [r for r in rows if r["estimator"] == "frozen_oracle" and r["regime"] == "tuned"]
    assert all(abs(r["rho"] + 1) < 1e-12 and not r["top1"] for r in tuned)


def test_oracle_is_not_judged_against_frozen_gold():
    rows = tc.cells(toy_gold(MODELS, MODELS))
    assert not [r for r in rows if r["estimator"] == "frozen_oracle" and r["regime"] == "frozen"]


def test_loto_never_reads_the_held_out_task():
    gold = toy_gold(MODELS, MODELS)
    before = tc.loto_scores(gold, "a", "cls")
    changed = copy.deepcopy(gold)
    for i, m in enumerate(MODELS[::-1]):
        changed[("a", "cls")][m]["tuned"] = (99.0 - i, 0.1)
    assert tc.loto_scores(changed, "a", "cls") == before


def test_loto_follows_the_other_tasks():
    gold = toy_gold(MODELS, MODELS)
    s = tc.loto_scores(gold, "a", "cls")
    assert max(s, key=s.get) == MODELS[-1]


def test_loto_aborts_without_other_tasks():
    gold = toy_gold(MODELS, MODELS, tasks=("a",))
    try:
        tc.loto_scores(gold, "a", "cls")
    except SystemExit as exc:
        assert "no other task" in str(exc)
    else:
        raise AssertionError("single-task gold did not abort")


def test_published_gold_counts():
    rows = tc.cells(tc.load_gold())
    run = [r for r in rows if r["task"] in tc.RUN_TASKS and r["regime"] == "tuned"]
    assert len(run) == 3 * 8, len(run)


def main() -> int:
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except (Exception, SystemExit):
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
