from __future__ import annotations

"""Tests for analysis/regret_compute.py: successive halving, the override margin, abstention and
the frontier on hand-built decisions. Run from the code root:

  PYTHONPATH=. python tests/test_regret_compute.py
"""

import sys

from analysis import regret_compute as rc


def decision(curves: dict, outcome: dict, hscore: dict, margin: float = 0.0) -> rc.Decision:
    costs = {c: [10.0 * (e + 1) for e in range(len(v))] for c, v in curves.items()}
    return rc.Decision("x", "t", "t/1", outcome, hscore, curves, costs, 5.0,
                       lambda d, a, b, k: margin)


def test_geometric_rungs() -> None:
    assert rc.geometric(1, 2, 8, 20) == [1, 2, 4]
    assert rc.geometric(5, 2, 8, 20) == [5, 10, 20]
    assert rc.geometric(3, 2, 24, 50) == [3, 6, 12, 24, 48]
    assert rc.geometric(1, 3, 8, 20) == [1, 3]


def test_halving_keeps_the_better_half_and_charges_epochs_run() -> None:
    curves = {"a": [0.9, 0.1, 0.1, 0.1], "b": [0.8, 0.9, 0.9, 0.9], "c": [0.1] * 4,
              "d": [0.2] * 4}
    d = decision(curves, {"a": 1.0, "b": 2.0, "c": 0.0, "d": 0.0}, {"a": 4, "b": 3, "c": 2, "d": 1})
    choice, cost = rc.halving(d, [1, 2], 2)
    assert choice == "b", choice
    # a and b trained to epoch 2 (cost 20 each), c and d stopped after epoch 1 (10 each)
    assert cost == 60.0, cost


def test_override_needs_more_than_the_margin() -> None:
    curves = {"a": [0.50], "b": [0.505]}
    out, hs = {"a": 1.0, "b": 2.0}, {"a": 2, "b": 1}
    assert rc.override(decision(curves, out, hs, margin=0.01), 1)[0] == "a"
    assert rc.override(decision(curves, out, hs, margin=0.001), 1)[0] == "b"
    assert rc.abstain(decision(curves, out, hs), 1)[0] is None


def test_frontier() -> None:
    rows = [{"method": "m", "compute_gpu_h": c, "regret": g, "chosen": 1, "decisions": 1}
            for c, g in ((0.0, 5.0), (1.0, 2.0), (2.0, 3.0), (3.0, 1.0), (1.0, 2.5))]
    assert rc.frontier(rows, False) == {0, 1, 3}


def main() -> None:
    tests = [test_geometric_rungs, test_halving_keeps_the_better_half_and_charges_epochs_run,
             test_override_needs_more_than_the_margin, test_frontier]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e!r}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
