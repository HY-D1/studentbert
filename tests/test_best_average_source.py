"""Tests for analysis/best_average_source.py (no GPU).

  PYTHONPATH=. python tests/test_best_average_source.py
"""

from __future__ import annotations

import copy
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis import best_average_source as ba  # noqa: E402

DS = ["a", "b", "c", "d"]
SEEDS = (1, 2, 42)


def toy_cells(order_by_target: dict) -> dict:
    """order_by_target[t] lists sources best first; values fall by 0.01 per rank."""
    cells = {}
    for t, order in order_by_target.items():
        vals = {f"src:{s}": {sd: 0.80 - 0.01 * i + 0.0001 * sd for sd in SEEDS}
                for i, s in enumerate(order)}
        vals["scratch"] = {sd: 0.70 for sd in SEEDS}
        cells[("B", t, "n3000")] = vals
    return cells


def base_orders() -> dict:
    return {t: ["b", "a", "c", "d"] for t in DS}


def test_held_out_target_never_enters_its_own_scores():
    cells = toy_cells(base_orders())
    cands = [f"src:{s}" for s in DS]
    before = ba.best_average_scores(cells, "a", cands)
    changed = copy.deepcopy(cells)
    for i, s in enumerate(["d", "c", "a", "b"]):
        changed[("B", "a", "n3000")][f"src:{s}"] = {sd: 0.90 - 0.01 * i for sd in SEEDS}
    assert ba.best_average_scores(changed, "a", cands) == before


def test_own_dataset_rank_is_not_used():
    orders = {"a": ["d", "b", "c", "a"], "b": ["d", "a", "c", "b"],
              "c": ["c", "a", "b", "d"], "d": ["d", "a", "b", "c"]}
    sc, n = ba.best_average_scores(toy_cells(orders), "a", [f"src:{s}" for s in DS])
    assert n["src:c"] == 2 and n["src:a"] == 3
    # src:c ranks 3rd on b and 4th on d; its 1st place on its own dataset c is left out
    assert sc["src:c"] == -3.5


def test_picks_the_best_foreign_average():
    sc, _n = ba.best_average_scores(toy_cells(base_orders()), "c", [f"src:{s}" for s in DS])
    assert max(sc, key=sc.get) == "src:b"


def test_aborts_without_a_foreign_rank():
    cells = toy_cells({"a": ["b", "a"], "b": ["b", "a"]})
    try:
        ba.best_average_scores(cells, "a", ["src:b"])
    except SystemExit as exc:
        assert "no foreign rank" in str(exc)
    else:
        raise AssertionError("a source seen only in-domain did not abort")


def test_scratch_is_never_a_candidate_rank():
    sc, _n = ba.best_average_scores(toy_cells(base_orders()), "a", [f"src:{s}" for s in DS])
    assert "scratch" not in sc


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
