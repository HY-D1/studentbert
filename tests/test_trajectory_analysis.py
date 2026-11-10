from __future__ import annotations

"""Tests for analysis/trajectory_analysis.py. Run from the code root:

  PYTHONPATH=. python tests/test_trajectory_analysis.py
"""

import sys

from analysis import trajectory_analysis as ta


def test_settle() -> None:
    assert ta.settle([True, False, True, True], True) == 3
    assert ta.settle([True, True], True) == 1
    assert ta.settle([True, True, False], True) is None
    assert ta.settle([False, False], False) == 1


def test_quicktune_detection() -> None:
    traj = {}
    for v in ("micro", "mini", "extended"):
        cands = {}
        for m, (start, slope) in {"a": (60.0, 0.0), "b": (10.0, 2.0)}.items():
            cands[m] = {e: (start + slope * e, 5.0 * e) for e in range(1, 51)}
        traj[(f"quicktune_{v}", "t", "")] = cands
    lines, rows = ta.quicktune(traj)
    text = "\n".join(lines)
    assert "settles by epoch 1 on 0" in text and "by 20 on 0" in text and "by 10 on 0" in text
    micro = [r for r in rows if r["benchmark"] == "quicktune_micro"]
    assert micro[0]["leader_regret"] == 110.0 - 60.0 and micro[-1]["leader_regret"] == 0.0


def main() -> None:
    tests = [test_settle, test_quicktune_detection]
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
