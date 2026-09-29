"""Tests for analysis/task4_gold_check.py: a correct table passes, one wrong digit fails.

  PYTHONPATH=. python tests/test_task4_gold_check.py
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import task4_gold_check as gc  # noqa: E402

HEAD = "dataset\tmodel\tpooling\tlogme\tfrozen_mean\tfrozen_sd\ttuned_mean\ttuned_sd\n"


def write(tmp, rows, reported):
    t, r = Path(tmp) / "t.tsv", Path(tmp) / "r.tsv"
    t.write_text("# comment line\n" + HEAD + "".join(
        f"d\tm{i}\tmean\t{lo}\t{fr}\t0.1\t{tu}\t0.1\n" for i, (lo, fr, tu) in enumerate(rows)))
    r.write_text("dataset\tpooling\tregime\tpearson\ttau_w\n" + "".join(
        f"d\tmean\t{reg}\t{p:.3f}\t{w:.3f}\n" for reg, p, w in reported))
    return t, r


ROWS = [(0.1, 90.0, 91.0), (0.2, 91.0, 90.5), (0.3, 92.5, 92.0), (0.4, 92.0, 93.0),
        (0.5, 94.0, 92.5), (0.6, 95.0, 94.0), (0.7, 95.5, 96.0)]


def reported_for(rows):
    lo = [r[0] for r in rows]
    out = []
    for reg, k in (("frozen", 1), ("tuned", 2)):
        y = [r[k] for r in rows]
        out.append((reg, gc.pearson(lo, y), gc.weighted_tau(lo, y) or 0.0))
    return out


def test_pearson_matches_a_hand_value():
    assert abs(gc.pearson([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-12
    assert abs(gc.pearson([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-12


def test_correct_table_passes():
    with tempfile.TemporaryDirectory() as tmp:
        t, r = write(tmp, ROWS, reported_for(ROWS))
        bad, lines = gc.check(t, r, 0.0015, 0.0015)
        assert bad == 0 and len(lines) == 2, lines


def test_one_wrong_mean_fails():
    with tempfile.TemporaryDirectory() as tmp:
        wrong = list(ROWS)
        wrong[3] = (0.4, 88.0, 93.0)
        t, r = write(tmp, wrong, reported_for(ROWS))
        bad, lines = gc.check(t, r, 0.0015, 0.0015)
        assert bad == 1 and "MISMATCH" in lines[0], lines


def test_missing_row_aborts():
    with tempfile.TemporaryDirectory() as tmp:
        t, r = write(tmp, ROWS[:6], reported_for(ROWS))
        try:
            gc.check(t, r, 0.0015, 0.0015)
        except SystemExit as ex:
            assert "expected 7" in str(ex)
        else:
            raise AssertionError("6 rows did not abort")


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
