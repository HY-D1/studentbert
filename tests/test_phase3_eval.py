from __future__ import annotations

"""Tests for analysis/phase3_eval.py (synthetic inputs; no cluster files needed).

  PYTHONPATH=. python tests/test_phase3_eval.py
"""

import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import phase3_eval as p3  # noqa: E402
from analysis import task4_evaluate as te  # noqa: E402

GOLD = te.load_gold()
MODELS = sorted(GOLD[("rte", "cls")])


def test_the_criterion_needs_the_whole_interval_below_zero():
    assert p3.holds((-1.0, -2.0, -0.01))
    assert not p3.holds((-1.0, -2.0, 0.0)) and not p3.holds((-1.0, -2.0, 0.5))


def test_the_difference_is_zero_when_a_method_always_keeps_h_score():
    rows = [{"task": t, "pooling": "cls", "seed": 6, "hscore": MODELS[0], "full_3": MODELS[0]}
            for t in ("agnews", "mnli", "qnli", "rte", "airline")]
    assert p3.diff_ci(rows, GOLD, "full_3") == (0.0, 0.0, 0.0)


def test_head_files_must_match_their_own_report():
    with tempfile.TemporaryDirectory() as d:
        for name in ("pairs", "recs"):
            (Path(d) / f"h_{name}.tsv").write_text(f"{name}\n")
        md5 = {n: hashlib.md5((Path(d) / f"h_{n}.tsv").read_bytes()).hexdigest()
               for n in ("pairs", "recs")}
        (Path(d) / "h_report.md").write_text(f"Files: h_pairs.tsv {md5['pairs']}, h_recs.tsv "
                                             f"{'0' * 32}\n")
        try:
            p3.load_head(str(Path(d) / "h"))
        except SystemExit as exc:
            assert "does not match" in str(exc)
        else:
            raise AssertionError("a head file that disagrees with its report was accepted")


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
