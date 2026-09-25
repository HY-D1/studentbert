"""Tests for analysis/nextskill_31.py (RESULTS.md 3.1 render-and-compare).

  PYTHONPATH=. python tests/test_nextskill_31.py
Synthetic per-seed values are rendered into a fixture RESULTS.md; the check must PASS on the
faithful render and FAIL on each injected defect. One test parses the real RESULTS.md 3.1.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import random
import sys
import tempfile
import traceback
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ns31", REPO / "analysis" / "nextskill_31.py")
ns = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ns)

HEADER = ("### 3.1 Next-skill N-sweep, assist2017 target (parsed from logs)\n\n"
          "Source: test fixture. 144 raw run records dedup to 72 unique (cond, N, seed). "
          "macro_top1 not logged in this sweep: NOT FOUND.\n\n")


def _values(seed=7):
    rng = random.Random(seed)
    rows = []
    for c in ns.CONDS:
        for n in ns.NS:
            for m, _ in ns.METRICS:
                base = 0.6 + 0.3 * rng.random()
                for s in ns.SEEDS:
                    rows.append((c, n, s, m, f"{base + rng.uniform(-0.02, 0.02):.6f}"))
    return rows


def _write_csv(path, rows, extra=()):
    lines = ["dataset,condition,N,seed,metric,value"]
    lines += [f"assist2017,{c},{n},{s},{m},{v}" for c, n, s, m, v in rows]
    lines += list(extra)
    Path(path).write_text("\n".join(lines) + "\n")


def _write_md(path, rows, sd="pop", edit=None):
    vals = {}
    for c, n, s, m, v in rows:
        vals.setdefault((c, n, m), {})[s] = Decimal(v)
    body = ns.render(vals, sd)
    std = ns.q4(ns.stats(vals[("ednet", 25, "top1")])[sd])
    note = f"\nNote: ednet at N=25 is high variance (top-1 std {std}).\n\n---\n### 3.2 next\n"
    text = HEADER + body + note
    if edit:
        old, new = edit
        assert text.count(old) >= 1, old
        text = text.replace(old, new, 1)
    Path(path).write_text(text)
    return vals


def _run(md, csv_path, parser_stdout=None):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = ns.check(md, csv_path, parser_stdout)
    return rc, buf.getvalue()


def test_faithful_render_passes():
    with tempfile.TemporaryDirectory() as t:
        rows = _values()
        _write_csv(f"{t}/l.csv", rows)
        _write_md(f"{t}/R.md", rows)
        rc, out = _run(f"{t}/R.md", f"{t}/l.csv")
        assert rc == 0 and "population matches 96/96" in out and "36 gap cells" in out, out


def test_sample_sd_detected():
    with tempfile.TemporaryDirectory() as t:
        rows = _values()
        _write_csv(f"{t}/l.csv", rows)
        _write_md(f"{t}/R.md", rows, sd="sample")
        rc, out = _run(f"{t}/R.md", f"{t}/l.csv")
        assert rc == 0 and "sample matches 96/96" in out, out


def test_changed_mean_fails():
    with tempfile.TemporaryDirectory() as t:
        rows = _values()
        _write_csv(f"{t}/l.csv", rows)
        vals = _write_md(f"{t}/R.md", rows)
        s = ns.stats(vals[("junyi", 200, "top5")])
        cell = f"{ns.q4(s['mean'])} ±{ns.q4(s['pop'])}"
        bad = f"{ns.q4(s['mean']) + Decimal('0.0003')} ±{ns.q4(s['pop'])}"
        _write_md(f"{t}/R.md", rows, edit=(cell, bad))
        rc, out = _run(f"{t}/R.md", f"{t}/l.csv")
        assert rc == 1 and "junyi N=200 top5" in out, out


def test_changed_gap_fails():
    with tempfile.TemporaryDirectory() as t:
        rows = _values()
        _write_csv(f"{t}/l.csv", rows)
        _write_md(f"{t}/R.md", rows)
        text = Path(f"{t}/R.md").read_text()
        line = next(ln for ln in text.splitlines() if ln.startswith("| ednet, macro AUC"))
        first = line.split("|")[2].strip()
        bumped = ns.q4(Decimal(first) + Decimal("0.0010"))
        new = line.replace(f"| {first} |", f"| {'+' if bumped >= 0 else ''}{bumped} |", 1)
        Path(f"{t}/R.md").write_text(text.replace(line, new))
        rc, out = _run(f"{t}/R.md", f"{t}/l.csv")
        assert rc == 1 and "gap ednet N=25 macro_auc" in out, out


def test_missing_seed_fails():
    with tempfile.TemporaryDirectory() as t:
        rows = _values()
        _write_md(f"{t}/R.md", rows)
        drop = next(r for r in rows if r[:4] == ("scratch", 500, 42, "top1"))
        _write_csv(f"{t}/l.csv", [r for r in rows if r is not drop])
        rc, out = _run(f"{t}/R.md", f"{t}/l.csv")
        assert rc == 1 and "scratch N=500 top1: n=2" in out, out


def test_changed_sd_fails():
    with tempfile.TemporaryDirectory() as t:
        rows = _values()
        _write_csv(f"{t}/l.csv", rows)
        vals = _write_md(f"{t}/R.md", rows)
        s = ns.stats(vals[("scratch", 50, "weighted_auc")])
        old = f"{ns.q4(s['mean'])} ±{ns.q4(s['pop'])}"
        bad = f"{ns.q4(s['mean'])} ±{ns.q4(s['pop']) + Decimal('0.0100')}"
        _write_md(f"{t}/R.md", rows, edit=(old, bad))
        rc, out = _run(f"{t}/R.md", f"{t}/l.csv")
        assert rc == 1 and "no single convention fits" in out, out


def test_off_by_one_unit_non_boundary_fails():
    with tempfile.TemporaryDirectory() as t:
        rows = _values()
        _write_csv(f"{t}/l.csv", rows)
        vals = _write_md(f"{t}/R.md", rows)
        s = ns.stats(vals[("ednet", 100, "macro_auc")])
        assert not ns.boundary(s["mean"])
        old = f"{ns.q4(s['mean'])} ±{ns.q4(s['pop'])}"
        bad = f"{ns.q4(s['mean']) + Decimal('0.0001')} ±{ns.q4(s['pop'])}"
        _write_md(f"{t}/R.md", rows, edit=(old, bad))
        rc, out = _run(f"{t}/R.md", f"{t}/l.csv")
        assert rc == 1 and "ednet N=100 macro_auc" in out, out


def test_missing_whole_seed_breaks_coverage():
    with tempfile.TemporaryDirectory() as t:
        rows = _values()
        _write_md(f"{t}/R.md", rows)
        _write_csv(f"{t}/l.csv", [r for r in rows if r[:3] != ("junyi", 25, 1)])
        rc, out = _run(f"{t}/R.md", f"{t}/l.csv")
        assert rc == 1 and "71 unique (cond, N, seed), 3.1 says 72" in out, out


def test_duplicate_row_aborts():
    with tempfile.TemporaryDirectory() as t:
        rows = _values()
        _write_csv(f"{t}/l.csv", rows + rows[:1])
        _write_md(f"{t}/R.md", rows)
        try:
            _run(f"{t}/R.md", f"{t}/l.csv")
        except SystemExit as ex:
            assert "duplicate row" in str(ex), ex
        else:
            raise AssertionError("duplicate did not abort")


def test_boundary_tie_accepted_either_way():
    with tempfile.TemporaryDirectory() as t:
        rows = [r if r[:2] != ("indomain", 1000) or r[3] != "top1" else r[:4] + ("0.700050",)
                for r in _values()]
        _write_csv(f"{t}/l.csv", rows)
        _write_md(f"{t}/R.md", rows, edit=("0.7001 ±0.0000", "0.7000 ±0.0000"))
        rc, out = _run(f"{t}/R.md", f"{t}/l.csv")
        assert "boundary: indomain N=1000 top1" in out, out


def test_macro_top1_rows_contradict_note():
    with tempfile.TemporaryDirectory() as t:
        rows = _values()
        _write_csv(f"{t}/l.csv", rows, extra=["assist2017,scratch,25,42,macro_top1,0.340000"])
        _write_md(f"{t}/R.md", rows)
        rc, out = _run(f"{t}/R.md", f"{t}/l.csv")
        assert rc == 1 and "macro_top1 cells" in out, out


def test_parser_stdout_claim():
    with tempfile.TemporaryDirectory() as t:
        rows = _values()
        _write_csv(f"{t}/l.csv", rows)
        _write_md(f"{t}/R.md", rows)
        Path(f"{t}/ok.txt").write_text("raw run records: 144   unique (ds,cond,N,seed): 72\n")
        Path(f"{t}/bad.txt").write_text("raw run records: 150   unique (ds,cond,N,seed): 72\n")
        assert _run(f"{t}/R.md", f"{t}/l.csv", f"{t}/ok.txt")[0] == 0
        rc, out = _run(f"{t}/R.md", f"{t}/l.csv", f"{t}/bad.txt")
        assert rc == 1 and "150 raw" in out, out


def test_real_results_md_parses_every_cell():
    sec = ns.section_31((REPO / "RESULTS.md").read_text(encoding="utf-8"))
    shown = ns.parse_31(sec)
    means = [k for k in shown if k[0] == "mean"]
    gaps = [k for k in shown if k[0] == "gap"]
    assert len(means) == 96 and len(gaps) == 36, (len(means), len(gaps))
    assert all(isinstance(shown[k][0], Decimal) and shown[k][1] is not None for k in means)
    assert all(isinstance(shown[k][0], Decimal) for k in gaps)
    assert shown[("mean", "ednet", 25, "top1")] == (Decimal("0.6917"), Decimal("0.0455"))
    assert shown[("gap", "indomain", 25, "top1")][0] == Decimal("0.0922")


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
