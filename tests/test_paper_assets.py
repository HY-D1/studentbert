"""Smoke tests for analysis/make_paperA_assets.py and analysis/make_paperB_assets.py (no GPU).

Both scripts read the committed RESULTS.md and stop when a pinned section changes. These tests
build every Paper B table from RESULTS.md, parse Paper A's four sections, and check that a
one-digit change inside a pinned section stops each script. Paper A's summary TSV lives on the
cluster only, so its tables are not built here.

  PYTHONPATH=. python tests/test_paper_assets.py
"""

from __future__ import annotations

import re
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis import make_paperA_assets as pa  # noqa: E402
from analysis import make_paperB_assets as pb  # noqa: E402

RESULTS = (REPO / "RESULTS.md").read_text(encoding="utf-8")
PAPER_B_TABLES = {"datasets.tex", "sources.tex", "estimators.tex", "frozen.tex", "nlp.tex",
                  "warning.tex", "detector.tex", "correction.tex", "recipes.tex"}


def bump_digit(text: str, head: str) -> str:
    """Change the first digit after `head`, so only that section's text differs."""
    i = text.index(head) + len(head)
    m = re.search(r"\d", text[i:])
    j = i + m.start()
    return text[:j] + str((int(text[j]) + 1) % 10) + text[j + 1:]


def expect_abort(fn, *args) -> str:
    try:
        fn(*args)
    except SystemExit as e:
        return str(e)
    raise AssertionError("expected an ABORT, got none")


def test_paper_b_builds_every_table():
    with tempfile.TemporaryDirectory() as tmp:
        pb.main(["--results", str(REPO / "RESULTS.md"),
                 "--recipes", str(REPO / "configs/paperB_encoder_recipes.log"), "--out", tmp])
        made = {p.name for p in (Path(tmp) / "tables").iterdir()}
    assert made == PAPER_B_TABLES, sorted(made ^ PAPER_B_TABLES)


def test_paper_b_every_pinned_section_matches():
    for key in pb.SECTIONS:
        pb.section(RESULTS, key)


def test_paper_b_stops_on_a_changed_section():
    msg = expect_abort(pb.section, bump_digit(RESULTS, "### 12.21 PARC"), "12.21")
    assert "12.21 changed" in msg


def test_paper_a_parses_every_section():
    r = pa.parse(RESULTS)
    assert set(r["datasets"]) == set(pa.TRAIN)
    assert set(r["kt"]) == {"assist2017", "ednet", "junyi"}
    assert set(r["loss10"]) == set(pa.LADDER)


def test_paper_a_stops_on_a_changed_section():
    msg = expect_abort(pa.parse, bump_digit(RESULTS, "### 2.1"))
    assert "2.1 changed" in msg


def test_paper_a_ignores_sections_it_does_not_read():
    pa.parse(bump_digit(RESULTS, "### 12.21 PARC"))


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
