from __future__ import annotations

"""Tests for analysis/warning_metrics.py. Stdlib only; run from the code root:

  PYTHONPATH=. python tests/test_warning_metrics.py
"""

import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from analysis import warning_metrics as wm

MODELS = [f"org/m{i}" for i in range(7)]


def test_auroc_and_ap() -> None:
    assert abs(wm.auroc([0.9, 0.7], [0.8, 0.6]) - 0.75) < 1e-12
    assert wm.auroc([1.0, 1.0], [1.0, 1.0]) == 0.5
    assert abs(wm.average_precision([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0]) - (0.5 + 1 / 3)) < 1e-12
    assert abs(wm.average_precision([1.0] * 4, [1, 0, 0, 0]) - 0.25) < 1e-12


def fixture(root: Path, seeds: tuple[int, ...]) -> str:
    """Five tasks x two poolings x the seeds; m0 is H-score's pick. In task t0 the pick is
    overtaken and m1 leads after epoch 1; elsewhere the pick leads and is not overtaken."""
    pairs, recs = [], []
    for t in [f"t{i}" for i in range(5)]:
        for pool in ("cls", "mean"):
            for s in seeds:
                over = t == "t0"
                recs.append({"task": t, "pooling": pool, "seed": s, "pick": MODELS[0],
                             "best": MODELS[1] if over else MODELS[0], "tied_with_best": 0,
                             "regret_points": 1.0 if over else 0.0, "reversed": int(over),
                             "n_overtaking": int(over)})
                for i, m in enumerate(MODELS[1:], 1):
                    pairs.append({"task": t, "pooling": pool, "seed": s, "pick": MODELS[0],
                                  "alternative": m, "hscore_pick": 10.0, "hscore_alt": 10.0 - i})
                for i, m in enumerate(MODELS):
                    acc = 0.80 - 0.01 * i
                    if over and i == 1:
                        acc = 0.85
                    run = {"epochs": [{"epoch": k, "val_acc": acc + 0.001 * k} for k in (1, 2, 3)]}
                    f = root / "ft" / t / f"{m.replace('/', '__')}_{pool}_s{s}.json"
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_text(json.dumps(run))
    prefix = str(root / "fx_head")
    md5s = []
    for name, rows in (("pairs", pairs), ("recs", recs)):
        p = Path(f"{prefix}_{name}.tsv")
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        md5s.append(f"{p.name} {hashlib.md5(p.read_bytes()).hexdigest()}")
    Path(prefix + "_report.md").write_text("files: " + ", ".join(md5s) + "\n")
    return prefix


def test_nlp_fixture() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        prefix = fixture(root, (9, 10, 11))
        recs = wm.nlp(prefix, str(root / "ft"), (9, 10, 11))
        assert len(recs) == 30 and sum(r["y"] for r in recs) == 6
        over = [r for r in recs if r["y"]]
        assert all(abs(r["s1"] - 0.05) < 1e-9 for r in over)
        assert all(abs(r["s1"] + 0.01) < 1e-9 for r in recs if not r["y"])
        lines, rows = wm.evaluate("9,10,11", recs)
        first = rows[0]
        assert first["precision"] == 1.0 and first["recall"] == 1.0 and first["auroc"] == 1.0
        assert abs(first["auprc"] - 1.0) < 1e-12 and abs(first["prevalence"] - 0.2) < 1e-12


def test_seed_and_count_checks_stop() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        prefix = fixture(root, (9, 10, 11))
        try:
            wm.nlp(prefix, str(root / "ft"), (42, 1, 2))
        except SystemExit as e:
            assert "expected" in str(e)
        else:
            raise AssertionError("wrong seeds were accepted")
        recs = wm.nlp(prefix, str(root / "ft"), (9, 10, 11))
        try:
            wm.evaluate("42,1,2", recs)
        except SystemExit as e:
            assert "do not reproduce RESULTS.md" in str(e)
        else:
            raise AssertionError("counts that differ from RESULTS.md were accepted")


def test_ties() -> None:
    assert wm.name_flag({"org/a": 0.8, "org/b": 0.8}, "org/b") is True
    assert wm.name_flag({"org/a": 0.8, "org/b": 0.8}, "org/a") is False
    recs = []
    for i in range(10):
        r = {"group": f"t{i % 5}", "id": f"r{i}", "y": int(i < 4), "frozen": float(i)}
        for k in (1, 2, 3):
            r[f"s{k}"] = 0.0 if i == 0 else (0.01 if i < 4 else -0.01)
            r[f"n{k}"] = i == 0 or i < 4
        recs.append(r)
    ties: list[str] = []
    lines, rows = wm.evaluate("9,10,11", recs, ties)
    assert rows[0]["recall"] == 0.75 and len(ties) == 3
    assert "flagged by the original rule only" in ties[0] and "overtaken" in ties[0]


def test_coverage() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        prefix = fixture(root, (9, 10, 11))
        recs = wm.nlp(prefix, str(root / "ft"), (9, 10, 11))
        lines, rows = wm.coverage("9,10,11", recs)
        assert rows[0]["followed"] == 24 and rows[0]["rule_regret"] == 0.0
        assert rows[0]["rule_overtaken"] == 0 and abs(rows[0]["always_regret"] - 0.2) < 1e-12


def main() -> None:
    tests = [test_auroc_and_ap, test_nlp_fixture, test_seed_and_count_checks_stop, test_ties,
             test_coverage]
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
