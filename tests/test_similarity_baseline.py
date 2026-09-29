"""Tests for analysis/similarity_baseline.py on a synthetic executions table and score file.

  PYTHONPATH=. python tests/test_similarity_baseline.py
"""

from __future__ import annotations

import contextlib
import io
import json
import statistics as st
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import evaluate_estimators as ev  # noqa: E402
from analysis import similarity_baseline as sb  # noqa: E402

SEVEN = ("assist2017", "ednet", "junyi", "algebra2005", "bridge2006", "assist2009", "algebra2006")
COLS = ("run_id", "log_file", "track", "metric", "target_dataset", "budget", "candidate",
        "finetune_seed", "value", "valid_for_primary_analysis")


def gold_values(target):
    """Own encoder best overall, Junyi best foreign, EdNet second, the rest lower."""
    base = {"scratch": 0.700, f"src:{target}": 0.760, "src:junyi": 0.750, "src:ednet": 0.740}
    return {c: base.get(c, 0.705 + 0.001 * i)
            for i, c in enumerate(["scratch"] + [f"src:{d}" for d in SEVEN])}


def write_exec(path, targets=("algebra2006",), extra=()):
    lines = ["\t".join(COLS)]
    for tgt in targets:
        for c, v in gold_values(tgt).items():
            for s, jitter in ((42, 0.0), (1, 0.0004), (2, -0.0004)):
                lines.append("\t".join([f"r_{tgt}_{c}_{s}", "x.log", "B", "test_auc", tgt,
                                        "n3000", c, str(s), f"{v + jitter:.6f}", "yes"]))
    lines += list(extra)
    Path(path).write_text("\n".join(lines) + "\n")


def write_scores(path, target):
    recs = []
    order = {target: 0.9, "junyi": 0.8, "ednet": 0.7}
    for seed in (42, 1, 2):
        for d in SEVEN:
            recs.append({"estimator": "hs", "access_regime": "F",
                         "candidate": f"edubert_{d}_pretrain_full_encoder.pt", "target": target,
                         "seed": seed, "score": order.get(d, 0.1), "score_direction":
                         "higher_is_better", "n_target_examples": 1, "n_target_labels": 1,
                         "feature_extraction_time": 1.0, "scoring_time": 1.0,
                         "peak_memory_mb": 1.0, "uncertainty_signals": {},
                         "metadata": {"n_students_requested": 3000, "score_split": "train"}})
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in recs))


def rows_for(path, scores=None, targets=("algebra2006",)):
    write_exec(path, targets)
    cells, _ = ev.load_gold(path)
    lsc = ev.logme_scores([scores]) if scores else {}
    return sb.evaluate(cells, lsc, sb.profiles(), 0.001, 200)


def test_own_encoder_is_out_of_the_view():
    with tempfile.TemporaryDirectory() as t:
        rows = rows_for(f"{t}/e.tsv")
        assert rows and all(r["choice"] != "src:algebra2006" for r in rows), rows
        assert all(r["coverage"] == "6/6" and r["best_foreign"] == "src:junyi" for r in rows)


def test_self_distance_is_zero_and_closer_scores_higher():
    prof = {"a": {"x": 0.0}, "b": {"x": 1.0}, "c": {"x": 3.0}}
    sc = sb.similarity_scores("a", ["a", "b", "c"], ("x",), prof)
    assert sc["src:a"] == 0.0 and sc["src:b"] > sc["src:c"], sc


def test_standardize_gives_zero_mean_unit_sd():
    z = sb.standardize(sb.profiles(), ("log_students", "correct_rate"))
    for k in ("log_students", "correct_rate"):
        vals = [z[n][k] for n in z]
        assert abs(st.mean(vals)) < 1e-12 and abs(st.pstdev(vals) - 1) < 1e-12


def test_estimator_is_judged_in_the_same_foreign_view():
    with tempfile.TemporaryDirectory() as t:
        write_scores(f"{t}/s.jsonl", "algebra2006")
        rows = [r for r in rows_for(f"{t}/e.tsv", f"{t}/s.jsonl") if r["estimator"] == "hs"]
        assert len(rows) == 3 and all(r["choice"] == "src:junyi" and r["top1"] for r in rows)
        assert all(r["regret"] == 0.0 for r in rows), rows


def test_largest_source_is_ednet_or_junyi_on_ednet():
    with tempfile.TemporaryDirectory() as t:
        rows = rows_for(f"{t}/e.tsv", targets=("algebra2006", "ednet"))
        pick = {r["target"]: r["choice"] for r in rows if r["estimator"] == "largest source"}
        assert pick == {"algebra2006": "src:ednet", "ednet": "src:junyi"}, pick


def test_unknown_source_aborts():
    with tempfile.TemporaryDirectory() as t:
        extra = ["\t".join(["r_m", "x.log", "B", "test_auc", "algebra2006", "n3000",
                            "src:mystery", str(s), "0.7", "yes"]) for s in (42, 1, 2)]
        write_exec(f"{t}/e.tsv", extra=extra)
        cells, _ = ev.load_gold(f"{t}/e.tsv")
        try:
            sb.evaluate(cells, {}, sb.profiles(), 0.001, 200)
        except SystemExit as ex:
            assert "no dataset profile" in str(ex), ex
        else:
            raise AssertionError("unknown source did not abort")


def test_main_writes_three_files():
    with tempfile.TemporaryDirectory() as t:
        write_exec(f"{t}/e.tsv")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = sb.main(["--executions", f"{t}/e.tsv", "--boots", "200", "--out", f"{t}/o"])
        assert rc == 0
        for suffix in ("_cells.tsv", "_summary.tsv", "_report.md"):
            assert Path(f"{t}/o{suffix}").is_file(), suffix
        rep = Path(f"{t}/o_report.md").read_text()
        assert "| algebra2006 | src:junyi |" in rep, rep


def test_report_rounding_is_decimal_half_up():
    assert sb.fmt(0.00015) == "0.0002"
    assert sb.fmt(-0.00015) == "-0.0002"
    assert sb.fmt(0.1234) == "0.1234"
    assert sb.fmt(7) == "7"


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
