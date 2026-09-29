"""Tests for analysis/finetune_trajectories.py (failure review D8; no GPU, no W&B access).

  PYTHONPATH=. python tests/test_finetune_trajectories.py
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import traceback
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis import finetune_trajectories as ft  # noqa: E402

COLS = ["track", "budget", "metric", "target_dataset", "valid_for_primary_analysis", "candidate",
        "finetune_seed", "log_file", "value_log", "value", "wandb_run_id"]


def log_text(run: str, vals: list[float], test: float) -> str:
    lines = [f"device=cuda  num_skills=5  run={run}  init=pretrained  train_students=10  seed=1"]
    lines += [f"epoch {i:3d}  train_loss=0.5000  val_AUC={v:.4f}" for i, v in enumerate(vals, 1)]
    lines += [f"best val AUC   : {max(vals):.4f}", f"test AUC       : {test:.4f}"]
    return "\n".join(lines) + "\n"


def build(tmp: Path, spec: dict, epochs: int = 3) -> Path:
    """spec[(candidate, seed)] = (val list, test, wandb id)."""
    rows = []
    for (c, s), (vals, test, wid) in spec.items():
        name = f"log_{c.replace(':', '_')}_{s}.log"
        (tmp / name).write_text(log_text(f"run_{c}_{s}", vals, test))
        rows.append({"track": "B", "budget": "n3000", "metric": "test_auc", "target_dataset": "t",
                     "valid_for_primary_analysis": "yes", "candidate": c, "finetune_seed": s,
                     "log_file": name, "value_log": f"{test:.4f}", "value": repr(test),
                     "wandb_run_id": wid})
    ex = tmp / "executions.tsv"
    with open(ex, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    return ex


def spec_two_sources():
    own, other = [0.70, 0.72, 0.71], [0.69, 0.73, 0.72]
    return {("src:t", 1): (own, 0.700, "a1"), ("src:t", 2): (own, 0.701, "a2"),
            ("src:o", 1): (other, 0.705, "b1"), ("src:o", 2): (other, 0.706, "b2")}


def test_parse_log_reads_epochs_test_and_run():
    lg = ft.parse_log(log_text("r", [0.6, 0.7], 0.65), 2)
    assert lg["val"] == {1: Decimal("0.6000"), 2: Decimal("0.7000")} and lg["run"] == "r"
    assert lg["test_log"] == Decimal("0.6500")


def test_missing_epoch_is_refused():
    try:
        ft.parse_log(log_text("r", [0.6, 0.7], 0.65), 3)
    except ValueError as ex:
        assert "expected 1 to 3" in str(ex)
    else:
        raise AssertionError("a log with missing epochs was accepted")


def test_winner_own_and_the_epoch_where_the_order_flips():
    with tempfile.TemporaryDirectory() as tmp:
        ex = build(Path(tmp), spec_two_sources())
        runs = ft.load_runs(str(ex), tmp, ("t",), "n3000", 3, {})
        traj, cmp_ = ft.analyse(runs, ("t",), 3)
        assert [r["leader"] for r in traj] == ["src:t", "src:o", "src:o"]
        assert cmp_[0]["winner"] == "src:o" and cmp_[0]["own"] == "src:t"
        assert cmp_[0]["source"] == "log 4 dp" and cmp_[1]["seeds_positive"] == "2/2"


def test_log_that_disagrees_with_the_benchmark_aborts():
    with tempfile.TemporaryDirectory() as tmp:
        ex = build(Path(tmp), spec_two_sources())
        rows = list(csv.DictReader(open(ex), delimiter="\t"))
        rows[0]["value_log"] = "0.9999"
        with open(ex, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        try:
            ft.load_runs(str(ex), tmp, ("t",), "n3000", 3, {})
        except SystemExit as exc:
            assert "the benchmark logged" in str(exc)
        else:
            raise AssertionError("a log disagreeing with the benchmark was accepted")


def test_wandb_best_val_is_used_by_id_and_checked():
    with tempfile.TemporaryDirectory() as tmp:
        spec = spec_two_sources()
        ex = build(Path(tmp), spec)
        wb = {wid: {"id": wid, "name": f"run_{c}_{s}", "summary": {"best/val_auc": max(v) + 1e-6}}
              for (c, s), (v, _t, wid) in spec.items()}
        runs = ft.load_runs(str(ex), tmp, ("t",), "n3000", 3, wb)
        r = runs[("t", "src:o", 1)]
        assert r["best_val_source"] == "W&B full precision"
        assert r["best_val"] == Decimal(repr(0.73 + 1e-6))
        wb["b1"]["name"] = "some_other_run"
        try:
            ft.load_runs(str(ex), tmp, ("t",), "n3000", 3, wb)
        except SystemExit as exc:
            assert "names some_other_run" in str(exc)
        else:
            raise AssertionError("a W&B record naming another run was accepted")
        wb["b1"]["name"] = "run_src:o_1"
        wb["b1"]["summary"]["best/val_auc"] = 0.5
        try:
            ft.load_runs(str(ex), tmp, ("t",), "n3000", 3, wb)
        except SystemExit as exc:
            assert "in W&B" in str(exc)
        else:
            raise AssertionError("a W&B best val far from the log was accepted")


def test_main_writes_three_files():
    with tempfile.TemporaryDirectory() as tmp:
        ex = build(Path(tmp), spec_two_sources())
        out = Path(tmp) / "o"
        assert ft.main(["--executions", str(ex), "--logdir", tmp, "--targets", "t",
                        "--epochs", "3", "--out", str(out)]) == 0
        for suffix in ("_epochs.tsv", "_valtest.tsv", "_report.md"):
            assert Path(f"{out}{suffix}").is_file(), suffix
        assert json.dumps(ft.q("0.00015")) == '"0.0002"'


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
