from __future__ import annotations

"""Tests for analysis/trajectory_table.py on synthetic sources. Run from the code root:

  PYTHONPATH=. python tests/test_trajectory_table.py
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

from analysis import trajectory_table as tt


def write_tsv(p: Path, rows: list[dict]) -> None:
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def test_all_sources() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        ep = [{"target": "t", "candidate": c, "finetune_seed": "1", "epoch": str(e),
               "val_auc": str(0.7 + 0.01 * e + (0.05 if c == "b" and e >= 2 else 0)),
               "log_file": f"run_{c}.log"} for c in ("a", "b") for e in (1, 2, 3)]
        write_tsv(root / "ep.tsv", ep)
        write_tsv(root / "ex.tsv", [{"run_id": f"r{c}", "log_file": f"logs/run_{c}.log",
                                     "elapsed": "00:03:00", "epochs_run": "3",
                                     "gpu_type": "v100-sxm2"} for c in ("a", "b")])
        nd = root / "early_ft" / "rte"
        nd.mkdir(parents=True)
        for m, acc in (("org/x", 0.6), ("org/y", 0.7)):
            (nd / f"{m.replace('/', '__')}_cls_s42.json").write_text(json.dumps(
                {"model": m, "task": "rte", "pooling": "cls", "seed": 42, "gpu": "V100",
                 "epochs": [{"epoch": e, "val_acc": acc + 0.01 * e, "epoch_s": 10.0}
                            for e in (1, 2, 3)]}))
        for v in tt.QT_VERSIONS:
            q = root / "qt" / v / v
            q.mkdir(parents=True)
            cols = ["", "cat__model_m1", "cat__model_m2", "lr"]
            runs = [("r0", "m1", 50), ("r1", "m1", 50), ("r2", "m2", 3), ("r3", "m2", 50)]
            write_csv = [{"": r, "cat__model_m1": "1.0" if m == "m1" else "0.0",
                          "cat__model_m2": "1.0" if m == "m2" else "0.0", "lr": "0.1"}
                         for r, m, _ in runs]
            with (q / "unnormalized_args_table.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                w.writerows(write_csv)
            top1 = {"task": {r: [float(i + (10 if r == "r1" else 0)) for i in range(n)]
                             for r, _, n in runs}}
            (q / "eval_top1.json").write_text(json.dumps(top1))
            (q / "train_time.json").write_text(json.dumps(
                {"task": {r: [2.0] * n for r, _, n in runs}}))
            (q / "eval_time.json").write_text(json.dumps(
                {"task": {r: [1.0] * n for r, _, n in runs}}))
        tt.main(["--studentbert", str(root / "ep.tsv"), "--executions", str(root / "ex.tsv"),
                 "--nlp", str(root / "early_ft"), "--quicktune", str(root / "qt"),
                 "--out", str(root / "traj")])
        rows = list(csv.DictReader((root / "traj.tsv").read_text().splitlines(), delimiter="\t"))
        sb = [r for r in rows if r["benchmark"] == "studentbert"]
        assert len(sb) == 6 and all(abs(float(r["epoch_s"]) - 60.0) < 1e-9 for r in sb)
        assert {r["rank"] for r in sb if r["epoch"] == "1" and r["candidate"] == "a"} == {"1"}
        assert {r["rank"] for r in sb if r["epoch"] == "3" and r["candidate"] == "b"} == {"1"}
        nl = [r for r in rows if r["benchmark"] == "nlp_early"]
        assert len(nl) == 6 and float(nl[-1]["cumulative_s"]) == 30.0
        qt = [r for r in rows if r["benchmark"] == "quicktune_micro"]
        assert {r["run_id"] for r in qt} == {"r0", "r3"}, "rule A must take the first complete run"
        last = [r for r in qt if r["epoch"] == "50"][0]
        assert len(qt) == 100 and float(last["cumulative_s"]) == 150.0


def test_missing_cost_stops() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        write_tsv(root / "ep.tsv", [{"target": "t", "candidate": "a", "finetune_seed": "1",
                                     "epoch": "1", "val_auc": "0.7", "log_file": "x.log"}])
        write_tsv(root / "ex.tsv", [{"run_id": "r", "log_file": "y.log", "elapsed": "00:01:00",
                                     "epochs_run": "1", "gpu_type": "t4"}])
        try:
            tt.main(["--studentbert", str(root / "ep.tsv"), "--executions", str(root / "ex.tsv"),
                     "--out", str(root / "traj")])
        except SystemExit as e:
            assert "no execution record" in str(e)
        else:
            raise AssertionError("a run without a cost record was accepted")


def main() -> None:
    tests = [test_all_sources, test_missing_cost_stops]
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
