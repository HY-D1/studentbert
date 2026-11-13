from __future__ import annotations

"""Tests for analysis/qt_model_level.py on a synthetic Quick-Tune micro set. Run from the code root:

  PYTHONPATH=. python tests/test_qt_model_level.py
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from analysis import qt_model_level as ml

# model: (dimension, complete-run finals, separable features)
MODELS = {"a": (64, [80.0, 70.0, 75.0], True), "b": (256, [20.0, 30.0, 25.0, 10.0], False),
          "c": (32, [90.0, 95.0], True), "d": (160, [40.0, 45.0, 50.0], False)}


def build(root: Path) -> None:
    rng = np.random.default_rng(0)
    d = root / "qt" / "micro" / "micro"
    d.mkdir(parents=True)
    cols = [""] + [f"cat__model_{m}" for m in MODELS] + ["lr"]
    table, top1, scores, log = [], {}, [], []
    y = np.array([0] * 200 + [1] * 200)
    files = np.array([f"{i:03d}.jpg" for i in range(400)])
    for t in range(3):
        task = f"mtlbm/micro/set0/T{t}"
        top1[task] = {}
        for m, (dim, finals, sep) in MODELS.items():
            for i, f in enumerate(finals + [99.0]):
                rid = f"{t}_{m}_{i}"
                row = {c: "0.0" for c in cols}
                row[""], row[f"cat__model_{m}"], row["lr"] = rid, "1.0", "0.1"
                table.append(row)
                n = 50 if i < len(finals) else 7
                top1[task][rid] = [f] * n
            (root / "features" / f"T{t}").mkdir(parents=True, exist_ok=True)
            F = rng.normal(size=(400, dim)) + (4.0 * y[:, None] if sep else 0.0)
            np.savez(root / "features" / f"T{t}" / f"{m}.npz", features=F.astype(np.float32),
                     labels=y, files=files, classes=np.array(["p", "q"]))
            log.append({"task": f"T{t}", "model": m, "dim": dim, "seconds": 1.0, "gpu": "g"})
            for s in (42, 1, 2):
                for e in ml.ESTIMATORS:
                    scores.append({"task": f"T{t}", "seed": s, "model": m, "estimator": e,
                                   "score": float(dim)})
    with (d / "unnormalized_args_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(table)
    (d / "eval_top1.json").write_text(json.dumps(top1))
    (root / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scores))
    (root / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))


def test_outcomes_need_three_complete_runs() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        build(root)
        out, excluded, pairs = ml.outcomes(root / "qt")
        assert pairs == 12 and excluded == 3 and len(out) == 9
        assert out[("T0", "a")] == {"median": 75.0, "mean": 75.0, "best": 80.0, "runs": 3}
        assert out[("T0", "b")]["median"] == 22.5 and out[("T0", "b")]["runs"] == 4


def test_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        build(root)
        ml.main(["--quicktune", str(root / "qt"), "--features", str(root / "features"),
                 "--scores", str(root / "scores.jsonl"), "--extract-log", str(root / "log.jsonl"),
                 "--out", str(root / "ml")])
        rows = list(csv.DictReader((root / "ml.tsv").read_text().splitlines(), delimiter="\t"))
        raw = {(r["estimator"], r["outcome"]): r for r in rows if r["variant"] == "raw"}
        # raw scores equal the dimension, so they pick b (256) over a (64): regret 75 - 22.5
        r = raw[("hscore_shrunk", "median")]
        assert float(r["regret"]) == 52.5 and r["top1"] == "0", r
        pca = {(r["estimator"], r["outcome"]): r for r in rows if r["variant"] == "pca128"}
        # with equal dimension, the separable model a wins on every task and seed
        p = pca[("hscore", "median")]
        assert float(p["regret"]) == 0.0 and p["top1"] == "9", p
        text = (root / "ml_report.md").read_text()
        assert "included (at least 3 complete runs): 9 on 3 tasks; excluded: 3" in text, text
        assert float(raw[("hscore_shrunk", "median")]["score_vs_dim"]) == 1.0


def main() -> None:
    tests = [test_outcomes_need_three_complete_runs, test_end_to_end]
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
