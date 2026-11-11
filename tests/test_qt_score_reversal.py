from __future__ import annotations

"""Tests for analysis/qt_score.py and analysis/qt_reversal.py on a synthetic Quick-Tune micro set
(30 tasks, 2 candidates). Run from the code root:

  PYTHONPATH=. python tests/test_qt_score_reversal.py
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from analysis import qt_reversal as qr
from analysis import qt_score as qs


def build(root: Path, overtake_task: int) -> None:
    rng = np.random.default_rng(0)
    rows, log = [], []
    y = np.array([0] * 20 + [1] * 20)
    files = np.array([f"{i:03d}.jpg" for i in range(40)])
    for t in range(30):
        code = f"T{t:02d}"
        (root / "features" / code).mkdir(parents=True)
        for m, sep, noise in (("good", 10.0, 0.1), ("weak", 0.0, 1.0)):
            F = noise * rng.normal(size=(40, 3)) + sep * y[:, None]
            np.savez(root / "features" / code / f"{m}.npz", features=F.astype(np.float32),
                     labels=y, files=files, classes=np.array(["a", "b"]))
            log.append({"task": code, "model": m, "seconds": 2.0, "gpu": "test-gpu"})
            for e in range(1, 51):
                if t == overtake_task:
                    score = 50.0 if m == "good" else (40.0 if e <= 10 else 60.0)
                else:
                    score = 60.0 if m == "good" else 40.0
                rows.append({"benchmark": "quicktune_micro", "target": f"mtlbm/micro/set0/{code}",
                             "candidate": m, "epoch": e, "score": score,
                             "cumulative_s": 10.0 * e})
    with (root / "traj.tsv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (root / "extract_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))


def test_half_is_stratified_and_seeded() -> None:
    y = np.array([0] * 40 + [1] * 40)
    a, b, c = qs.half(y, 42), qs.half(y, 42), qs.half(y, 1)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert (y[a] == 0).sum() == 20 and (y[a] == 1).sum() == 20


def test_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        build(root, overtake_task=5)
        qs.main(["--features", str(root / "features"), "--trajectories", str(root / "traj.tsv"),
                 "--out", str(root / "scores.jsonl"), "--no-pin"])
        n = len((root / "scores.jsonl").read_text().splitlines())
        assert n == 30 * 3 * 2 * 3, n
        qs.main(["--features", str(root / "features"), "--trajectories", str(root / "traj.tsv"),
                 "--out", str(root / "scores.jsonl"), "--no-pin"])
        assert len((root / "scores.jsonl").read_text().splitlines()) == n, "rerun must skip"
        qr.main(["--trajectories", str(root / "traj.tsv"), "--scores", str(root / "scores.jsonl"),
                 "--extract-log", str(root / "extract_log.jsonl"), "--out", str(root / "rev"),
                 "--no-pin"])
        rows = list(csv.DictReader((root / "rev.tsv").read_text().splitlines(), delimiter="\t"))
        assert len(rows) == 90 and all(r["pick"] == "good" for r in rows)
        over = [r for r in rows if r["overtaken"] == "True"]
        assert {r["task"] for r in over} == {"T05"} and len(over) == 3
        assert all(r["settle"] == "11" and r["flag_e10"] == "False" and r["flag_e20"] == "True"
                   for r in over)
        text = (root / "rev_report.md").read_text()
        assert "| 0 (H-score alone) | 0 | 0/3 | 0/87 | 90/90 |" in text, text
        assert "| 10 | " in text and "| 3/3 | 0/87 | 87/90 | 0.00 |" in text, text
        assert "Undefined scores" in text


def test_undefined_logme_is_reported_not_ranked() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        build(root, overtake_task=5)
        qs.main(["--features", str(root / "features"), "--trajectories", str(root / "traj.tsv"),
                 "--out", str(root / "scores.jsonl"), "--no-pin"])
        recs = [json.loads(x) for x in (root / "scores.jsonl").read_text().splitlines()]
        for r in recs:
            if r["estimator"] == "logme" and r["model"] == "weak":
                r["score"], r["info"] = None, {"error": "labels can be interpolated exactly"}
        (root / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        qr.main(["--trajectories", str(root / "traj.tsv"), "--scores", str(root / "scores.jsonl"),
                 "--extract-log", str(root / "extract_log.jsonl"), "--out", str(root / "rev"),
                 "--no-pin"])
        text = (root / "rev_report.md").read_text()
        assert "| logme | 90/180 | 0 |" in text and "logme 90" in text, text


def test_mismatched_files_stop() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        build(root, overtake_task=5)
        z = dict(np.load(root / "features" / "T00" / "weak.npz"))
        z["files"] = z["files"][::-1]
        np.savez(root / "features" / "T00" / "weak.npz", **z)
        try:
            qs.main(["--features", str(root / "features"), "--trajectories",
                     str(root / "traj.tsv"), "--out", str(root / "s.jsonl"), "--no-pin"])
        except SystemExit as e:
            assert "files or labels differ" in str(e)
        else:
            raise AssertionError("mismatched image order was accepted")


def main() -> None:
    tests = [test_half_is_stratified_and_seeded, test_end_to_end,
             test_undefined_logme_is_reported_not_ranked, test_mismatched_files_stop]
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
