from __future__ import annotations

"""Tests for tools/compare_ft_epochs.py on synthetic run records: identical and differing first
epochs, a missing epoch, changed settings, a missing recorded run and the run count. Run from the
code root:

  PYTHONPATH=. python tests/test_compare_ft_epochs.py
"""

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

from tools import compare_ft_epochs as cf


def record(accs: list[float], **over) -> dict:
    r = {"model": "bert-base-uncased", "task": "rte", "pooling": "cls", "seed": 42,
         "n_train": 10000, "n_val": 277, "train_idx_md5": "a", "val_idx_md5": "b", "lr": 5e-5,
         "batch_size": 32, "max_length": 256, "gpu": "Tesla V100-SXM2-32GB",
         "epochs": [{"epoch": i + 1, "val_acc": v, "epoch_s": 36.0} for i, v in enumerate(accs)]}
    r.update(over)
    return r


OLD = [0.5, 0.6, 0.7]
NEW = OLD + [0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77]


def test_identical_first_epochs() -> None:
    r = cf.compare(record(NEW), record(OLD), 10)
    assert r["identical"] and r["max_diff"] == 0.0 and r["run_s"] == 360.0, r


def test_difference_is_reported_not_fatal() -> None:
    new = record([0.5, 0.625, 0.7] + NEW[3:])
    r = cf.compare(new, record(OLD), 10)
    assert not r["identical"] and abs(r["max_diff"] - 0.025) < 1e-12, r


def expect_error(new: dict, old: dict, text: str) -> None:
    try:
        cf.compare(new, old, 10)
    except ValueError as e:
        assert text in str(e), e
        return
    raise AssertionError(f"no error for {text}")


def test_missing_epoch_and_changed_settings_stop() -> None:
    expect_error(record(NEW[:9]), record(OLD), "expected 1..10")
    expect_error(record(NEW, lr=1e-4), record(OLD), "['lr']")
    expect_error(record(NEW, val_idx_md5="c"), record(OLD), "val_idx_md5")
    expect_error(record(NEW), record(OLD[:2]), "lacks epochs 1 to 3")


def run_main(new: Path, old: Path, expect: int) -> tuple[str, str]:
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            cf.main(["--new", str(new), "--old", str(old), "--expect", str(expect)])
    except SystemExit as e:
        return out.getvalue(), str(e.code)
    return out.getvalue(), ""


def test_main_counts_and_missing_recorded_run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        new, old = Path(tmp) / "new" / "rte", Path(tmp) / "old" / "rte"
        new.mkdir(parents=True)
        old.mkdir(parents=True)
        (new / "a_cls_s42.json").write_text(json.dumps(record(NEW)))
        (old / "a_cls_s42.json").write_text(json.dumps(record(OLD)))
        (new / "b_cls_s42.json").write_text(json.dumps(record([0.5, 0.6, 0.75] + NEW[3:])))
        (old / "b_cls_s42.json").write_text(json.dumps(record(OLD)))
        text, err = run_main(new.parent, old.parent, 2)
        assert err == "", err
        assert "epochs 1 to 3 identical: 1/2" in text and "largest difference 0.0500" in text, text
        assert "GPU-hours of these runs: 0.200" in text, text
        _, err = run_main(new.parent, old.parent, 3)
        assert err.startswith("ABORT: 2 runs"), err
        (new / "c_cls_s42.json").write_text(json.dumps(record(NEW)))
        _, err = run_main(new.parent, old.parent, 3)
        assert err.startswith("ABORT: no recorded run"), err


def main() -> None:
    tests = [test_identical_first_epochs, test_difference_is_reported_not_fatal,
             test_missing_epoch_and_changed_settings_stop, test_main_counts_and_missing_recorded_run]
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
