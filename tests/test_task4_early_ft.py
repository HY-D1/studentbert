from __future__ import annotations

"""Tests for scripts/task4_early_ft.py parts that need no torch or GPU.

  PYTHONPATH=. python tests/test_task4_early_ft.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("t4ft", REPO / "scripts/task4_early_ft.py")
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)


def test_validation_rows_exclude_training_rows_and_cap_at_2000():
    train = ft.ex.sample_indices(120000, 10000, 42)
    v = ft.val_indices(120000, train, 42)
    assert v.size == 2000 and np.intersect1d(v, train).size == 0
    assert np.array_equal(v, ft.val_indices(120000, train, 42))
    assert not np.array_equal(v, ft.val_indices(120000, ft.ex.sample_indices(120000, 10000, 1), 1))


def test_small_official_validation_split_is_used_whole():
    v = ft.val_indices(277, None, 42)
    assert np.array_equal(v, np.arange(277))


def test_output_path_names_model_task_pooling_seed():
    p = ft.out_path("/x", "rte", "dmis-lab/biobert-v1.1", "mean", 2)
    assert str(p) == "/x/rte/dmis-lab__biobert-v1.1_mean_s2.json"


def test_epoch_check_rejects_missing_or_bad_epochs():
    good = {"epochs": [{"epoch": e, "val_acc": 0.5, "train_loss": 0.7} for e in (1, 2, 3)]}
    ft.check_epochs(good, 3)
    for bad in ({"epochs": good["epochs"][:2]},
                {"epochs": [{"epoch": 1, "val_acc": 1.5, "train_loss": 0.7}] + good["epochs"][1:]},
                {"epochs": [{"epoch": 1, "val_acc": 0.5, "train_loss": float("nan")}]
                 + good["epochs"][1:]}):
        try:
            ft.check_epochs(bad, 3)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


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
