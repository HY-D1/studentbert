"""Tests for scripts/split_bootstrap_kt.py (failure review D10).

  PYTHONPATH=. python tests/test_split_bootstrap.py                 (torch tests skip without torch)
  PYTHONPATH=. python tests/test_split_bootstrap.py --require-torch (a skip counts as a failure)
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sb = _load("split_bootstrap_kt", "scripts/split_bootstrap_kt.py")


class _Skip(Exception):
    pass


def _torch():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        raise _Skip("torch not installed") from None


def toy(n_learners=40, per=30, seed=0):
    rng = np.random.default_rng(seed)
    owner = np.repeat(np.arange(n_learners), per)
    y = rng.integers(0, 2, size=owner.size).astype(float)
    return y, owner, rng


def test_learner_index_covers_every_position_once():
    owner = np.array([2, 0, 0, 1, 2, 2])
    pos = sb.learner_index(owner, 4)
    assert [sorted(p.tolist()) for p in pos] == [[1, 2], [3], [0, 4, 5], []]


def test_identical_models_give_a_zero_statistic_on_every_resample():
    y, owner, rng = toy()
    p = rng.random(y.size)
    d = sb.bootstrap(y, owner, [p, p], [p.copy(), p.copy()], 40, 50, np.random.default_rng(1))
    assert np.all(d == 0)


def test_bootstrap_is_seeded_and_centred_on_a_real_difference():
    y, owner, rng = toy(n_learners=200, per=40)
    noise = rng.normal(size=y.size)
    pa = [y + 1.5 * noise]
    pb = [y + 1.0 * noise]
    obs = sb.paired_stat(y, pa, pb, np.arange(y.size))
    d1 = sb.bootstrap(y, owner, pa, pb, 200, 200, np.random.default_rng(3))
    d2 = sb.bootstrap(y, owner, pa, pb, 200, 200, np.random.default_rng(3))
    assert np.array_equal(d1, d2) and obs > 0
    assert np.percentile(d1, 2.5) > 0 and abs(np.mean(d1) - obs) < 0.01


def test_recorded_test_needs_every_run():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "e.tsv"
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["run_id", "metric", "valid_for_primary_analysis",
                                               "value"], delimiter="\t")
            w.writeheader()
            w.writerow({"run_id": "r1", "metric": "test_auc", "valid_for_primary_analysis": "yes",
                        "value": "0.7"})
        assert sb.recorded_test(str(p), ["r1"]) == {"r1": 0.7}
        try:
            sb.recorded_test(str(p), ["r1", "r2"])
        except SystemExit as exc:
            assert "r2" in str(exc)
        else:
            raise AssertionError("a run without a recorded test AUC was accepted")


def test_end_to_end_reproduces_and_guards_the_recorded_test_auc():
    _torch()
    import torch

    from tests.test_estimators import _make_processed

    ft = _load("finetune_edubert", "scripts/finetune_edubert.py")
    with tempfile.TemporaryDirectory() as tmp:
        tgt = _make_processed(Path(tmp), "tgt", n_students=40)
        ck_dir = Path(tmp) / "ck"
        ck_dir.mkdir()
        from src.data.dataset import InteractionDataset

        test_ds = InteractionDataset(str(tgt), "test", 512)
        values = {}
        for name in ("a_seed1", "a_seed2", "b_seed1", "b_seed2"):
            torch.manual_seed(len(values))
            m = ft.EduBERTForKT(num_skills=6, d_model=16, n_layers=2, max_len=512)
            torch.save({"model_state": m.state_dict(), "num_skills": 6, "epoch": 1,
                        "config": {"d_model": 16, "n_layers": 2, "dropout": 0.1,
                                   "max_seq_len": 512}}, ck_dir / f"{name}_best.pt")
            y, p, _, _ = sb.predict(m, test_ds, "cpu")
            values[name] = sb.auc(y, p)
        ex = Path(tmp) / "e.tsv"

        def write(vals):
            with open(ex, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["run_id", "metric",
                                                   "valid_for_primary_analysis", "value"],
                                   delimiter="\t")
                w.writeheader()
                for k, v in vals.items():
                    w.writerow({"run_id": k, "metric": "test_auc",
                                "valid_for_primary_analysis": "yes", "value": repr(v)})

        write(values)
        argv = ["--processed_dir", str(tgt), "--executions", str(ex), "--ckpt_dir", str(ck_dir),
                "--a_run", "a_seed{s}", "--b_run", "b_seed{s}", "--seeds", "1", "2",
                "--boots", "20", "--out", str(Path(tmp) / "o")]
        assert sb.main(argv) == 0
        assert (Path(tmp) / "o_report.md").is_file()
        write({**values, "b_seed2": values["b_seed2"] + 0.01})
        try:
            sb.main(argv)
        except SystemExit as exc:
            assert "does not reproduce" in str(exc)
        else:
            raise AssertionError("a checkpoint that does not reproduce its test AUC was accepted")


def main() -> int:
    require = "--require-torch" in sys.argv
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except _Skip as ex:
            if require:
                failed += 1
                print(f"FAIL  {name}  (skipped under --require-torch: {ex})")
            else:
                skipped += 1
                print(f"SKIP  {name}  ({ex})")
        except (Exception, SystemExit):
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed - skipped} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
