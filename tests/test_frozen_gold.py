"""Tests for scripts/frozen_gold_kt.py (failure review D6).

  PYTHONPATH=. python tests/test_frozen_gold.py                 (torch tests skip without torch)
  PYTHONPATH=. python tests/test_frozen_gold.py --require-torch (a skip counts as a failure)
"""

from __future__ import annotations

import importlib.util
import json
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


fg = _load("frozen_gold_kt", "scripts/frozen_gold_kt.py")


class _Skip(Exception):
    pass


def _torch():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        raise _Skip("torch not installed") from None


def test_linear_readout_recovers_a_separable_signal():
    from src.eval.metrics import auc

    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 5))
    y = (x[:, 0] + 0.1 * rng.normal(size=2000) > 0).astype(int)
    params, info = fg.fit_logistic(x, y, lam=1e-4)
    assert info["converged"]
    assert auc(y, fg.predict(params, x)) > 0.99


def test_skill_bias_learns_base_rates_that_a_shared_readout_cannot():
    from src.eval.metrics import auc

    rng = np.random.default_rng(1)
    g = rng.integers(0, 4, size=4000)
    y = (rng.random(4000) < np.array([0.1, 0.4, 0.6, 0.9])[g]).astype(int)
    x = rng.normal(size=(4000, 3))
    lin, _ = fg.fit_logistic(x, y, lam=1e-4)
    sb, _ = fg.fit_logistic(x, y, groups=g, n_groups=4, lam=1e-4)
    assert auc(y, fg.predict(lin, x)) < 0.6
    assert auc(y, fg.predict(sb, x, g)) > 0.75


def test_standardize_uses_train_statistics_only():
    rng = np.random.default_rng(2)
    xtr, xte = rng.normal(size=(50, 3)), rng.normal(size=(20, 3))
    a_tr, a_te = fg.standardize(xtr, xte)
    b_tr, b_te = fg.standardize(xtr, xte * 100 + 7)
    assert np.array_equal(a_tr, b_tr)
    assert np.allclose(b_te, (xte * 100 + 7 - xtr.mean(0)) / xtr.std(0))
    assert np.allclose(a_tr.mean(0), 0) and np.allclose(a_tr.std(0), 1)


def test_test_draw_is_fixed_sorted_and_capped():
    a, b = fg.test_learners(100, 30), fg.test_learners(100, 30)
    assert a == b and a == sorted(set(a)) and len(a) == 30
    assert fg.test_learners(10, 30) == list(range(10))


def test_records_cannot_be_read_as_estimator_scores():
    from analysis.evaluate_estimators import logme_scores

    rec = {"target": "t", "candidate": "edubert_t_pretrain_full_encoder.pt", "seed": 1,
           "gold_side": True, "readout": "frozen_linear", "lam": 1e-4, "frozen_test_auc": 0.7,
           "details": {}}
    assert not {"estimator", "score", "metadata"} & set(rec)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "r.jsonl"
        p.write_text(json.dumps(rec) + "\n")
        try:
            logme_scores([str(p)])
        except KeyError:
            return
        raise AssertionError("a frozen-gold record was accepted as an estimator score")


def test_end_to_end_is_rerun_safe_and_keeps_test_learners_apart():
    _torch()
    from tests.test_estimators import _make_processed

    with tempfile.TemporaryDirectory() as tmp:
        tgt = _make_processed(Path(tmp), "tgt")
        out = Path(tmp) / "fg.jsonl"
        argv = ["--target_dir", str(tgt), "--candidates", "scratch", "--seeds", "7", "3",
                "--n_students", "12", "--max_positions", "150", "--max_seq_len", "64",
                "--out", str(out)]
        assert fg.main(argv) == 0
        recs = [json.loads(ln) for ln in out.read_text().splitlines()]
        assert len(recs) == 2 * len(fg.READOUTS) * len(fg.LAMS)
        aucs = [r["frozen_test_auc"] for r in recs]
        assert all(0.0 <= v <= 1.0 for v in aucs if v == v), aucs
        assert {r["details"]["n_test_learners"] for r in recs} == {2}
        before = out.read_bytes()
        assert fg.main(argv) == 0
        assert out.read_bytes() == before, "a rerun wrote records again"


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
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed - skipped} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
