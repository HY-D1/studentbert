"""Tests for the Task 4 pipeline parts that need no transformers or GPU.

  PYTHONPATH=. python tests/test_task4.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
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


ex = _load("t4_extract", "scripts/task4_extract.py")
sc = _load("t4_score", "scripts/task4_score.py")
from analysis import task4_evaluate as ev  # noqa: E402
from src.estimators.logme import logme  # noqa: E402

MODELS = ex.MODELS


def fake_features(tmp, task, model, seed, sep):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, 300)
    F = rng.normal(size=(300, 8)) + sep * np.eye(8)[y]
    p = ex.out_path(tmp, task, model, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, cls=F, mean=F + 0.01, labels=y, idx=np.arange(300))
    p.with_suffix(".json").write_text(json.dumps(
        {"task": task, "model": model, "seed": seed, "extract_s": 1.0, "peak_gpu_mb": 10.0}))
    return p


def run_score(argv):
    with contextlib.redirect_stdout(io.StringIO()):
        return sc.main(argv)


def test_sample_is_deterministic_sorted_unique_and_capped():
    a, b = ex.sample_indices(1000, 50, 42), ex.sample_indices(1000, 50, 42)
    assert (a == b).all() and (np.diff(a) > 0).all() and a.size == 50
    assert not (a == ex.sample_indices(1000, 50, 1)).all()
    assert ex.sample_indices(30, 50, 42).size == 30


def test_out_path_keeps_task_and_slugs_model():
    p = ex.out_path("/x", "rte", "dmis-lab/biobert-v1.1", 2)
    assert p.as_posix() == "/x/rte/dmis-lab__biobert-v1.1_s2.npz"


def test_multiclass_logme_is_the_mean_of_one_vs_rest():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    F = rng.normal(size=(200, 5)) + np.eye(5)[y]
    want = np.mean([logme(F, (y == c).astype(float))[0] for c in range(3)])
    assert abs(sc.logme_multiclass(F, y)[0] - want) < 1e-12


def test_separable_features_score_higher():
    with tempfile.TemporaryDirectory() as t:
        good = sc.score_file(fake_features(t, "rte", MODELS[0], 1, 3.0))
        bad = sc.score_file(fake_features(t, "mnli", MODELS[0], 1, 0.0))
        for est in sc.ESTIMATORS:
            g = [r["score"] for r in good if r["estimator"] == est and r["pooling"] == "cls"][0]
            b = [r["score"] for r in bad if r["estimator"] == est and r["pooling"] == "cls"][0]
            assert g > b, (est, g, b)


def test_scoring_is_rerun_safe_and_keeps_tasks_apart():
    with tempfile.TemporaryDirectory() as t:
        fake_features(t, "rte", MODELS[0], 1, 1.0)
        fake_features(t, "mnli", MODELS[0], 1, 1.0)
        out = f"{t}/s.jsonl"
        run_score(["--features", t, "--out", out])
        run_score(["--features", t, "--out", out])
        recs = [json.loads(ln) for ln in Path(out).read_text().splitlines()]
        assert len(recs) == 2 * 2 * len(sc.ESTIMATORS), len(recs)
        assert {r["task"] for r in recs} == {"rte", "mnli"}


def write_gold(tmp, means, sds=None):
    p = Path(tmp) / "g.tsv"
    lines = ["dataset\tmodel\tpooling\tlogme\tfrozen_mean\tfrozen_sd\ttuned_mean\ttuned_sd"]
    for i, (m, v) in enumerate(zip(MODELS, means)):
        sd = sds[i] if sds else 0.01
        lines.append(f"rte\t{m}\tcls\t{-0.7 + 0.01 * i}\t{v}\t{sd}\t{100 - v}\t{sd}")
    p.write_text("\n".join(lines) + "\n")
    return ev.load_gold(p)


def scores_from(values):
    return {("hscore", "rte", "cls"): {s: dict(zip(MODELS, values)) for s in (42, 1)},
            ("logme", "rte", "cls"): {s: {m: -0.7 + 0.01 * i for i, m in enumerate(MODELS)}
                                     for s in (42, 1)}}


def test_perfect_scores_on_frozen_are_reversed_on_tuned():
    with tempfile.TemporaryDirectory() as t:
        means = [50, 52, 54, 56, 58, 60, 62]
        rows, repro = ev.evaluate(scores_from(means), write_gold(t, means))
        fr = [r for r in rows if r["estimator"] == "hscore" and r["regime"] == "frozen"]
        tu = [r for r in rows if r["estimator"] == "hscore" and r["regime"] == "tuned"]
        assert all(r["rho"] == 1.0 and r["top1"] and r["regret"] == 0 for r in fr)
        assert all(r["rho"] == -1.0 and not r["top1"] for r in tu)
        assert all(abs(r["seed_stability"] - 1.0) < 1e-12 for r in fr)
        assert repro and abs(repro[0]["spearman_vs_published"] - 1.0) < 1e-12


def test_tie_rule_margin_and_interval():
    assert ev.tied((80.00, 0.0), (80.10, 0.0))
    assert not ev.tied((80.00, 0.0), (80.20, 0.0))
    assert ev.tied((80.0, 1.0), (81.0, 1.0))
    assert not ev.tied((80.0, 0.1), (81.0, 0.1))


def test_tied_pick_is_not_counted_as_a_miss():
    with tempfile.TemporaryDirectory() as t:
        means = [50, 52, 54, 56, 58, 62.05, 62]
        vals = [0, 1, 2, 3, 4, 5, 6]
        rows, _ = ev.evaluate(scores_from(vals), write_gold(t, means))
        r = [x for x in rows if x["estimator"] == "hscore" and x["regime"] == "frozen"][0]
        assert not r["top1"] and r["tied_with_best"] and abs(r["regret"] - 0.05) < 1e-9, r


def test_missing_model_aborts():
    with tempfile.TemporaryDirectory() as t:
        s = scores_from([1, 2, 3, 4, 5, 6, 7])
        del s[("hscore", "rte", "cls")][42][MODELS[0]]
        try:
            ev.evaluate(s, write_gold(t, [50, 52, 54, 56, 58, 60, 62]))
        except SystemExit as exc:
            assert "of 7 models" in str(exc)
        else:
            raise AssertionError("missing model did not abort")


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
