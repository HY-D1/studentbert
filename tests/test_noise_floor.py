from __future__ import annotations

import csv
import json
import random
import sys
import tempfile
from fractions import Fraction as F
from pathlib import Path

from analysis import noise_floor as nf
from analysis import regret_compute as rc
from analysis import reversal_table as rt

CANDS = [f"c{i}" for i in range(8)]


def sb_decision(target: str, seed: int, curves: dict, outcome: dict) -> rc.Decision:
    costs = {c: [6.0 * (e + 1) for e in range(20)] for c in CANDS}
    hs = {c: float(8 - i) for i, c in enumerate(CANDS)}
    return rc.Decision("studentbert", target, f"{target}/{seed}", outcome, hs, curves, costs,
                       4.0, lambda d, a, b, k: rc.SB_MARGIN, pick="c0")


def sb_fixture(rng: random.Random) -> tuple[list[rc.Decision], dict]:
    ds, curves = [], {}
    for t in rt.TARGETS:
        outcome = {c: 0.7 + 0.01 * i for i, c in enumerate(CANDS)}
        for s in (42, 1, 2, 3, 4, 5):
            curves[(t, s)] = {c: [0.6 + rng.random() * 0.1 for _ in range(20)] for c in CANDS}
        for s in (42, 1, 2):
            ds.append(sb_decision(t, s, curves[(t, s)], outcome))
    return ds, curves


def test_methods_reproduce_regret_compute() -> None:
    ds, _ = sb_fixture(random.Random(1))
    rows = nf.run_methods(ds, nf.method_list("studentbert", rc.BUDGETS["studentbert"]))
    nf.check_methods("studentbert", ds, rows)
    assert {r["method"] for r in rows} >= {"static H-score", "full fine-tuning",
                                           "successive halving (factor 3)"}
    bad = [dict(r) for r in rows]
    bad[0]["regret"] += 1.0
    try:
        nf.check_methods("studentbert", ds, bad)
    except SystemExit as e:
        assert "differs from regret_compute" in str(e.code)
    else:
        raise AssertionError("a changed row was not caught")


def test_full_floor_and_spread() -> None:
    ds, curves = sb_fixture(random.Random(2))
    fl = nf.full_floor(ds)
    for d in ds:
        best = max(CANDS, key=lambda c: (max(d.curves[c]), -CANDS.index(c)))
        assert fl[d.key] == F(d.best) - F(d.outcome[best])
    table = {(t, s): {c: dict(enumerate(v, 1)) for c, v in cv.items()}
             for (t, s), cv in curves.items()}
    sp = nf.sb_spread(ds, table)
    assert sorted(sp) == [1, 2, 3, 4, 5, 42]
    for s in (42, 1, 2):
        want = sum((fl[f"{t}/{s}"] for t in rt.TARGETS), F(0)) / len(rt.TARGETS)
        assert sp[s] == want, (s, sp[s], want)


def write_executions(path: Path, gold: dict) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["track", "budget", "metric", "valid_for_primary_analysis", "target_dataset",
                    "candidate", "finetune_seed", "value"])
        for (t, c, s), v in gold.items():
            w.writerow(["B", "n3000", "test_auc", "yes", t, c, s, repr(v)])
        w.writerow(["A", "n3000", "test_auc", "yes", rt.TARGETS[0], "c0", 42, "0.1"])


def test_seed_oracle() -> None:
    gold = {}
    for t in rt.TARGETS:
        for i, c in enumerate(CANDS):
            for s in (42, 1, 2, 3, 4, 5):
                gold[(t, c, s)] = 0.70 + 0.01 * i + (0.10 if (c, s) == ("c0", 3) else 0.0)
    ds = []
    for t in rt.TARGETS:
        outcome = {c: sum(gold[(t, c, s)] for s in (42, 1, 2, 3, 4, 5)) / 6 for c in CANDS}
        flat = {c: [0.5] * 20 for c in CANDS}
        ds += [sb_decision(t, s, flat, outcome) for s in (42, 1, 2)]
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "executions.tsv"
        write_executions(p, gold)
        orc = nf.sb_seed_oracle(ds, p)
        assert orc[42] == 0 and orc[3] > 0, orc
        d = ds[0]
        assert orc[3] == F(d.best) - F(d.outcome["c0"])
        bad = dict(gold)
        bad[(rt.TARGETS[0], "c1", 42)] += 0.06
        write_executions(p, bad)
        try:
            nf.sb_seed_oracle(ds, p)
        except SystemExit as e:
            assert "differs from the gold" in str(e.code)
        else:
            raise AssertionError("a changed gold was not caught")


def qt_tree(root: Path, runs: dict) -> None:
    d = root / "micro" / "micro"
    d.mkdir(parents=True)
    models = sorted({m for _, m in runs})
    rows, top1 = [], {}
    for (task, m), curves in runs.items():
        for i, cv in enumerate(curves):
            rid = f"{task}_{m}_{i}"
            rows.append({"id": rid, **{f"cat__model_{x}": "1" if x == m else "0" for x in models}})
            top1.setdefault(f"set/{task}", {})[rid] = cv
    with (d / "unnormalized_args_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (d / "eval_top1.json").write_text(json.dumps(top1))


def qt_decision(task: str, curves: dict) -> rc.Decision:
    outcome = {m: v[-1] for m, v in curves.items()}
    costs = {m: [1.0] * rc.QT_EPOCHS for m in curves}
    return rc.Decision("quicktune", task, f"{task}/42", outcome, {m: 1.0 for m in curves},
                       curves, costs, 0.0, lambda d, a, b, k: rc.QT_MARGIN,
                       static_on_axis=False)


def test_quicktune_runs_and_floor() -> None:
    n = rc.QT_EPOCHS
    runs = {("t1", "ma"): [[50.0] * n, [60.0] * n, [70.0] * n],
            ("t1", "mb"): [[65.0] * n, [65.0] * n, [65.0] * n],
            ("t1", "mc"): [[10.0] * n, [10.0] * n]}
    with tempfile.TemporaryDirectory() as tmp:
        qt_tree(Path(tmp), runs)
        got = nf.qt_runs(Path(tmp))
    assert set(got) == {("t1", "ma"), ("t1", "mb")}, "models under QT_MIN_RUNS are dropped"
    d = qt_decision("t1", {"ma": [60.0] * n, "mb": [65.0] * n})
    nf.check_qt_runs([d], got)
    floor, spread = nf.qt_floor([d], got, 3000)
    assert abs(float(floor["t1"]) - 5.0 / 3) < 0.15, floor
    assert spread[0] <= float(floor["t1"]) <= spread[1]
    bad = qt_decision("t1", {"ma": [61.0] * n, "mb": [65.0] * n})
    try:
        nf.check_qt_runs([bad], got)
    except SystemExit as e:
        assert "run medians differ" in str(e.code)
    else:
        raise AssertionError("a changed curve was not caught")


def nlp_decision(seed: int, v10: tuple[float, float]) -> rc.Decision:
    curves = {"a": [0.5] * 9 + [v10[0]], "b": [0.5] * 9 + [v10[1]]}
    return rc.Decision("nlp", "rte", f"rte/cls/{seed}", {"a": 80.0, "b": 70.0},
                       {"a": 2.0, "b": 1.0}, curves, {c: [1.0] * 10 for c in curves}, 1.0,
                       lambda d, a, b, k: 0.0, pick="a")


def test_nlp_leave_one_out() -> None:
    ds = [nlp_decision(s, (0.75, 0.5)) for s in (42, 1)] + [nlp_decision(2, (0.5, 0.875))]
    got = nf.nlp_leave_one_out(ds)
    want = (F(625, 100) + F(625, 100) + F(25)) / 3
    assert got == want, (got, want)


def test_relative_and_label() -> None:
    ds = [nlp_decision(s, (0.6, 0.5)) for s in (42, 1)]
    ds[1].cluster = "mnli"
    rows = [{"method": "static H-score", "param": "", "decisions": 2, "chosen": 2,
             "compute_gpu_h": 0.0, "regret": 0.0, "choices": {d.key: "a" for d in ds}},
            {"method": "early leader", "param": "epoch 1", "decisions": 2, "chosen": 2,
             "compute_gpu_h": 0.0, "regret": 10.0, "choices": {d.key: "b" for d in ds}},
            {"method": "warning, abstain", "param": "epoch 1", "decisions": 2, "chosen": 0,
             "compute_gpu_h": 0.0, "regret": float("nan"), "choices": {}}]
    floor = {ds[0].key: F(0), ds[1].key: F(10)}
    out = {r["method"]: r for r in nf.relative(ds, rows, floor, F(5))}
    assert out["static H-score"]["excess"] == F(-5) and out["static H-score"]["norm_excess"] == -1
    assert out["early leader"]["excess"] == F(5) and out["early leader"]["vs_static"] == 10
    assert out["early leader"]["label"] == "not distinguishable from the floor"
    assert nf.label((1.0, 2.0)) == "above the floor"
    assert out["warning, abstain"]["followed"] == 0
    assert nf.label((-1.0, 1.0)) == "not distinguishable from the floor"
    assert nf.label((-2.0, -1.0)) == "below the floor"


def main() -> None:
    tests = [test_methods_reproduce_regret_compute, test_full_floor_and_spread, test_seed_oracle,
             test_quicktune_runs_and_floor, test_nlp_leave_one_out, test_relative_and_label]
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
