from __future__ import annotations

import json
import math
import sys
import tempfile
from fractions import Fraction as F
from pathlib import Path

from analysis import nlp_budget_extension as nbe
from analysis import regret_compute as rc

CANDS = ("enc/a", "b", "c")


def margin(d: rc.Decision, a: str, b: str, k: int) -> float:
    q = (d.val(a, k) + d.val(b, k)) / 2
    return 1.96 * math.sqrt(2 * q * (1 - q) / 277)


def decision(task: str, seed: int, curves: dict, outcome=(80.0, 70.0, 60.0)) -> rc.Decision:
    costs = {c: [10.0 * (i + 1) for i in range(len(v))] for c, v in curves.items()}
    return rc.Decision("nlp", task, f"{task}/cls/{seed}", dict(zip(CANDS, outcome, strict=True)),
                       {"enc/a": 3.0, "b": 2.0, "c": 1.0}, curves, costs, 5.0, margin,
                       pick="enc/a")


def record(accs: list[float], model: str, task: str, seed: int, **over) -> dict:
    r = {"model": model, "task": task, "pooling": "cls", "seed": seed, "n_train": 10000,
         "n_val": 277, "train_idx_md5": "a", "val_idx_md5": "b", "lr": 5e-5, "batch_size": 32,
         "max_length": 256, "gpu": "Tesla V100-SXM2-32GB",
         "epochs": [{"epoch": i + 1, "val_acc": v, "epoch_s": 30.0} for i, v in enumerate(accs)]}
    r.update(over)
    return r


OLD = {"enc/a": [0.5, 0.6, 0.7], "b": [0.55, 0.58, 0.6], "c": [0.4, 0.45, 0.5]}


def tree(tmp: Path, new_tail: dict, seed: int = 42, task: str = "rte") -> tuple[list, list]:
    old_dirs = [tmp / f"old{g}" for g in range(3)]
    new_dirs = [tmp / f"new{g}" for g in range(3)]
    g = nbe.SEED_GROUP.get(seed, 0)
    for m, v in OLD.items():
        name = f"{m.replace('/', '__')}_cls_s{seed}.json"
        for root, accs in ((old_dirs[g], v), (new_dirs[g], v + new_tail[m])):
            (root / task).mkdir(parents=True, exist_ok=True)
            (root / task / name).write_text(json.dumps(record(accs, m, task, seed)))
    return old_dirs, new_dirs


TAIL = {m: [0.7 + 0.01 * i for i in range(7)] for m in OLD}


def expect_exit(fn, text: str) -> None:
    try:
        fn()
    except SystemExit as e:
        assert text in str(e.code), e.code
        return
    raise AssertionError(f"no exit for {text}")


def test_extend_replaces_curves_only() -> None:
    with tempfile.TemporaryDirectory() as t:
        old_dirs, new_dirs = tree(Path(t), TAIL)
        d = decision("rte", 42, {m: list(v) for m, v in OLD.items()})
        (nd,), comps = nbe.extend([d], new_dirs, old_dirs, 10)
        assert nd.epochs == 10 and d.epochs == 3 and d.curves["b"] == OLD["b"]
        assert nd.curves["b"] == OLD["b"] + TAIL["b"] and nd.costs["c"][9] == 300.0
        assert (nd.pick, nd.order, nd.outcome, nd.static_s) == (d.pick, d.order, d.outcome,
                                                               d.static_s)
        assert len(comps) == 3 and all(c["identical"] for c in comps)
        assert nbe.manifest(new_dirs[:1], {c["file"] for c in comps}, 3)
        expect_exit(lambda: nbe.manifest(new_dirs[:1], set(), 3), "belong to no decision")
        expect_exit(lambda: nbe.manifest(new_dirs[:1], set(), 4), "expected 4")


def test_extend_stops_on_a_wrong_run() -> None:
    with tempfile.TemporaryDirectory() as t:
        old_dirs, new_dirs = tree(Path(t), {**TAIL, "c": TAIL["c"][:5]})
        d = decision("rte", 42, {m: list(v) for m, v in OLD.items()})
        expect_exit(lambda: nbe.extend([d], new_dirs, old_dirs, 10), "expected 1..10")
    with tempfile.TemporaryDirectory() as t:
        old_dirs, new_dirs = tree(Path(t), TAIL)
        d = decision("rte", 42, {**{m: list(v) for m, v in OLD.items()}, "b": [0.5, 0.5, 0.5]})
        expect_exit(lambda: nbe.extend([d], new_dirs, old_dirs, 10), "not the run behind")
    with tempfile.TemporaryDirectory() as t:
        old_dirs, new_dirs = tree(Path(t), TAIL, seed=9)
        d = decision("rte", 9, {m: list(v) for m, v in OLD.items()})
        expect_exit(lambda: nbe.extend([d], new_dirs, old_dirs, 10), "no seed group")


def three_epoch() -> list[rc.Decision]:
    return [decision("rte", 42, {"enc/a": [0.5, 0.6, 0.7], "b": [0.6, 0.65, 0.66],
                                 "c": [0.4, 0.5, 0.75]}),
            decision("mnli", 1, {"enc/a": [0.7, 0.72, 0.74], "b": [0.6, 0.65, 0.66],
                                 "c": [0.5, 0.55, 0.6]}, (60.0, 80.0, 70.0))]


def test_references_reproduce_regret_compute() -> None:
    ds = three_epoch()
    nbe.check_references(ds)
    rows = {(r["method"], r["param"]): r for r in nbe.references(ds, (1, 2, 3), False)}
    assert rows[("early leader", "epoch 1")]["choices"] == ["b", "enc/a"]
    assert rows[("early leader", "epoch 3")]["choices"] == ["c", "enc/a"]
    sh = rows[("successive halving", "rungs 1,2,3")]
    assert sh["choices"] == ["b", "enc/a"] and sh["regret"] == 15.0, sh
    assert sh["compute_gpu_h"] == 50 / 3600, "two candidates to epoch 2, one to epoch 1"


def test_geometric_rows_cap_at_ten() -> None:
    curves = {m: [0.5 + 0.01 * i + 0.001 * j for i in range(10)]
              for j, m in enumerate(CANDS)}
    ds = [decision("rte", 42, curves)]
    rows = {(r["method"], r["param"]): r for r in nbe.references(ds, (1, 2, 3, 5, 10), True)}
    assert len(rows) == 5 + 2 * 4
    assert rc.geometric(3, 2, 3, 10) == [3, 6]
    assert rows[("early leader", "epoch 10")]["choices"] == ["c"]


def entries(rs: list[F]) -> list[dict]:
    return [{"label": f"k{i}", "compute": float(i), "R": r} for i, r in enumerate(rs)]


def test_readout_flat_monotone_u_and_tie() -> None:
    ro = nbe.readout(entries([F(1, 2)] * 4))
    assert ro["best"]["label"] == "k0" and ro["flat"]["label"] == "k0" and not ro["worse"]
    ro = nbe.readout(entries([F(5, 10), F(3, 10), F(2, 10), F(19, 100)]))
    assert ro["best"]["label"] == "k3" and ro["flat"]["label"] == "k2", ro
    ro = nbe.readout(entries([F(5, 10), F(1, 10), F(2, 10), F(3, 10)]))
    assert ro["best"]["label"] == "k1" and [e["label"] for e in ro["worse"]] == ["k2", "k3"]
    assert ro["flat"]["label"] == "k1"
    ro = nbe.readout(entries([F(12, 100), F(10, 100)]))
    assert ro["flat"]["label"] == "k0", "a drop of exactly 0.02 is flat"
    ro = nbe.readout(entries([F(13, 100), F(10, 100)]))
    assert ro["flat"]["label"] == "k1"


def test_paired_bootstrap_matches_boot_ci_on_differences() -> None:
    ds = [decision(t, 42, {m: [0.5] for m in CANDS}) for t in ("a", "b", "c", "d")]
    va = {d.key: v for d, v in zip(ds, (0.1, 0.4, 0.2, 0.9), strict=True)}
    vb = {d.key: v for d, v in zip(ds, (0.0, 0.1, 0.3, 0.2), strict=True)}
    ci, empty = nbe.paired_boot(ds, lambda d: va[d.key], lambda d: vb[d.key])
    want = rc.boot_ci(ds, [va[d.key] - vb[d.key] for d in ds])
    assert empty == 0 and abs(ci[0] - want[0]) < 1e-12 and abs(ci[1] - want[1]) < 1e-12
    ci, empty = nbe.paired_boot(ds, lambda d: va[d.key] if d.cluster == "a" else None,
                                lambda d: vb[d.key])
    assert empty > 0 and ci is not None


def test_selective_value() -> None:
    ds = [decision(d.cluster, int(d.key.split("/")[2]),
                   {m: v + [v[-1]] * 7 for m, v in d.curves.items()},
                   tuple(d.outcome[c] for c in CANDS)) for d in three_epoch()]
    norm = F(10)
    assert nbe.selective_value(ds[0], 1, norm) is None
    assert nbe.selective_value(ds[1], 1, norm) == float(F(20) / norm)
    ents = nbe.selective_entries(ds, norm)
    assert [e["k"] for e in ents] == [0, 1, 2, 3, 5, 10], ents
    assert ents[0]["trusted"] == 2 and ents[1]["trusted"] == 1
    only = nbe.selective_entries(ds[:1], norm)
    assert [e["k"] for e in only] == [0], "budgets with no trusted decision are left out"


def main() -> None:
    tests = [test_extend_replaces_curves_only, test_extend_stops_on_a_wrong_run,
             test_references_reproduce_regret_compute, test_geometric_rows_cap_at_ten,
             test_readout_flat_monotone_u_and_tie,
             test_paired_bootstrap_matches_boot_ci_on_differences, test_selective_value]
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
