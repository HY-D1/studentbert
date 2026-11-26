from __future__ import annotations

import random
import sys
from fractions import Fraction as F

from analysis import prior_informed_halving as pih
from analysis import regret_compute as rc


def decision(bench: str, curves: dict, outcome: dict, hs: dict, key: str = "t/1",
             cluster: str = "t") -> rc.Decision:
    n = len(next(iter(curves.values())))
    costs = {c: [10.0 * (e + 1) for e in range(n)] for c in curves}
    return rc.Decision(bench, cluster, key, outcome, hs, curves, costs, 7.0,
                       lambda d, a, b, k: 0.0)


def random_decision(rng: random.Random, bench: str, n_cand: int, n_ep: int,
                    key: str) -> rc.Decision:
    cands = [f"m{i}" for i in range(n_cand)]
    curves = {c: [round(rng.random(), 1) for _ in range(n_ep)] for c in cands}
    return decision(bench, curves, {c: rng.random() for c in cands},
                    {c: rng.random() for c in cands}, key, key.split("/")[0])


def test_halving_reproduces_regret_compute() -> None:
    rng = random.Random(3)
    for bench, n_cand, n_ep in (("studentbert", 8, 20), ("quicktune", 6, 50), ("nlp", 7, 10)):
        ds = [random_decision(rng, bench, n_cand, n_ep, f"t{i}/1") for i in range(15)]
        pih.check_i0(bench, ds)


FOUR = {"a": [0.9, 0.9, 0.9], "b": [0.8, 0.95, 0.99], "c": [0.1, 0.2, 0.3], "d": [0.2, 0.1, 0.1]}
OUT = {"a": 70.0, "b": 80.0, "c": 50.0, "d": 40.0}


def test_variants_on_a_hand_example() -> None:
    hs = {"c": 4.0, "a": 3.0, "b": 2.0, "d": 1.0}
    d = decision("nlp", {k: v + [v[-1]] * 7 for k, v in FOUR.items()}, OUT, hs)
    assert d.order == ["c", "a", "b", "d"]
    assert pih.halving(d, [1, 2, 3], 2, d.order) == ("b", 60.0), "a, b survive; b leads at 2"
    c, s = pih.halving(d, [1, 2, 3], 2, d.order, protect="c")
    assert c == "b" and s == 90.0, "c kept at rung 1 makes 3 alive, so rung 3 is reached"
    c, _ = pih.halving(d, [1, 2, 3], 2, d.order, prior=True)
    assert c == "c", "rung 1 keeps a (0.5) and c (1.5, ties b, H-score first); rung 2 ties to c"
    r, s = pih.run_variant(d, "I1b", "rungs 1,2,3", 2, [])
    assert r == F(80) - F(50) and s == 7.0, "top ceil(4/4) = 1 by H-score, scoring cost only"
    r, s = pih.run_variant(d, "I0", "rungs 1,2,3", 2, [])
    assert r == 0 and s == 60.0 + 7.0


def test_uninformed_average() -> None:
    hs = {"c": 4.0, "a": 3.0, "b": 2.0, "d": 1.0}
    d = decision("nlp", {k: v + [v[-1]] * 7 for k, v in FOUR.items()}, OUT, hs)
    orders = pih.random_orders([d], 50)[d.key]
    assert len(orders) == 50 and all(sorted(o) == sorted(OUT) for o in orders)
    assert orders == pih.random_orders([d], 50)[d.key], "seeded"
    r, s = pih.run_variant(d, "U", "rungs 1,2,3", 2, orders)
    assert r == 0 and s == 60.0, "no ties, so every order gives b at 60 s and no scoring cost"


def test_readouts() -> None:
    def row(v: str, comp: float, reg: int) -> dict:
        return {"variant": v, "eta": 2, "param": "p", "compute_gpu_h": comp, "exact": F(reg),
                "per": [F(reg)] * 2}
    rows = [row("U", 1.0, 5), row("U", 2.0, 2), row("I0", 0.5, 6), row("I0", 1.2, 4),
            row("I1a", 0.4, 2), row("I2", 3.0, 9), row("I1b", 0.1, 1), row("I3", 0.2, 6)]
    got = {(m["u"]["compute_gpu_h"], m["variant"]): m["best"] for m in pih.matched_regret(rows)}
    assert got[(1.0, "I0")]["compute_gpu_h"] == 1.2 and got[(2.0, "I0")] is None
    assert got[(2.0, "I1a")]["compute_gpu_h"] == 0.4 and got[(1.0, "I2")] is None
    ds = [decision("nlp", {"a": [0.5], "b": [0.4]}, {"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 0.0},
                   f"t{i}/1", f"t{i}") for i in range(2)]
    mc = {m["row"]["compute_gpu_h"]: m for m in pih.matched_compute(ds, rows)}
    assert mc[1.2]["u"]["compute_gpu_h"] == 1.0 and mc[1.2]["diff"] == -1
    assert mc[3.0]["u"]["compute_gpu_h"] == 2.0 and mc[3.0]["diff"] == 7
    sr = pih.static_replaced(ds, rows)
    assert sr["static"] == 0 and sr["best"] is None


def test_resamples_are_boot_ci_draws() -> None:
    rng = random.Random(5)
    ds = [random_decision(rng, "nlp", 3, 2, f"t{i % 4}/{i}") for i in range(11)]
    vals = [rng.random() for _ in ds]
    means = []
    for cnt in pih.resamples(ds):
        xs = [v for d, v in zip(ds, vals, strict=True) for _ in range(cnt[d.cluster])]
        means.append(sum(xs) / len(xs))
    means.sort()
    lo, hi = rc.boot_ci(ds, vals)
    assert abs(means[int(0.025 * rc.BOOT)] - lo) < 1e-12
    assert abs(means[int(0.975 * rc.BOOT) - 1] - hi) < 1e-12


def test_boot_matched() -> None:
    ds = [decision("nlp", {"a": [0.5], "b": [0.4]}, {"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 0.0},
                   f"t{i}/1", f"t{i}") for i in range(2)]

    def row(v: str, per: list, secs: list) -> dict:
        return {"variant": v, "eta": 2, "param": f"p{v}", "per": [F(x) for x in per],
                "per_s": secs, "exact": F(sum(per), len(per)), "compute_gpu_h": sum(secs) / 7200}
    rows = [row("U", [2, 2], [3600.0, 3600.0]), row("I0", [1, 1], [1800.0, 1800.0]),
            row("I1a", [3, 3], [0.0, 0.0]), row("I2", [0, 4], [0.0, 0.0])]
    n = 200
    bm = pih.boot_matched(ds, rows, n)
    i0 = bm["pairs"][(2, "pU", "I0")]
    assert i0 == {"save_ci": (0.5, 0.5), "share_ci": (0.5, 0.5), "no_match": 0}
    i1 = bm["pairs"][(2, "pU", "I1a")]
    assert i1["no_match"] == n and i1["save_ci"] is None, "never at or below U's regret"
    both_t1 = sum(c == {"t1": 2} for c in pih.resamples(ds, n))
    i2 = bm["pairs"][(2, "pU", "I2")]
    assert 0 < both_t1 < n and i2["no_match"] == both_t1, "misses only when t1 is drawn twice"
    assert i2["save_ci"] == (1.0, 1.0)
    assert bm["pairs"][(2, "pU", "I3")]["no_match"] == n, "variant without configurations"
    assert bm["static_ci"] is None and bm["static_no_match"] == n, "static regret is 0"
    m, ci = pih.pair_diff(ds, rows[1], rows[0])
    assert m == -1 and ci == (-1.0, -1.0)


def main() -> None:
    tests = [test_halving_reproduces_regret_compute, test_variants_on_a_hand_example,
             test_uninformed_average, test_readouts, test_resamples_are_boot_ci_draws,
             test_boot_matched]
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
