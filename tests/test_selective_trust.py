from __future__ import annotations

"""Tests for analysis/selective_trust.py on hand-built decisions: the normalized lead, coverage and
selective regret over thresholds, t = 0 against the 12.27 warning rule, the normalizer, the area,
the leave-one-domain-out choice (by decision and with equal domain weight), the budget rule, the
nearest-compute reference and rounding. Run from the code root:

  PYTHONPATH=. python tests/test_selective_trust.py
"""

import sys
from fractions import Fraction as F

from analysis import regret_compute as rc
from analysis import selective_trust as stt


def make(curve: tuple[float, float, float], outcome: tuple[float, float, float],
         cluster: str = "t", on_axis: bool = True) -> rc.Decision:
    """Three candidates; "p" is H-score's choice. curve: epoch-1 validation scores of p, a, b."""
    names = ("p", "a", "b")
    curves = {c: [v] for c, v in zip(names, curve, strict=True)}
    costs = {c: [10.0] for c in names}
    return rc.Decision("x", cluster, f"{cluster}/1", dict(zip(names, outcome, strict=True)),
                       {"p": 3.0, "a": 2.0, "b": 1.0}, curves, costs, 5.0,
                       lambda d, a, b, k: 0.0, static_on_axis=on_axis)


GOOD = (1.0, 0.0, 0.0)   # p is the best candidate: regret 0; random candidate 2/3
BAD = (0.0, 1.0, 1.0)    # p is the worst: regret 1; random candidate 1/3


def test_normalized_lead() -> None:
    assert stt.lead(make((0.75, 0.5, 0.25), GOOD), 1) == F(1, 2)
    assert stt.lead(make((0.5, 0.75, 0.25), GOOD), 1) == F(-1, 2)
    assert stt.lead(make((1.0, 0.0, 0.0), GOOD), 1) == 1
    assert stt.lead(make((0.5, 0.5, 0.5), GOOD), 1) == 0, "zero spread gives 0"
    tie = make((0.75, 0.75, 0.25), GOOD)
    assert stt.lead(tie, 1) == 0 and stt.is_trusted(tie, 1, F(0)), "a tie is trusted at t = 0"


def fixture() -> list[rc.Decision]:
    # leads 1, 1/2, 0, -1/2; regrets 0, 0, 1, 1
    return [make((1.0, 0.0, 0.0), GOOD, "t1"), make((0.75, 0.5, 0.25), GOOD, "t2"),
            make((0.75, 0.75, 0.25), BAD, "t3"), make((0.5, 0.75, 0.25), BAD, "t4")]


def test_coverage_and_selective_regret() -> None:
    ds = fixture()
    norm = stt.normalizer(ds)
    assert norm == F(1, 2), norm
    want = {F(-1): (4, F(1, 2)), F(-1, 2): (4, F(1, 2)), F(-9, 20): (3, F(1, 3)),
            F(0): (3, F(1, 3)), F(1, 20): (2, F(0)), F(1, 2): (2, F(0)), F(11, 20): (1, F(0)),
            F(1): (1, F(0))}
    for t, (n, reg) in want.items():
        p = stt.point(ds, 1, t, norm)
        assert (p["trusted"], p["regret"]) == (n, reg), (t, p)
        assert p["coverage"] == F(n, 4) and p["norm_regret"] == reg / norm
    p0 = stt.point(ds, 0, None, norm)
    assert p0["trusted"] == 4 and p0["regret"] == F(1, 2), "k = 0 trusts every decision"


def test_t0_equals_abstention() -> None:
    for d in fixture():
        assert stt.is_trusted(d, 1, F(0)) == (rc.abstain(d, 1)[0] is not None)


def test_compute_and_full_share() -> None:
    on, off = make((1.0, 0.0, 0.0), GOOD), make((1.0, 0.0, 0.0), GOOD, on_axis=False)
    assert stt.compute_gpu_h([on], 1) == 35.0 / 3600
    assert stt.compute_gpu_h([off], 1) == 30.0 / 3600, "Quick-Tune scoring is off the axis"
    assert stt.compute_gpu_h([on], 0) == 5.0 / 3600 and stt.compute_gpu_h([off], 0) == 0.0
    assert stt.full_gpu_h([on]) == 30.0 / 3600


def test_area() -> None:
    pts = [{"trusted": 0, "coverage": F(0), "norm_regret": None},
           {"trusted": 1, "coverage": F(1, 4), "norm_regret": F(0)},
           {"trusted": 2, "coverage": F(1, 2), "norm_regret": F(1, 2)},
           {"trusted": 2, "coverage": F(1, 2), "norm_regret": F(1, 2)},
           {"trusted": 4, "coverage": F(1), "norm_regret": F(1)}]
    assert stt.area(pts) == (F(7, 16), F(1, 4))
    assert stt.area(pts[:1]) == (None, None)
    ds = fixture()
    a, cmin = stt.area([stt.point(ds, 1, t, F(1, 2)) for t in stt.THRESHOLDS])
    # points (1/4, 0), (1/2, 0), (3/4, 2/3), (1, 1)
    assert (a, cmin) == (F(1, 12) + F(5, 24), F(1, 4)), (a, cmin)


def test_own_best_threshold_and_tie_band() -> None:
    ds = [make((1.0, 0.0, 0.0), GOOD), make((0.75, 0.5, 0.25), GOOD),
          make((0.5, 0.75, 0.25), BAD)]
    c = stt.choose([(ds, stt.normalizer(ds))], 1)
    assert (c["t"], c["t_low"], c["coverage"], c["norm_regret"]) == (F(1, 2), F(-9, 20), F(2, 3),
                                                                     F(0)), c


def test_none_when_no_threshold_qualifies() -> None:
    ds = [make((0.75, 0.5, 0.25), BAD)]
    assert stt.choose([(ds, stt.normalizer(ds))], 1) is None
    assert stt.choose([(ds, stt.normalizer(ds))], 1, "domain") is None


def big_and_small() -> tuple[list, list]:
    big = [make((0.75, 0.5, 0.25), GOOD, f"b{i}") for i in range(10)]
    # s1: lead 1/2, regret 0; s2: lead 1/4, regret 1/2 (normalized 6/7)
    small = [make((1.0, 0.5, 0.0), GOOD, "s1"), make((1.0, 0.75, 0.0), (0.5, 1.0, 0.0), "s2")]
    return big, small


def test_equal_domain_weight_differs_from_pooling_by_decision() -> None:
    big, small = big_and_small()
    srcs = [(big, stt.normalizer(big)), (small, stt.normalizer(small))]
    assert stt.normalizer(small) == F(7, 12)
    prim = stt.choose(srcs, 1, "decision")
    # by decision s2's 6/7 is diluted over 12 decisions (1/14 <= 0.10), so coverage 1 qualifies
    assert (prim["t"], prim["t_low"], prim["coverage"]) == (F(1, 4), F(-1), F(1)), prim
    sens = stt.choose(srcs, 1, "domain")
    # equal weight: (0 + 3/7) / 2 > 0.10 at t <= 1/4, so s2 is left out; mean coverage 3/4
    assert (sens["t"], sens["t_low"], sens["coverage"]) == (F(1, 2), F(3, 10), F(3, 4)), sens


def test_equal_domain_weight_needs_coverage_in_both_sources() -> None:
    big, _ = big_and_small()
    behind = [make((0.5, 1.0, 0.0), BAD, f"w{i}") for i in range(2)]  # lead -1/2, regret 1
    srcs = [(big, stt.normalizer(big)), (behind, stt.normalizer(behind))]
    prim = stt.choose(srcs, 1, "decision")
    assert (prim["t"], prim["t_low"], prim["coverage"]) == (F(1, 2), F(-9, 20), F(10, 12)), prim
    assert stt.choose(srcs, 1, "domain") is None


def test_budget_rule() -> None:
    good = [make((1.0, 0.0, 0.0), GOOD, f"g{i}") for i in range(2)]
    assert stt.budget_rule(good, stt.normalizer(good), (1,))["k"] == 0
    mixed = [make((0.75, 0.5, 0.25), GOOD, "m1"), make((0.5, 0.75, 0.25), BAD, "m2")]
    r = stt.budget_rule(mixed, stt.normalizer(mixed), (1,))
    assert (r["k"], r["t"], r["t_low"]) == (1, F(1, 2), F(-9, 20)), r
    low = mixed + [make((0.5, 0.75, 0.25), BAD, "m3")]
    assert stt.budget_rule(low, stt.normalizer(low), (1,)) is None, "coverage 1/3 < 0.50"


def test_nearest_reference() -> None:
    rows = [{"method": "early leader", "compute_gpu_h": c, "param": str(c)} for c in (0.25, 0.75)]
    rows.append({"method": "successive halving", "compute_gpu_h": 0.5, "param": "sh"})
    assert stt.nearest(rows, "early leader", 0.5)["compute_gpu_h"] == 0.25, "tie to lower"
    assert stt.nearest(rows, "early leader", 0.625)["compute_gpu_h"] == 0.75
    assert stt.nearest(rows, "successive halving", 0.0)["param"] == "sh"


def test_rounding_half_away_from_zero() -> None:
    assert stt.dec(F(1, 8), 2) == "0.13" and stt.dec(F(-1, 8), 2) == "-0.13"
    assert stt.dec(0.125, 2) == "0.13" and stt.dec(None, 2) == "n/a"
    assert stt.dec(float("nan"), 2) == "nan" and stt.dec(F(0), 2) == "0.00"


def main() -> None:
    tests = [test_normalized_lead, test_coverage_and_selective_regret, test_t0_equals_abstention,
             test_compute_and_full_share, test_area, test_own_best_threshold_and_tie_band,
             test_none_when_no_threshold_qualifies,
             test_equal_domain_weight_differs_from_pooling_by_decision,
             test_equal_domain_weight_needs_coverage_in_both_sources, test_budget_rule,
             test_nearest_reference, test_rounding_half_away_from_zero]
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
