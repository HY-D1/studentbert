from __future__ import annotations

"""Tests for analysis/nlp_option1_supplement.py (no cluster files or datasets package needed).

  PYTHONPATH=. python tests/test_nlp_option1_supplement.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import nlp_option1_supplement as sup  # noqa: E402


def test_wilson_matches_known_values():
    lo, hi = sup.wilson(18, 20)
    assert abs(lo - 0.6990) < 5e-4 and abs(hi - 0.9721) < 5e-4
    assert sup.wilson(0, 10)[0] == 0.0 and abs(sup.wilson(10, 10)[1] - 1.0) < 1e-12


def recs():
    out = []
    for i, t in enumerate(sup.TASKS):
        for j in range(6):
            out.append({"target": t, "est_seed": f"cls:{j}", "regret": str(float(j % 2)),
                        "reversed": "1" if j % 2 else "0",
                        "risk_noisy_or": "0.9" if j < 3 else "0.1",
                        "tau_fold": "0.5"})
    return out


def test_summary_counts_and_calibration():
    s = sup.summarize(recs(), lambda r: float(r["risk_noisy_or"]) > 0.5)
    assert (s["caught"], s["n_rev"], s["fa"], s["n_ok"], s["followed"]) == (5, 15, 10, 15, 15)
    assert s["rev_among_flagged"] == (5, 15) and s["rev_among_followed"] == (10, 15)


def test_cluster_bootstrap_is_deterministic_and_brackets_the_estimate():
    ab = lambda r: float(r["risk_noisy_or"]) > 0.5  # noqa: E731
    a = sup.cluster_boot(recs(), ab, boots=500)
    assert a == sup.cluster_boot(recs(), ab, boots=500)
    s = sup.summarize(recs(), ab)
    assert a[0] <= s["sel"] <= a[1]


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
