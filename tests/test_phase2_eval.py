from __future__ import annotations

"""Tests for analysis/phase2_eval.py on synthetic early fine-tunes (no cluster files needed).

  PYTHONPATH=. python tests/test_phase2_eval.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import phase2_eval as ev  # noqa: E402
from analysis import task4_evaluate as te  # noqa: E402

GOLD = te.load_gold()
MODELS = sorted(GOLD[("rte", "cls")])
PICK, OTHER = MODELS[0], MODELS[1]


def pairs(task="rte", pooling="cls", seed=3):
    return [{"task": task, "pooling": pooling, "seed": str(seed), "pick": PICK, "alternative": m,
             "hscore_pick": "1.0", "hscore_alt": str(0.9 - 0.01 * j), "logme_pick": "0.1",
             "logme_alt": "0.5" if m == OTHER else "0.0", "label": "stable"}
            for j, m in enumerate(MODELS[1:])]


def runs(acc_pick: float, acc_other: float, n_val: int = 2000):
    out = {}
    for t in ev.TASKS:
        for p in ev.POOLINGS:
            for s in ev.SEEDS:
                for m in MODELS:
                    a = acc_pick if m == PICK else (acc_other if m == OTHER else 0.5)
                    out[(t, p, s, m)] = {"epochs": [{"epoch": e, "val_acc": a, "epoch_s": 60.0}
                                                    for e in ev.EPOCHS], "n_val": n_val}
    return out


def test_the_full_framework_switches_only_beyond_the_noise_margin():
    d = ev.delta(0.80, 0.80, 2000)
    assert abs(d - 1.96 * (2 * 0.8 * 0.2 / 2000) ** 0.5) < 1e-12 and 0.024 < d < 0.025
    small = ev.choices(pairs(), runs(0.80, 0.80 + d / 2), GOLD)[0]
    assert small["early_1"] == OTHER and small["full_1"] == PICK and small["abstain_1"] == ""
    big = ev.choices(pairs(), runs(0.80, 0.80 + 2 * d), GOLD)[0]
    assert big["full_1"] == OTHER


def test_ties_for_the_leader_go_to_h_score_and_logme_uses_its_own_scores():
    r = ev.choices(pairs(), runs(0.80, 0.80), GOLD)[0]
    assert r["early_1"] == PICK and r["full_1"] == PICK and r["abstain_1"] == PICK
    assert r["logme"] == OTHER


def test_false_corrections_and_harmless_switches_follow_the_tie_rule():
    perf = {m: GOLD[("rte", "cls")][m]["tuned"] for m in MODELS}
    a, b = MODELS[0], MODELS[1]
    o = ev.outcome(GOLD, "rte", "cls", b, a)
    expect = ("tie" if te.tied(perf[b], perf[a]) else
              ("better" if perf[b][0] > perf[a][0] else "worse"))
    assert o["vs_pick"] == expect
    assert ev.outcome(GOLD, "rte", "cls", a, a)["vs_pick"] == "same"


def test_abstention_counts_coverage_and_calibration():
    rows = [{"task": "rte", "pooling": "cls", "seed": s, "hscore": PICK,
             "abstain_1": PICK if s == 3 else ""} for s in ev.SEEDS]
    recs = {("rte", "cls", s): {"reversed": "1" if s == 5 else "0"} for s in ev.SEEDS}
    s = ev.summarize(rows, recs, GOLD, "abstain_1")
    assert (s["followed"], s["flag_n"], s["flag_rev"], s["fol_rev"]) == (1, 2, 1, 0)


def test_head_files_must_be_the_fresh_report():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "x_report.md").write_text("not the report\n")
        try:
            ev.load_head(str(Path(d) / "x"))
        except SystemExit as exc:
            assert "ABORT" in str(exc)
        else:
            raise AssertionError("a wrong head-pair report was accepted")


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
