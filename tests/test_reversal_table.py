from __future__ import annotations

"""Tests for analysis/reversal_table.py on synthetic inputs (no cluster files needed).

  PYTHONPATH=. python tests/test_reversal_table.py
"""

import copy
import csv
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import reversal_table as rt  # noqa: E402

CANDS = ["scratch"] + [f"src:{t}" for t in rt.TARGETS]
FT_SEEDS = (1, 2, 3, 4, 5, 42)


def synthetic(seed: int = 0) -> tuple:
    rng = random.Random(seed)
    gold = {t: {c: {s: 0.7 + 0.002 * i + rng.uniform(-0.0003, 0.0003) for s in FT_SEEDS}
                for i, c in enumerate(CANDS)} for t in rt.TARGETS}
    signs = {t: {c: ("positive" if i % 2 else "none") for i, c in enumerate(CANDS)}
             for t in rt.TARGETS}
    hs = {t: {s: {c: {L: 0.05 + 0.001 * (len(CANDS) - i) + 0.0001 * L for L in range(7)}
                  for i, c in enumerate(CANDS)} for s in rt.EST_SEEDS} for t in rt.TARGETS}
    lm = {t: {s: {c: -0.5 + 0.001 * i for i, c in enumerate(CANDS)} for s in rt.EST_SEEDS}
          for t in rt.TARGETS}
    reset = {t: {s: {f"src:{t}": hs[t][s][f"src:{t}"][6] - 0.01} for s in rt.EST_SEEDS}
             for t in rt.TARGETS}
    val = {t: {c: {s: {e: 0.6 + 0.001 * i + 0.01 * e for e in rt.EARLY} for s in FT_SEEDS}
               for i, c in enumerate(CANDS)} for t in rt.TARGETS}
    return gold, signs, hs, lm, reset, val


def idx() -> list:
    r = random.Random(0)
    return [[r.randrange(6) for _ in range(6)] for _ in range(2000)]


def test_label_rule_stable_reversal_and_both_kinds_of_tie():
    g = {"t": {"a": {s: 0.80 for s in FT_SEEDS}, "b": {s: 0.79 for s in FT_SEEDS},
               "c": {s: 0.7995 for s in FT_SEEDS},
               "d": {s: 0.80 + (0.02 if s % 2 else -0.02) for s in FT_SEEDS}}}
    assert rt.label(g, "t", "a", "b", idx())["label_main"] == "stable"
    assert rt.label(g, "t", "b", "a", idx())["label_main"] == "reversal"
    assert rt.label(g, "t", "a", "c", idx())["label_main"] == "near_tie"  # within the 0.001 margin
    noisy = rt.label(g, "t", "d", "b", idx())
    assert noisy["gold_ci_lo"] <= 0 <= noisy["gold_ci_hi"] and noisy["label_main"] == "near_tie"
    rev = rt.label(g, "t", "b", "a", idx())
    assert abs(rev["pair_regret"] - 0.01) < 1e-12 and rev["gold_diff"] < 0


def test_only_the_algebra2006_junyi_pair_is_ambiguous():
    g = {"algebra2006": {"src:algebra2006": {s: 0.78 for s in FT_SEEDS},
                         "src:junyi": {s: 0.79 for s in FT_SEEDS},
                         "src:ednet": {s: 0.77 for s in FT_SEEDS}}}
    amb = rt.label(g, "algebra2006", "src:algebra2006", "src:junyi", idx())
    assert amb["label_main"] == "reversal" and amb["label_robust"] == "ambiguous"
    other = rt.label(g, "algebra2006", "src:ednet", "src:junyi", idx())
    assert other["label_robust"] == other["label_main"] == "reversal"


def test_history_never_reads_the_held_out_target_or_the_candidates_own_dataset():
    gold, signs, *_ = synthetic()
    base = rt.history(gold, signs, "assist2017")
    g2 = copy.deepcopy(gold)
    for c in CANDS:
        for s in FT_SEEDS:
            g2["assist2017"][c][s] = random.Random(c + str(s)).uniform(0.5, 0.9)
    assert rt.history(g2, signs, "assist2017") == base
    g3 = copy.deepcopy(gold)
    g3["junyi"]["src:junyi"] = {s: 0.99 for s in FT_SEEDS}
    assert rt.history(g3, signs, "assist2017")["src:junyi"] == base["src:junyi"]
    assert all(base[c]["n_targets"] == (6 if c == "scratch" else 5) for c in base
               if c != "src:assist2017")
    assert base["src:assist2017"]["n_targets"] == 6


def test_pairs_are_oriented_by_hscore_and_complete():
    rows, targets = rt.build(*synthetic())
    assert len(rows) == 7 * 3 * 28 and len(targets) == 21
    assert all(r["frozen_rank_a"] < r["frozen_rank_b"] and r["hscore_a"] >= r["hscore_b"]
               for r in rows)
    keys = {(r["target"], r["est_seed"], frozenset((r["cand_a"], r["cand_b"]))) for r in rows}
    assert len(keys) == len(rows)


def test_target_regret_follows_the_hscore_top_pick():
    gold, signs, hs, lm, reset, val = synthetic()
    _, targets = rt.build(gold, signs, hs, lm, reset, val)
    gm = rt.gold_means(gold)
    for r in targets:
        t = r["target"]
        src = [c for c in CANDS if c != "scratch"]
        pick = max(src, key=lambda c: (hs[t][r["est_seed"]][c][6], c))
        assert r["frozen_pick_pretrained"] == pick
        assert abs(r["target_regret_pretrained"]
                   - (max(gm[t][c] for c in src) - gm[t][pick])) < 1e-12


def test_early_features_never_read_the_test_column():
    out = []
    for test_value in ("0.70", "0.99"):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "early.tsv"
            with p.open("w", newline="") as fh:
                w = csv.writer(fh, delimiter="\t", lineterminator="\n")
                w.writerow(["target", "candidate", "finetune_seed", "epoch", "val_auc",
                            "best_val_auc", "best_val_source", "test_auc_reference", "log_file"])
                for t in rt.TARGETS:
                    for c in CANDS:
                        for s in FT_SEEDS:
                            for e in range(1, 21):
                                w.writerow([t, c, s, e, 0.6 + e / 1000, 0.7, "x", test_value, "l"])
            out.append(rt.load_early(str(p)))
    assert out[0] == out[1]


def test_missing_inputs_abort():
    gold, signs, hs, lm, reset, val = synthetic()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "early.tsv"
        p.write_text("target\tcandidate\tfinetune_seed\tepoch\tval_auc\tbest_val_auc\t"
                     "best_val_source\ttest_auc_reference\tlog_file\n")
        try:
            rt.load_early(str(p))
        except SystemExit as exc:
            assert "ABORT" in str(exc)
        else:
            raise AssertionError("an empty early-epoch file did not abort")


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
