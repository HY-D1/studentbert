#!/usr/bin/env python3
"""Paired-by-seed statistics for the pretraining-objective contrasts, all 7 targets.

Fills the gap where only Algebra 2006 had a numeric interval and ASSISTments 2017
and Junyi were recorded as "CI excludes 0" with no numbers. Needs no GPU and no
reruns: it reads the per-seed test AUCs already sitting in the ablation logs.

INPUT
    objabl_perseed.csv, HEADERLESS, four columns:
        log_prefix, objective, seed, test_auc
    Regenerate from the repo root with:
        for P in w7_objabl w7_objabl2 w8_regime_ednet w8_algabl w8_bridgeabl \
                 w8_a09abl w8_alg06abl; do
          for O in full skill_only correct_only; do
            for S in 42 1 2 3 4 5; do
              V=$(grep -h "test AUC" ${P}_${O}_s${S}_*.log 2>/dev/null | head -1 \
                  | awk '{print $NF}')
              echo "${P},${O},${S},${V}"
            done
          done
        done > objabl_perseed.csv

WHAT IS PAIRED, AND WHY IT IS VALID
    Runs are paired by seed. The seed fixes the initialisation AND, for the three
    targets fine-tuned with --n_students 1000 (assist2017, junyi, ednet), the
    target subsample drawn by first_n_students. Two objectives at the same seed
    therefore see the same target data, so the per-seed difference removes both
    the subsample draw and the initialisation draw. The remaining four targets use
    every available student, so the seed controls initialisation only.

WHAT IS REPORTED
    Per-seed differences, the mean, how many of the 6 seeds are positive, a
    percentile bootstrap CI resampling seeds with replacement, and the EXACT
    one-sided sign test. Read the n=6 caveat the script prints. A bootstrap over
    6 paired values is coarse and the CI endpoints can only land on a small set of
    values; the sign test and the raw per-seed spread are the honest companions to
    it, which is why all three are printed.

USAGE (from the repo root)
    python3 analysis/paired_bootstrap_objective.py
    python3 analysis/paired_bootstrap_objective.py --boot 50000 --seed 7
    python3 analysis/paired_bootstrap_objective.py --out objective_ci.md
"""
import argparse
import csv
import math
import random

# log prefix -> dataset name as used in RESULTS.md
PREFIX = [
    ("w7_objabl", "assist2017"),
    ("w8_regime_ednet", "ednet"),
    ("w7_objabl2", "junyi"),
    ("w8_algabl", "algebra2005"),
    ("w8_bridgeabl", "bridge2006"),
    ("w8_a09abl", "assist2009"),
    ("w8_alg06abl", "algebra2006"),
]

# (label, minuend, subtrahend)
CONTRASTS = [
    ("skill - correct", "skill_only", "correct_only"),
    ("full - correct", "full", "correct_only"),
    ("full - skill", "full", "skill_only"),
]

SEEDS = ["42", "1", "2", "3", "4", "5"]


def load(path):
    """-> {(prefix, objective, seed): value}. Tolerates a stray header row."""
    out = {}
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) != 4:
                continue
            prefix, obj, seed, val = (c.strip() for c in row)
            try:
                out[(prefix, obj, seed)] = float(val)
            except ValueError:
                continue
    return out


def percentile(sorted_vals, q):
    """Linear-interpolated percentile, q in [0, 1]."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def boot_ci(diffs, n_boot, rng, alpha=0.05):
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    return percentile(means, alpha / 2), percentile(means, 1 - alpha / 2)


def sign_test_one_sided(n_pos, n):
    """P(X >= n_pos) under X ~ Binomial(n, 0.5). Exact, no approximation."""
    tail = sum(math.comb(n, i) for i in range(n_pos, n + 1))
    return tail / (2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="objabl_perseed.csv")
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the bootstrap; output is deterministic")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = load(args.csv)
    rng = random.Random(args.seed)

    lines = []
    W = lines.append

    W("### Paired-by-seed objective contrasts "
      "(analysis/paired_bootstrap_objective.py, objabl_perseed.csv)")
    W("")
    W(f"Test AUC differences paired by seed, 6 seeds (42, 1, 2, 3, 4, 5). "
      f"Percentile bootstrap over seeds, {args.boot:,} resamples, rng seed "
      f"{args.seed}. Sign test is exact and one-sided.")
    W("")
    W("| Target | Contrast | Mean | 95% CI | Positive | Sign p (1-sided, "
      "observed dir) | Min | Max |")
    W("|---|---|---|---|---|---|---|---|")

    detail = []
    missing = []
    for prefix, name in PREFIX:
        for label, a, b in CONTRASTS:
            diffs = []
            per_seed = []
            for s in SEEDS:
                va = data.get((prefix, a, s))
                vb = data.get((prefix, b, s))
                if va is None or vb is None:
                    missing.append(f"{name} {label} seed {s}")
                    continue
                d = va - vb
                diffs.append(d)
                per_seed.append((s, d))
            if not diffs:
                W(f"| {name} | {label} | NOT FOUND | - | - | - | - | - |")
                continue
            n = len(diffs)
            mean = sum(diffs) / n
            n_pos = sum(1 for d in diffs if d > 0)
            lo, hi = boot_ci(diffs, args.boot, rng)
            p = sign_test_one_sided(max(n_pos, n - n_pos), n)
            W(f"| {name} | {label} | {mean:+.4f} | "
              f"[{lo:+.4f}, {hi:+.4f}] | {n_pos}/{n} | {p:.4f} | "
              f"{min(diffs):+.4f} | {max(diffs):+.4f} |")
            detail.append((name, label, per_seed))

    W("")
    W("**Per-seed differences.**")
    W("")
    W("| Target | Contrast | " + " | ".join("s" + s for s in SEEDS) + " |")
    W("|---|---|" + "---|" * len(SEEDS))
    for name, label, per_seed in detail:
        cells = {s: d for s, d in per_seed}
        row = " | ".join(f"{cells[s]:+.4f}" if s in cells else "-" for s in SEEDS)
        W(f"| {name} | {label} | {row} |")
    W("")

    if missing:
        W("**Missing cells:** " + ", ".join(missing))
        W("")

    W("_Read: differences are paired by seed, so each value removes the "
      "initialisation draw and, for the three targets fine-tuned at 1000 train "
      "students, the target subsample draw as well. With 6 paired values a "
      "percentile bootstrap is coarse, its endpoints can only take a limited set "
      "of values, and it should be read alongside the exact sign test and the "
      "per-seed minimum and maximum rather than on its own. The sign test is "
      "one-sided in the direction the data actually went; for the skill-correct "
      "contrast that direction was specified in advance by the regime label, so "
      "it reads as a predicted-direction test there, while for full-skill no "
      "direction was specified and the p value is therefore post hoc. With 6 "
      "seeds the smallest attainable one-sided p is 1/64 = 0.0156, so a 6/6 or "
      "0/6 split is the strongest sign-test evidence this design can produce._")

    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
