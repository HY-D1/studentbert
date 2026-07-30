#!/usr/bin/env python3
"""Paired-by-seed statistics for the next-skill N-sweep gaps.

This is the analysis listed as "optional LAK paired-bootstrap CIs" and never
written. It needs no GPU and no reruns: it reads the per-seed long CSV that
analysis/parse_nextskill_full.py already produces.

Input : nextskill_results_long.csv  (HEADERLESS, written by parse_nextskill_full.py)
        columns: dataset,cond,N,seed,metric,value
Output: a markdown block per dataset/metric, ready to paste into RESULTS.md.

Every pretrained condition is compared to scratch at the same N, paired by seed
(the same seed means the same target subsample, since first_n_students draws per
seed). Reports the per-seed gaps, the mean, how many seeds are positive, a
paired bootstrap CI over seeds, and the EXACT one-sided sign test.

Read the caveat this prints about n=3. It matters for what LAK can claim.

Usage from the repo root:
    python3 analysis/paired_bootstrap_lak.py
    python3 analysis/paired_bootstrap_lak.py --csv nextskill_results_long.csv \
        --dataset assist2017 --metric top1 --boot 20000
"""
import argparse
import csv
import math
import os
import random
from collections import defaultdict


def load(path):
    """Long CSV is headerless: dataset,cond,N,seed,metric,value."""
    rows = {}
    with open(path, newline="") as f:
        for i, r in enumerate(csv.reader(f), 1):
            if len(r) != 6:
                continue
            ds, cond, N, seed, metric, val = r
            if not N.strip().lstrip("-").isdigit():      # tolerate a stray header
                continue
            try:
                rows[(ds.strip(), cond.strip(), int(N), int(seed), metric.strip())] = float(val)
            except ValueError:
                continue
    if not rows:
        raise SystemExit(f"no usable rows parsed from {path}; expected 6 columns "
                         "dataset,cond,N,seed,metric,value")
    return rows


def sign_test_one_sided(k, n):
    """P(X >= k) under X~Binomial(n, 0.5). Exact, no approximation."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def boot_ci(gaps, B, alpha=0.05, seed=0):
    """Percentile CI from resampling the paired per-seed gaps with replacement."""
    rng = random.Random(seed)
    n = len(gaps)
    means = []
    for _ in range(B):
        means.append(sum(gaps[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * B)]
    hi = means[min(B - 1, int((1 - alpha / 2) * B))]
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="nextskill_results_long.csv")
    ap.add_argument("--dataset", default=None, help="default: every dataset present")
    ap.add_argument("--metric", default="top1", help="top1 | top5 | macro_auc | weighted_auc | macro_top1")
    ap.add_argument("--baseline", default="scratch")
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--all-metrics", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"{args.csv} not found. Regenerate it first:\n"
                         "  python analysis/parse_nextskill_full.py --dir .")
    rows = load(args.csv)

    datasets = sorted({k[0] for k in rows}) if args.dataset is None else [args.dataset]
    metrics = sorted({k[4] for k in rows}) if args.all_metrics else [args.metric]

    for ds in datasets:
        for metric in metrics:
            sub = {k: v for k, v in rows.items() if k[0] == ds and k[4] == metric}
            if not sub:
                continue
            Ns = sorted({k[2] for k in sub})
            conds = sorted({k[1] for k in sub} - {args.baseline})
            if not conds:
                continue

            print(f"\n#### {ds}, {metric}, paired vs {args.baseline} "
                  f"(bootstrap B={args.boot}, seeds paired)\n")
            print("| N | condition | seeds | per-seed gaps | mean gap | 95% CI (paired boot) | +/n | sign-test p |")
            print("|---|---|---|---|---|---|---|---|")

            min_n = 99
            for N in Ns:
                for cond in conds:
                    seeds = sorted({k[3] for k in sub if k[1] == cond and k[2] == N}
                                   & {k[3] for k in sub if k[1] == args.baseline and k[2] == N})
                    if not seeds:
                        continue
                    gaps = [sub[(ds, cond, N, s, metric)] - sub[(ds, args.baseline, N, s, metric)]
                            for s in seeds]
                    n = len(gaps)
                    min_n = min(min_n, n)
                    mean = sum(gaps) / n
                    lo, hi = boot_ci(gaps, args.boot)
                    pos = sum(g > 0 for g in gaps)
                    p = sign_test_one_sided(pos, n)
                    excl = "excludes 0" if (lo > 0 or hi < 0) else "includes 0"
                    per_seed = " ".join(f"{g:+.4f}" for g in gaps)
                    print(f"| {N} | {cond} | {n} | {per_seed} | {mean:+.4f} | "
                          f"[{lo:+.4f}, {hi:+.4f}] {excl} | {pos}/{n} | {p:.3f} |")

            print(f"\n_Read: gaps are pretrained minus {args.baseline} at the same N, paired by seed. "
                  "The bootstrap resamples the paired per-seed gaps._\n")

            if min_n <= 4:
                need = next(k for k in range(1, 40) if 0.5 ** k < 0.05)
                print(f"> CAVEAT, n={min_n} seeds per cell. Two consequences, both real:")
                print(f"> 1. The exact one-sided sign test cannot go below "
                      f"{0.5 ** min_n:.3f} even when every seed agrees, so no cell here can reach "
                      f"p<0.05 by that test. Reaching p<0.05 needs at least {need} seeds.")
                print(f"> 2. A bootstrap over {min_n} values resamples from only "
                      f"{math.comb(2 * min_n - 1, min_n)} distinct multisets, so the CI is coarse "
                      "and anti-conservative. Treat it as indicative, not as a significance test.")
                print("> The defensible claim at this seed count is the per-seed evidence itself: "
                      "the gap direction and how many seeds agree. That is what the poster reports.")


if __name__ == "__main__":
    main()
