#!/usr/bin/env python3
# Paired-by-seed bootstrap over the inventory produced by inventory_kt_auc.py.
#
# Why: RESULTS.md carries several confidence intervals marked "recorded", with
# no resample count and no per-seed values, so they cannot be reproduced. This
# recomputes any of them from the raw per-run numbers, which means every
# interval that enters the paper has a command behind it.
#
# Pairs two conditions on the seeds they have in common, never on a difference
# of independently averaged means, and reports the per-seed direction count
# alongside the interval so the sign-test claim and the CI claim agree.
#
# Usage, from the repo root on the cluster:
#   python3 analysis/inventory_kt_auc.py --logdir . --logdir ../logs --out kt_inventory.tsv
#   python3 analysis/paired_bootstrap_pair.py --tsv kt_inventory.tsv \
#       --a edubert_ednet_ktfull_ednet_indomain_n20000 \
#       --b edubert_ednet_ktfull_ednet_scratch_n20000
#
# Run names are matched by stem, with the trailing _seed<N> removed, so the
# seed4 / seed42 prefix collision cannot occur.

from __future__ import annotations

import argparse
import random
import re
import statistics as st
import sys
from collections import defaultdict

SEED_SUFFIX = re.compile(r"_seed[0-9]+$")


def load(tsv):
    rows = []
    with open(tsv, encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}
        for name in ("run", "seed", "auc"):
            if name not in idx:
                sys.exit("column %r missing from %s" % (name, tsv))
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            run = parts[idx["run"]]
            seed = parts[idx["seed"]]
            if not run or not seed:
                continue
            rows.append((SEED_SUFFIX.sub("", run), int(seed), float(parts[idx["auc"]])))
    return rows


def by_seed(rows, stem):
    out = {}
    for r_stem, seed, auc in rows:
        if r_stem == stem:
            if seed in out and abs(out[seed] - auc) > 1e-9:
                sys.exit("stem %s seed %d has two different values, %.4f and %.4f; "
                         "resolve the duplicate before bootstrapping"
                         % (stem, seed, out[seed], auc))
            out[seed] = auc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="kt_inventory.tsv")
    ap.add_argument("--a", required=True, help="run stem, without _seed<N>")
    ap.add_argument("--b", required=True, help="baseline run stem, without _seed<N>")
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load(args.tsv)
    A, B = by_seed(rows, args.a), by_seed(rows, args.b)
    if not A:
        sys.exit("no rows for stem %s" % args.a)
    if not B:
        sys.exit("no rows for stem %s" % args.b)
    shared = sorted(set(A) & set(B))
    if not shared:
        sys.exit("no shared seeds: %s has %s, %s has %s"
                 % (args.a, sorted(A), args.b, sorted(B)))
    only_a = sorted(set(A) - set(B))
    only_b = sorted(set(B) - set(A))
    if only_a or only_b:
        print("WARNING dropping unpaired seeds: %s only in a, %s only in b"
              % (only_a, only_b))

    gaps = [A[s] - B[s] for s in shared]
    n = len(gaps)
    mean = st.mean(gaps)
    pos = sum(1 for g in gaps if g > 0)

    rng = random.Random(args.seed)
    means = []
    for _ in range(args.resamples):
        means.append(sum(gaps[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * args.resamples)]
    hi = means[int(0.975 * args.resamples) - 1]

    print("a           : %s" % args.a)
    print("b           : %s" % args.b)
    print("paired seeds: %s  (n=%d)" % (shared, n))
    print("per-seed gap: %s" % ["%+.4f" % g for g in gaps])
    print("mean gap    : %+.4f" % mean)
    print("direction   : %d/%d positive" % (pos, n))
    print("95%% CI      : [%+.4f, %+.4f]  (%d resamples, rng seed %d)"
          % (lo, hi, args.resamples, args.seed))
    excludes = (lo > 0) or (hi < 0)
    print("excludes 0  : %s" % excludes)
    if pos in (0, n):
        print("sign test   : one-sided exact p = %.4f (floor at this n)" % (0.5 ** n))
    else:
        print("sign test   : not unanimous, %d/%d, do not quote a sign test" % (pos, n))


if __name__ == "__main__":
    main()
