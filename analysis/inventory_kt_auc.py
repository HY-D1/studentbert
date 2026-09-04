#!/usr/bin/env python3
# Read-only inventory of every knowledge-tracing test AUC in the logs.
#
# Why this exists: RESULTS.md section 1 (the 7-dataset baseline table) is marked
# "Recorded", and RESULTS_generated.md carries the identical hardcoded block, so
# 16 of its 28 cells have never been reconstructed from a log. audit_all_claims.py
# cannot cover them either: its BANNER regex requires a parenthesised run name
# beginning "edubert_", and train_baseline.py prints "=== DKT on <dataset> ===",
# which has no parentheses at all.
#
# This script does not decide which run belongs in which table cell. It reports
# every block it finds, with the seed and the source file, so the mapping can be
# made deliberately instead of by grep.
#
# Print formats it targets (verified against the scripts in this repo):
#   scripts/train_baseline.py:252-254
#       === DKT on algebra2005 ===
#       test AUC     : 0.7983
#   scripts/finetune_edubert.py:281-284
#       === EduBERT-KT (kt_assist_scratch_n3000_seed42, init=scratch) ===
#       test AUC       : 0.6702
#   both scripts print, earlier in the same block:
#       device=cuda  num_skills=102  run=<run_name>  seed=<n>
#
# Seed attribution never uses a prefix match on the run name, so the documented
# seed4/seed42 collision cannot occur here.
#
# Usage, from the repo root on the cluster:
#   python3 analysis/inventory_kt_auc.py
#   python3 analysis/inventory_kt_auc.py --logdir . --logdir ../logs --out kt_inventory.tsv

from __future__ import annotations

import argparse
import glob
import os
import re
import statistics as st
from collections import defaultdict

BASELINE_BANNER = re.compile(r"^===\s+([A-Z][A-Za-z+]*)\s+on\s+(\S+)\s+===")
EDUBERT_BANNER = re.compile(r"^===\s+EduBERT-KT\s+\(([^,]+),\s*init=([A-Za-z]+)\)\s*===")
TEST_AUC = re.compile(r"^test AUC\s*:\s*([0-9]*\.?[0-9]+)")
RUN_LINE = re.compile(r"\brun=(\S+)\s+seed=([0-9]+)")
SEED_IN_NAME = re.compile(r"_seed([0-9]+)(?:_|$)")


def seed_from(run_name, fallback):
    m = SEED_IN_NAME.search(run_name)
    if m:
        return int(m.group(1))
    return fallback


def scan_file(path):
    """Return a list of dicts, one per completed test-AUC block in this file."""
    out = []
    pending = None
    last_run, last_seed = None, None
    with open(path, "r", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            rl = RUN_LINE.search(line)
            if rl:
                last_run, last_seed = rl.group(1), int(rl.group(2))
                continue
            mb = BASELINE_BANNER.match(line)
            if mb:
                pending = {
                    "kind": "baseline",
                    "model": mb.group(1).upper(),
                    "dataset": mb.group(2),
                    "run": last_run or "",
                    "init": "",
                    "seed": seed_from(last_run or "", last_seed),
                    "file": os.path.basename(path),
                    "line": lineno,
                }
                continue
            me = EDUBERT_BANNER.match(line)
            if me:
                run = me.group(1).strip()
                pending = {
                    "kind": "edubert",
                    "model": "EduBERT-KT",
                    "dataset": "",
                    "run": run,
                    "init": me.group(2),
                    "seed": seed_from(run, last_seed),
                    "file": os.path.basename(path),
                    "line": lineno,
                }
                continue
            ma = TEST_AUC.match(line.strip())
            if ma and pending is not None:
                pending["auc"] = float(ma.group(1))
                out.append(pending)
                pending = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", action="append", default=None,
                    help="directory to scan for *.log; repeatable")
    ap.add_argument("--out", default="kt_inventory.tsv")
    args = ap.parse_args()

    dirs = args.logdir or ["."]
    files = []
    for d in dirs:
        files.extend(sorted(glob.glob(os.path.join(d, "*.log"))))
    print("scanned %d log files across %d directories" % (len(files), len(dirs)))
    if not files:
        print("NO LOGS FOUND. Pass --logdir pointing at the directory holding *.log")
        return

    rows = []
    for f in files:
        rows.extend(scan_file(f))
    print("found %d completed test-AUC blocks" % len(rows))

    with open(args.out, "w") as fh:
        fh.write("kind\tmodel\tdataset\trun\tinit\tseed\tauc\tfile\tline\n")
        for r in rows:
            fh.write("%s\t%s\t%s\t%s\t%s\t%s\t%.4f\t%s\t%d\n" % (
                r["kind"], r["model"], r["dataset"], r["run"], r["init"],
                r["seed"] if r["seed"] is not None else "", r["auc"],
                r["file"], r["line"]))
    print("wrote %s" % args.out)

    # Resubmitted jobs write the same run into a second log. Aggregate over
    # unique (run, seed) only, or every resubmit inflates n and shrinks pstdev.
    uniq, keys = [], set()
    for r in rows:
        k = (r["kind"], r["model"], r["dataset"], r["run"], r["seed"])
        if k in keys:
            continue
        keys.add(k)
        uniq.append(r)
    print("deduplicated to %d unique (run, seed) results" % len(uniq))

    print("\n--- BASELINES: model x dataset ---")
    groups = defaultdict(list)
    for r in uniq:
        if r["kind"] == "baseline":
            groups[(r["model"], r["dataset"])].append(r)
    for key in sorted(groups):
        vals = [g["auc"] for g in groups[key]]
        seeds = sorted(str(g["seed"]) for g in groups[key])
        sd = st.pstdev(vals) if len(vals) > 1 else 0.0
        print("%-6s %-14s mean %.4f  pstdev %.4f  n=%d  seeds=%s"
              % (key[0], key[1], st.mean(vals), sd, len(vals), ",".join(seeds)))

    print("\n--- EDUBERT-KT: run name x init ---")
    egroups = defaultdict(list)
    for r in uniq:
        if r["kind"] == "edubert":
            stem = SEED_IN_NAME.sub("_seed*", r["run"])
            egroups[(stem, r["init"])].append(r)
    for key in sorted(egroups):
        vals = [g["auc"] for g in egroups[key]]
        seeds = sorted(str(g["seed"]) for g in egroups[key])
        sd = st.pstdev(vals) if len(vals) > 1 else 0.0
        print("%-46s %-11s mean %.4f  pstdev %.4f  n=%d  seeds=%s"
              % (key[0], key[1], st.mean(vals), sd, len(vals), ",".join(seeds)))

    print("\n--- DUPLICATE RUNS (same run name and seed seen more than once) ---")
    seen = defaultdict(list)
    for r in rows:
        if r["run"]:
            seen[(r["run"], r["seed"])].append((r["file"], r["auc"]))
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    if not dupes:
        print("none")
    for k in sorted(dupes):
        vals = sorted(set(round(a, 4) for _, a in dupes[k]))
        tag = "IDENTICAL" if len(vals) == 1 else "DISAGREE"
        print("%s seed=%s  %s  %s  files=%s"
              % (k[0], k[1], tag, vals, [f for f, _ in dupes[k]]))


if __name__ == "__main__":
    main()
