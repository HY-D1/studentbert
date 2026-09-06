#!/usr/bin/env python3
# Unified read-only inventory of every test metric in the logs.
#
# Supersedes inventory_kt_auc.py, which matched only two banner shapes and was
# therefore blind to all 72 next-skill runs of the N=3000 grid.
#
# Banner shapes handled, verified against the scripts in this repo:
#   scripts/train_baseline.py:252      === DKT on algebra2005 ===
#   scripts/finetune_edubert.py:281    === EduBERT-KT (<run>, init=<init>) ===
#   scripts/downstream_edubert.py:515  === next_skill (<run>, init=<init>) ===
#                                      === dropout (<run>, init=<init>) ===
#                                      === probe (<run>, init=<init>) ===
# Metric lines, same sources:
#   test AUC              : 0.6710
#   test top-1 acc        : 0.7873
#   test macro-OVR AUC    : 0.9819  (over 92 classes)
#   test F1 (minority)    : 0.3120
# Anything starting "best " is ignored, so validation numbers never leak in.
#
# Emits one row per (run, seed, metric). Seeds come from the run string where
# present and from the "run=... seed=..." line otherwise, never from a prefix
# match, so seed4 cannot be confused with seed42. Resubmitted jobs write the
# same run twice; those are deduplicated and reported separately, flagged
# DISAGREE when the two copies differ.
#
# Usage, from the repo root on the cluster:
#   python3 analysis/inventory_runs.py --logdir . --logdir ../logs --out run_inventory.tsv

from __future__ import annotations

import argparse
import glob
import os
import re
import statistics as st
import sys
from collections import defaultdict

BASELINE_BANNER = re.compile(r"^===\s+([A-Z][A-Za-z+]*)\s+on\s+(\S+)\s+===")
KT_BANNER = re.compile(r"^===\s+EduBERT-KT\s+\(([^,]+),\s*init=([A-Za-z]+)\)\s*===")
# A task banner may carry a parenthesised qualifier before the run name, as the
# deprecated v1 probe does: "=== probe (skill-identity) (<run>, init=...) ===".
# The v1 pattern swallowed the qualifier into the run name, producing rows whose
# run began "skill-identity)". The qualifier is now captured separately and the
# run group forbids parentheses so it cannot absorb one.
TASK_BANNER = re.compile(
    r"^===\s+([a-z][a-z_]*)(?:\s+\(([^()]*)\))?\s+\(([^(),]+),\s*init=([A-Za-z]+)\)\s*===")
# Families whose values must never enter a paper. Matched against `kind`.
QUARANTINE = {
    "probe(skill-identity)":
        "deprecated v1 circular probe, the target is recoverable from the input",
}
METRIC = re.compile(r"^test\s+(.+?)\s*:\s*([0-9]*\.?[0-9]+)")
RUN_LINE = re.compile(r"\brun=(\S+)\s+seed=([0-9]+)")
SEED_IN_NAME = re.compile(r"_seed([0-9]+)(?:_|$)")
SUMMARY_METRICS = ("test_auc", "test_macro_ovr_auc", "test_top_1_acc")


def slug(name):
    s = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower())
    return "test_" + s.strip("_")


def seed_from(run_name, fallback):
    m = SEED_IN_NAME.search(run_name or "")
    return int(m.group(1)) if m else fallback


def scan_file(path):
    rows = []
    cur = None
    last_run, last_seed = None, None
    with open(path, "r", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            rl = RUN_LINE.search(line)
            if rl:
                last_run, last_seed = rl.group(1), int(rl.group(2))
                continue
            mb = BASELINE_BANNER.match(line)
            if mb:
                cur = {"kind": "baseline", "model": mb.group(1).upper(),
                       "dataset": mb.group(2), "run": last_run or "", "init": "",
                       "seed": seed_from(last_run, last_seed),
                       "file": os.path.basename(path), "line": lineno}
                continue
            mk = KT_BANNER.match(line)
            if mk:
                run = mk.group(1).strip()
                cur = {"kind": "kt", "model": "EduBERT-KT", "dataset": "",
                       "run": run, "init": mk.group(2),
                       "seed": seed_from(run, last_seed),
                       "file": os.path.basename(path), "line": lineno}
                continue
            mt = TASK_BANNER.match(line)
            if mt:
                run = mt.group(3).strip()
                kind = mt.group(1)
                if mt.group(2):
                    kind = "%s(%s)" % (kind, mt.group(2).strip())
                cur = {"kind": kind, "model": "EduBERT", "dataset": "",
                       "run": run, "init": mt.group(4),
                       "seed": seed_from(run, last_seed),
                       "file": os.path.basename(path), "line": lineno}
                continue
            if line.startswith("==="):
                cur = None
                continue
            mm = METRIC.match(line.strip())
            if mm and cur is not None:
                r = dict(cur)
                r["metric"] = slug(mm.group(1))
                r["value"] = float(mm.group(2))
                r["line"] = lineno
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", action="append", default=None)
    ap.add_argument("--out", default="run_inventory.tsv")
    args = ap.parse_args()

    dirs = args.logdir or ["."]
    files = []
    for d in dirs:
        files.extend(sorted(glob.glob(os.path.join(d, "*.log"))))
    print("scanned %d log files across %d directories" % (len(files), len(dirs)))
    if not files:
        sys.exit("NO LOGS FOUND. Pass --logdir pointing at the directory holding *.log")

    rows = []
    for f in files:
        rows.extend(scan_file(f))
    print("found %d metric rows" % len(rows))

    quarantined = [r for r in rows if r["kind"] in QUARANTINE]
    rows = [r for r in rows if r["kind"] not in QUARANTINE]
    if quarantined:
        qpath = args.out.replace(".tsv", "_quarantined.tsv")
        with open(qpath, "w") as fh:
            fh.write("kind\trun\tseed\tmetric\tvalue\tfile\treason\n")
            for r in quarantined:
                fh.write("%s\t%s\t%s\t%s\t%.6f\t%s\t%s\n" % (
                    r["kind"], r["run"], r["seed"], r["metric"], r["value"],
                    r["file"], QUARANTINE[r["kind"]]))
        print("\n!! QUARANTINED %d rows, written to %s and EXCLUDED from the main file"
              % (len(quarantined), qpath))
        for k in sorted(set(r["kind"] for r in quarantined)):
            print("!!   %s : %s" % (k, QUARANTINE[k]))

    uniq, seen = [], set()
    dupes = defaultdict(list)
    for r in rows:
        k = (r["kind"], r["model"], r["dataset"], r["run"], r["seed"], r["metric"])
        dupes[k].append((r["file"], r["value"]))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    print("deduplicated to %d unique (run, seed, metric) rows" % len(uniq))

    with open(args.out, "w") as fh:
        fh.write("kind\tmodel\tdataset\trun\tinit\tseed\tmetric\tvalue\tfile\tline\n")
        for r in uniq:
            fh.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.6f\t%s\t%d\n" % (
                r["kind"], r["model"], r["dataset"], r["run"], r["init"],
                r["seed"] if r["seed"] is not None else "", r["metric"],
                r["value"], r["file"], r["line"]))
    print("wrote %s" % args.out)

    print("\n--- metrics present ---")
    mc = defaultdict(int)
    for r in uniq:
        mc[r["metric"]] += 1
    for m in sorted(mc, key=lambda x: -mc[x]):
        print("%-28s %d" % (m, mc[m]))

    print("\n--- BASELINES (test_auc) ---")
    g = defaultdict(list)
    for r in uniq:
        if r["kind"] == "baseline" and r["metric"] == "test_auc":
            g[(r["model"], r["dataset"], SEED_IN_NAME.sub("_seed*", r["run"]))].append(r)
    fams = defaultdict(set)
    for k in g:
        fams[(k[0], k[1])].add(k[2])
    for k in sorted(g):
        v = [x["value"] for x in g[k]]
        sd = st.pstdev(v) if len(v) > 1 else 0.0
        warn = "  <-- MULTIPLE CAMPAIGNS ON THIS CELL" if len(fams[(k[0], k[1])]) > 1 else ""
        print("%-6s %-14s %-46s mean %.4f  pstdev %.4f  n=%d  seeds=%s%s"
              % (k[0], k[1], k[2] or "(no run name)", st.mean(v), sd, len(v),
                 ",".join(sorted(str(x["seed"]) for x in g[k])), warn))
    noseed = [r for r in uniq if r["kind"] == "baseline" and r["seed"] is None]
    if noseed:
        print("\n%d baseline rows have no recoverable seed and cannot be paired:"
              % len(noseed))
        for r in noseed:
            print("   %s %s %s in %s" % (r["model"], r["dataset"], r["metric"], r["file"]))

    for metric in SUMMARY_METRICS[:1] + SUMMARY_METRICS[1:]:
        eg = defaultdict(list)
        for r in uniq:
            if r["kind"] != "baseline" and r["metric"] == metric and r["run"]:
                eg[(SEED_IN_NAME.sub("_seed*", r["run"]), r["kind"])].append(r)
        if not eg:
            continue
        print("\n--- EduBERT runs (%s) ---" % metric)
        for k in sorted(eg):
            v = [x["value"] for x in eg[k]]
            sd = st.pstdev(v) if len(v) > 1 else 0.0
            print("%-54s %-11s mean %.4f  pstdev %.4f  n=%d  seeds=%s"
                  % (k[0], k[1], st.mean(v), sd, len(v),
                     ",".join(sorted(str(x["seed"]) for x in eg[k]))))

    print("\n--- DUPLICATE (run, seed, metric) ---")
    bad = {k: v for k, v in dupes.items() if len(v) > 1}
    if not bad:
        print("none")
    for k in sorted(bad):
        vals = sorted(set(round(x, 6) for _, x in bad[k]))
        print("%s seed=%s %s  %s  %s  files=%s"
              % (k[3], k[4], k[5], "IDENTICAL" if len(vals) == 1 else "DISAGREE",
                 vals, sorted(set(f for f, _ in bad[k]))))


if __name__ == "__main__":
    main()
