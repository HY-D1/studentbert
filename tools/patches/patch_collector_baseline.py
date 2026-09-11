#!/usr/bin/env python3
# Stop collect_all_results.py reintroducing superseded numbers.
#
# The script already refuses to overwrite RESULTS.md when hand-added sections
# are present, which is good. Three gaps remain.
#
#  1. BASELINE_TABLE is a hardcoded dict still holding the pre-campaign values,
#     including DKT on ASSISTments 2017 as 0.690 against a measured 0.6974. A
#     single --force run reintroduces the wrong number that took a 270-job
#     campaign to find.
#  2. The MANUAL guard lists "### 2.1" but not "### 2.2" or "## 1.", so those
#     two rebuilt sections are less protected than the rest.
#  3. The section 1 renderer prints "(Recorded uniform table)" and a _Read_ line
#     asserting DKT wins the 4 newer datasets, which is no longer true.
#
# This patch makes the collector READ the table from run_inventory.tsv when it
# is available, so the numbers come from logs rather than from a constant, and
# refuse to emit section 1 at all when it is not. That retires the constant as
# a source of truth instead of just correcting it once.
#
#   python3 tools/patches/patch_collector_baseline.py --file analysis/collect_all_results.py --dry-run
#   python3 tools/patches/patch_collector_baseline.py --file analysis/collect_all_results.py

from __future__ import annotations

import argparse
import hashlib
import sys

OLD_DICT = '''BASELINE_TABLE = {
  "ASSIST2017":  [0.690,0.650,0.670,0.693],
  "EdNet":       [0.680,0.672,0.678,0.685],
  "Junyi":       [0.759,0.754,0.757,0.758],
  "Algebra2005": [0.798,0.776,0.781,0.787],
  "Bridge2006":  [0.795,0.773,0.773,0.775],
  "ASSIST2009":  [0.876,0.863,0.870,0.870],
  "Algebra2006": [0.803,0.775,0.787,0.790],
}'''

NEW_DICT = '''# Section 1 is no longer carried as a constant. It is read from the run
# inventory so the numbers come from logs, which is what the header of this
# file promises. The old hardcoded dict held pre-campaign values, including
# DKT on ASSISTments 2017 as 0.690 against a measured 0.6974, and a single
# --force run would have reintroduced it.
BASELINE_STEMS = {
  "ASSIST2017":  ("assist2017",  "edubert_assist2017_kt_assist_scratch_n3000",
                                 "edubert_assist2017_kt_assist_indomain_n3000"),
  "EdNet":       ("ednet",       "edubert_ednet_ktfull_ednet_scratch_n20000",
                                 "edubert_ednet_ktfull_ednet_indomain_n20000"),
  "Junyi":       ("junyi",       "edubert_junyi_ktfull_junyi_scratch_n40000",
                                 "edubert_junyi_ktfull_junyi_indomain_n40000"),
  "Algebra2005": ("algebra2005", "edubert_algebra2005_scratch_algebra2005",
                                 "edubert_algebra2005_algabl_full"),
  "Bridge2006":  ("bridge2006",  "edubert_bridge2006_scratch_bridge2006",
                                 "edubert_bridge2006_bridgeabl_full"),
  "ASSIST2009":  ("assist2009",  "edubert_assist2009_scratch_assist2009",
                                 "edubert_assist2009_a09abl_full"),
  "Algebra2006": ("algebra2006", "edubert_algebra2006_scratch_algebra2006",
                                 "edubert_algebra2006_alg06abl_full"),
}


def baseline_table_from_inventory(tsv):
    """DKT / AKT / SAINT / scratch / pretrained per dataset, from the inventory.

    Baselines are taken from the base2_ campaign only, so the older campaigns on
    the same model and dataset cannot be averaged in. Returns None if the file is
    missing, and the caller then omits section 1 rather than inventing it.
    """
    import csv as _csv
    import os as _os
    import re as _re
    import statistics as _st
    from collections import defaultdict as _dd
    if not _os.path.exists(tsv):
        return None
    base, edu = _dd(dict), _dd(dict)
    for r in _csv.DictReader(open(tsv, encoding="utf-8", errors="replace"), delimiter="\\t"):
        if r.get("metric") != "test_auc" or not r.get("seed"):
            continue
        run, seed, val = r["run"], int(r["seed"]), float(r["value"])
        if r.get("kind") == "baseline" and "base2_" in run:
            m = _re.match(r"([a-z]+)_([a-z0-9]+)_base2_", run)
            if m:
                base[(m.group(1).upper(), m.group(2))][seed] = val
        else:
            stem = _re.sub(r"_seed[0-9]+$", "", run)
            edu[stem][seed] = val
    out = {}
    for label, (ds, scr, pre) in BASELINE_STEMS.items():
        row = []
        for model in ("DKT", "AKT", "SAINT"):
            d = base.get((model, ds), {})
            row.append(_st.mean(d.values()) if d else None)
        for stem in (scr, pre):
            d = edu.get(stem, {})
            row.append(_st.mean(d.values()) if d else None)
        out[label] = row
    return out'''

OLD_RENDER = '''    W("\\n---\\n## 1. Baseline table (7 datasets)\\n")
    W("Knowledge-tracing test AUC. Columns: DKT / AKT / EduBERT-scratch / EduBERT-pretrained. "
      "(Recorded uniform table; EduBERT-pretrained = full-objective number for the 4 newer sets.)\\n")
    W("| Dataset | DKT | AKT | scratch | EduBERT-pt |")
    W("|---|---|---|---|---|")
    for ds,row in BASELINE_TABLE.items():
        W(f"| {ds} | {row[0]:.3f} | {row[1]:.3f} | {row[2]:.3f} | **{row[3]:.3f}** |")
    W("\\n_Read: DKT is the strongest simple baseline and wins on the 4 newer datasets; the pretraining "
      "edge concentrates on the original 3 (esp in-domain EdNet); AKT weakest. Pretraining is regime/scale "
      "specific, not universal._")'''

NEW_RENDER = '''    bt = coll.get("baseline_table")
    if not bt:
        W("\\n---\\n## 1. Baseline table (7 datasets)\\n")
        W("NOT FOUND: run_inventory.tsv was not available, so this section was not "
          "regenerated. Build it with `python analysis/inventory_runs.py --logdir . "
          "--logdir ../logs --out run_inventory.tsv` and rerun, or keep the existing "
          "hand-verified section.")
    else:
        W("\\n---\\n## 1. Baseline table (7 datasets)\\n")
        W("Knowledge-tracing test AUC, mean over seeds, read from run_inventory.tsv. "
          "Columns: DKT / AKT / SAINT+ / EduBERT-scratch / EduBERT-pretrained. Baselines "
          "are the base2_ campaign, budget matched to the EduBERT row on that dataset. "
          "EduBERT-pretrained is in-domain for the original 3 and the EdNet-source "
          "full-objective encoder for the 4 newer sets. Rows are not comparable with "
          "each other because target budgets differ.\\n")
        W("| Dataset | DKT | AKT | SAINT+ | scratch | EduBERT-pt |")
        W("|---|---|---|---|---|---|")
        for ds, row in bt.items():
            cells = []
            best = max((v for v in row if v is not None), default=None)
            for v in row:
                if v is None:
                    cells.append("NOT FOUND")
                elif best is not None and abs(v - best) < 1e-9:
                    cells.append(f"**{v:.4f}**")
                else:
                    cells.append(f"{v:.4f}")
            W(f"| {ds} | " + " | ".join(cells) + " |")
        W("\\n_Read: bold is the highest AUC in the row. Any NOT FOUND cell means the "
          "inventory held no rows for that stem; do not fill it by hand, extend the "
          "inventory instead._")'''

OLD_GUARD = '''    MANUAL = ["### 2.1", "### 3.1", "### 3.2", "### 6.1", "### 8.1", "### 8.2", "## 10."]'''
NEW_GUARD = '''    MANUAL = ["### 2.1", "### 2.2", "### 3.1", "### 3.2", "### 6.1", "### 8.1",
              "### 8.2", "## 10."]'''

OLD_CALL = '''    coll["baseline_table"] = BASELINE_TABLE'''
NEW_CALL = '''    coll["baseline_table"] = baseline_table_from_inventory(args.inventory)
    if coll["baseline_table"] is None:
        print("WARNING: %s not found, section 1 will be written as NOT FOUND"
              % args.inventory)'''

OLD_ARG = '''    ap.add_argument("--force", action="store_true",'''
NEW_ARG = '''    ap.add_argument("--inventory", default="run_inventory.tsv",
                    help="TSV from analysis/inventory_runs.py, source for section 1")
    ap.add_argument("--force", action="store_true",'''

EDITS = [("baseline dict", OLD_DICT, NEW_DICT),
         ("section-1 renderer", OLD_RENDER, NEW_RENDER),
         ("manual guard list", OLD_GUARD, NEW_GUARD),
         ("inventory arg", OLD_ARG, NEW_ARG),
         ("baseline_table call", OLD_CALL, NEW_CALL)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="analysis/collect_all_results.py")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t = open(args.file, encoding="utf-8").read()
    print("before md5 %s" % hashlib.md5(t.encode("utf-8")).hexdigest())
    applied, skipped = [], []
    for name, old, new in EDITS:
        if new in t:
            skipped.append(name)
            continue
        if t.count(old) != 1:
            sys.exit("ABORT, no write. Anchor %r matched %d times, expected 1."
                     % (name, t.count(old)))
        t = t.replace(old, new)
        applied.append(name)
    for n in applied:
        print("  apply   %s" % n)
    for n in skipped:
        print("  already %s" % n)
    if args.dry_run:
        print("dry run, nothing written")
        return
    if not applied:
        print("nothing to do")
        return
    open(args.file, "w", encoding="utf-8").write(t)
    print("after  md5 %s" % hashlib.md5(t.encode("utf-8")).hexdigest())


if __name__ == "__main__":
    main()
