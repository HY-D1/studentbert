#!/usr/bin/env python3
"""Patch the two analysis scripts so regenerating RESULTS.md cannot regress it.

WHY THIS EXISTS
    RESULTS.md has been hand-corrected several times. collect_all_results.py was
    not updated alongside it, so re-running the collector would silently revert
    corrected numbers to superseded ones. audit_all_claims.py derives one pps
    value with the wrong denominator. This patch brings both scripts in line with
    the verified record before RESULTS.md is regenerated to pick up the new
    K=512 scratch control.

WHAT IT CHANGES (11 edits, all in analysis/)

  collect_all_results.py
    1  prospective note        removes "first prospective predict-before-test";
                               git shows the written prediction was committed
                               2026-07-25 18:39:09, after the skill_only logs at
                               18:22-18:25, so it is not a documented advance
                               prediction. Replaced with the seventh-dataset
                               framing, which the timestamps do support.
    2  K=512 note              "pps5.0" -> untruncated, pps 4.32. Median length is
                               441 < 512, so K=512 truncates nothing and pps is
                               441/102 = 4.324, not 512/102.
    3  pps_ordering values     2dp -> 3dp, matching section 0.1.
    4  lodo_negative note      unsourced 5/6 and 3/6 -> the reproducible 7-fold
                               result from analysis/lodo_regime.py.
    5  paper tags legend       venue tags -> analysis-script provenance.
    6  section 4 CIs           adds the paired-by-seed intervals for all 7.
    7  K=512 render line       "pps~5.0" -> untruncated, pps 4.32.
    8  scratch/crossover line  removes "rises smoothly" and "Crossover ~pps 1.5",
                               both of which the truncation table contradicts,
                               and states the measured K=512 scratch result.
    9  honest limit (sec 5)    "0.33-2.79" -> 0.325-2.414 raw, 0.325-1.041 capped.
   10  honest negatives        same gap correction in section 9.
   11  headings                strips [EDM]/[LAK]/[NeurIPS]/[ICLR] tags.

  audit_all_claims.py
   12  pps denominator         min(K, 441)/102 instead of K/102, so the audit
                               stops reporting the K=512 endpoint as pps 5.02.

IDEMPOTENT
    Every edit tests for the post-patch string first. Running twice is a no-op and
    leaves the md5 unchanged. If an edit's old text is missing AND its new text is
    missing, the file has drifted and the script aborts without writing anything.

USAGE (from the repo root, after backing up)
    python3 tools/patches/patch_results_provenance.py --dry-run
    python3 tools/patches/patch_results_provenance.py
"""
import argparse
import ast
import hashlib
import os
import sys

COLLECT = "analysis/collect_all_results.py"
AUDIT = "analysis/audit_all_claims.py"

EDITS = []


def edit(path, name, old, new):
    EDITS.append({"path": path, "name": name, "old": old, "new": new})


# ---------------------------------------------------------------- collector
edit(COLLECT, "01 prospective note",
     '     "note": "first prospective predict-before-test to hold (recorded)"},',
     '     "note": "seventh and final dataset; preprocessed 2026-07-25 17:41 and '
     'evaluated the same evening, after the ordering had been set on the other six. '
     'The written prediction was committed 18:39:09, AFTER the skill_only logs '
     '(18:22-18:25), so this is an out-of-sample test and NOT a documented advance '
     'prediction (recorded)"},')

edit(COLLECT, "02 K=512 note",
     '     "note": "ASSIST2017 truncated K=512/pps5.0: skill-driven (correct-skill negative) (recorded)"},',
     '     "note": "ASSIST2017 at K=512 is UNTRUNCATED (median length 441 < 512), so '
     'pps = 441/102 = 4.32, not 512/102: skill-driven (correct-skill negative) (recorded)"},')

edit(COLLECT, "03 pps_ordering values",
     '     "values": {"Junyi":0.066,"EdNet":0.211,"ASSIST2009":0.325,"Algebra2006":2.41,\n'
     '                "Bridge2006":2.79,"ASSIST2017":4.33,"Algebra2005":5.33}},',
     '     "values": {"Junyi":0.066,"EdNet":0.211,"ASSIST2009":0.325,"Algebra2006":2.414,\n'
     '                "Bridge2006":2.791,"ASSIST2017":4.324,"Algebra2005":5.330}},')

edit(COLLECT, "04 lodo_negative note",
     '     "note": "leave-one-dataset-out: multivariate 5/6 but overparameterized at n=6, '
     'not trustworthy; single features n_students 5/6, pps 3/6. No combo generalizes. NEGATIVE."},',
     '     "note": "leave-one-dataset-out over all 7, reproducible via analysis/lodo_regime.py: '
     'n_students 6/7, n_interactions 6/7, n_skills 6/7, pps 5/7 under both definitions, '
     'correct_rate 3/7. Four features separate all 7 in-sample, so in-sample separation is '
     'uninformative. No feature reaches 7/7. The 2 folds holding out a correctness-driven '
     'dataset leave a single such example in training. NEGATIVE."},')

edit(COLLECT, "05 paper tags legend",
     '    W("\\n**Paper tags:** `[EDM]` source comparison + seq-length ; `[LAK]` low-N advantage ; "\n'
     '      "`[NeurIPS]` mechanism (objective reversal, causal flip, pps) ; `[ICLR]` representation/mechanism.\\n")',
     '    W("\\n**Analysis scripts:** paired_bootstrap_objective.py (objective CIs, all 7 targets) ; "\n'
     '      "lodo_regime.py (leave-one-dataset-out) ; parse_nextskill_full.py (section 3) ; "\n'
     '      "audit_all_claims.py (standing audit).\\n")')

edit(COLLECT, "06 section 4 paired CIs",
     '    W("\\n**Recorded reversals (paired bootstrap):**")',
     '    W("\\n**Paired-by-seed contrasts, all 7 targets '
     '(analysis/paired_bootstrap_objective.py over objabl_perseed.csv, 6 seeds, '
     '20000 bootstrap resamples, rng seed 0):**\\n")\n'
     '    W("| Target | skill_only - correct_only | 95% CI | seeds in predicted direction |")\n'
     '    W("|---|---|---|---|")\n'
     '    for _t, _m, _ci in [("assist2017","+0.0240","[+0.0222, +0.0261]"),\n'
     '                        ("ednet","-0.0069","[-0.0078, -0.0055]"),\n'
     '                        ("junyi","-0.0122","[-0.0136, -0.0110]"),\n'
     '                        ("algebra2005","+0.0228","[+0.0173, +0.0281]"),\n'
     '                        ("bridge2006","+0.0105","[+0.0093, +0.0116]"),\n'
     '                        ("assist2009","+0.0030","[+0.0023, +0.0038]"),\n'
     '                        ("algebra2006","+0.0186","[+0.0148, +0.0228]")]:\n'
     '        W(f"| {_t} | {_m} | {_ci} | 6/6 |")\n'
     '    W("\\n_All 7 intervals exclude 0 and all 7 are unanimous across 6 seeds in the '
     'direction the regime label predicts. Exact one-sided sign p = 0.0156 for every row, '
     'which is the floor at 6 seeds. NOTE: ednet skill-correct -0.0069 is NOT the same '
     'quantity as the in-domain EdNet full-scale KT gain +0.0069 in section 3._")\n'
     '    W("\\n**Recorded reversals (paired bootstrap):**")')

edit(COLLECT, "07 K=512 render",
     "    W(f\"- K=512 (pps~5.0): correct-skill {t512['value']} CI {t512['ci']} ({t512['seeds']}) -> {t512['note']}\")",
     "    W(f\"- K=512 (untruncated, pps 4.32): correct-skill {t512['value']} CI {t512['ci']} ({t512['seeds']}) -> {t512['note']}\")")

edit(COLLECT, "08 scratch control line",
     '    W("- Scratch control rises smoothly with K (rules out pure data-quantity). Crossover ~pps 1.5.")',
     '    W("- Scratch control (6 seeds at every K including 512, the K=512 cell added 2026-08-11, '
     'jobs 9081378-9081383): 0.6347, 0.6476, 0.6423, 0.6523, 0.6586, 0.6597, 0.6676 at K=10..512. '
     'It is NOT monotone. From K=320 to K=512 scratch gains +0.0079 while full gains +0.0327 and '
     'skill_only +0.0331, and correct_only gains +0.0056, less than scratch. The endpoint '
     'divergence is therefore objective-specific and not a generic data-quantity effect. At '
     'K=512 correct_only (0.6655) falls BELOW scratch (0.6676). The skill_only minus '
     'correct_only gap is negative at every K through 320 (pps 3.14) and positive only at '
     'K=512 (pps 4.32); the sign change is bracketed between pps 3.14 and 4.32, NOT near 1.5 '
     '(at K=160, pps 1.57, the gap is still -0.0046). Truncation keeps the most '
     'recent K interactions per learner, so it lowers total interactions, time horizon and skill '
     'composition together; density is not isolated.")')

edit(COLLECT, "09 honest limit section 5",
     '    W("\\n**Honest limit:** precise threshold fuzzy (empty gap 0.33-2.79). LODO does not generalize at n=6 "',
     '    W("\\n**Honest limit:** precise threshold fuzzy. Empty pps gap is 0.325-2.414 on raw '
     'median length, and 0.325-1.041 under the 512-step model cap, which lowers Algebra2005 '
     'to 4.697, Bridge2006 to 1.041 and Algebra2006 to 1.058. The ordering and the regime split '
     'hold under both definitions; state which one any interval claim uses. LODO does not generalize "')

edit(COLLECT, "10 honest negatives gap",
     '    W("- Regime threshold is fuzzy (empty pps gap 0.33-2.79); pps ordering holds, exact boundary not localized.")',
     '    W("- Regime threshold is fuzzy (empty pps gap 0.325-2.414 raw, 0.325-1.041 under the '
     '512-step cap); pps ordering holds under both definitions, exact boundary not localized.")')

for _sec, _tags in [
    ("## 1. Baseline table (7 datasets)", "  `[EDM]` `[all]`"),
    ("## 2. Source comparison: scale > granularity", "  `[EDM]`"),
    ("## 3. Low-resource advantage & scale boundary", "  `[LAK]`"),
    ("## 4. Objective reversal (headline)", "  `[NeurIPS]` `[ICLR]`"),
    ("## 5. What governs the regime: practice-per-skill", "  `[NeurIPS]`"),
    ("## 6. Probe mechanism (7 datasets)", "  `[ICLR]` `[NeurIPS]`"),
    ("## 7. Embedding geometry (honest negative)", "  `[ICLR]`"),
    ("## 8. Downstream tasks", "  `[LAK]`"),
]:
    edit(COLLECT, f"11 heading tags: {_sec[:28]}",
         f'{_sec}{_tags}\\n")',
         f'{_sec}\\n")')

# -------------------------------------------------------------------- audit
edit(AUDIT, "12 audit pps denominator",
     '            lo, hi = max(neg), min(pos)\n'
     '            say(INFO, f"sign change lies between K={lo} (pps {lo/102:.2f}) and K={hi} (pps {hi/102:.2f})")',
     '            lo, hi = max(neg), min(pos)\n'
     '            def epps(k, med=441.0, ns=102.0):\n'
     '                return min(k, med) / ns\n'
     '            say(INFO, f"sign change lies between K={lo} (pps {epps(lo):.2f}) and K={hi} (pps {epps(hi):.2f})")')

edit(AUDIT, "13 audit pps denominator, FAIL branch",
     '                          f"the change is between pps {lo/102:.2f} and {hi/102:.2f}")',
     '                          f"the change is between pps {epps(lo):.2f} and {epps(hi):.2f}")')


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for p in (COLLECT, AUDIT):
        if not os.path.exists(p):
            sys.exit(f"ABORT: {p} not found. Run from the repo root.")

    before = {p: md5(p) for p in (COLLECT, AUDIT)}
    texts = {p: open(p).read() for p in (COLLECT, AUDIT)}

    applied, skipped, failed = 0, 0, []
    for e in EDITS:
        t = texts[e["path"]]
        if e["new"] in t:
            print(f"  SKIP     {e['name']}  (already applied)")
            skipped += 1
            continue
        n = t.count(e["old"])
        if n == 0:
            print(f"  NOT FOUND {e['name']}")
            failed.append(e["name"])
            continue
        if n > 1:
            print(f"  AMBIGUOUS {e['name']}  ({n} matches)")
            failed.append(e["name"])
            continue
        texts[e["path"]] = t.replace(e["old"], e["new"], 1)
        print(f"  APPLY    {e['name']}")
        applied += 1

    if failed:
        sys.exit(f"\nABORT, nothing written. {len(failed)} edit(s) unresolved: "
                 + ", ".join(failed)
                 + "\nThe file has drifted from the expected content. Inspect before retrying.")

    for p, t in texts.items():
        try:
            ast.parse(t)
        except SyntaxError as exc:
            sys.exit(f"\nABORT, nothing written. Patched {p} does not parse: {exc}")
    print("\n  ast.parse OK on both patched files")

    if args.dry_run:
        print(f"\nDRY RUN. {applied} would apply, {skipped} already applied. Nothing written.")
        return

    for p, t in texts.items():
        with open(p, "w") as fh:
            fh.write(t)

    print(f"\n{applied} applied, {skipped} already applied.")
    for p in (COLLECT, AUDIT):
        print(f"  {p}\n    before {before[p]}\n    after  {md5(p)}")
    if applied == 0:
        print("\nNo changes. md5 values above should be identical (idempotent).")


if __name__ == "__main__":
    main()
