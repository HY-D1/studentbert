#!/usr/bin/env python3
"""Bring RESULTS.md in line with the verified record, without regenerating it.

WHY NOT REGENERATE
    collect_all_results.py was patched on 2026-08-11, but RESULTS.md was never
    rewritten from it, because regeneration drops every hand-added section
    (0.1, 2.1, 2.2, 3.1, 3.2, 3.3, 6.1, 8.1, 8.2, 10). The corrections therefore
    live in the generator but not in the file it writes, and RESULTS.md is now
    behind the manuscript in five places. This patches the file in place and
    touches nothing else.

WHAT IT CHANGES
    1  section 4  Algebra2006 "first prospective predict-before-test" removed.
                  Git shows the written prediction was committed 2026-07-25
                  18:39:09, after the skill_only logs at 18:22-18:25, so there
                  is no timestamped record separating specification from
                  outcome. Replaced with the seventh-dataset framing.
    2  section 5  K=512 relabelled: the ASSIST2017 median is 441, so K=512
                  truncates nothing and pps is 441/102 = 4.32, not 512/102.
    3  section 5  truncation table gains the K=512 scratch cell, 0.6676
                  (jobs 9081378-9081383, 6 seeds, 1366 train students, 30 ep).
    4  section 5  scratch-control paragraph rewritten: the endpoint is now
                  controlled, correct_only falls below scratch, and the
                  data-quantity explanation is ruled out.
    5  section 5  honest limit: pps gap stated under both definitions, and the
                  unsourced LODO 5/6 and 3/6 replaced by the reproducible 7-fold
                  result from analysis/lodo_regime.py.

IDEMPOTENT
    Each edit tests for its post-patch text first, so a second run is a no-op and
    leaves the md5 unchanged. If an anchor is missing and its replacement is also
    missing, the file has drifted and the script aborts without writing.

USAGE (from the repo root)
    python3 tools/patches/patch_results_verified_aug21.py --dry-run
    python3 tools/patches/patch_results_verified_aug21.py
"""
import argparse
import hashlib
import os
import sys

TARGET = "RESULTS.md"
EDITS = []


def edit(name, old, new):
    EDITS.append((name, old, new))


edit("1 prospective claim withdrawn",
     "- Algebra2006 (prospective): skill-correct +0.0186 CI [+0.0149,+0.0228] "
     "6/6. first prospective predict-before-test to hold (recorded)",
     "- Algebra2006 (seventh and last dataset added): skill-correct +0.0186 CI "
     "[+0.0148,+0.0228] 6/6 (analysis/paired_bootstrap_objective.py). NOT a "
     "preregistered prediction: git shows the written prediction entered the "
     "repo at ca13880 on 2026-07-25 18:39:09, while the skill_only logs finished "
     "18:22-18:25 the same evening, so no timestamped record separates "
     "specification from outcome. Report as an out-of-sample test: the dataset "
     "was prepared and evaluated after the ordering was fixed on the other six. "
     "Do NOT use the words prospective, predict-before-test or preregistration.")

edit("2 K=512 pps relabel",
     "- K=512 (pps~5.0): correct-skill -0.0265 CI [-0.0280,-0.0253] (0/6) -> "
     "ASSIST2017 truncated K=512/pps5.0: skill-driven (correct-skill negative) "
     "(recorded)",
     "- K=512 (UNTRUNCATED, pps 4.32): correct-skill -0.0265 CI "
     "[-0.0280,-0.0253] (0/6) -> ASSIST2017 at K=512 is the untruncated "
     "condition, because the median sequence is 441 < 512, so pps is 441/102 = "
     "4.32 and NOT 512/102 = 5.02 (recorded)")

edit("3 truncation table scratch cell",
     "| 512 | 0.6941 | 0.6920 | 0.6655 | - |",
     "| 512 | 0.6941 | 0.6920 | 0.6655 | 0.6676 |")

edit("4 scratch control paragraph",
     "- A from-scratch control was run to K=320 (0.6347, 0.6476, 0.6423, "
     "0.6523, 0.6586, 0.6597 at K=10..320). It is not monotone, it does not "
     "reproduce the endpoint reversal, and there is no scratch run at K=512, so "
     "the longest endpoint is uncontrolled. The skill_only minus correct_only "
     "gap is negative at every tested K through 320 (pps 3.14) and positive only "
     "at K=512 (pps 5.02): the sign change is bracketed between pps 3.14 and "
     "5.02, NOT near 1.5 (at K=160, pps 1.57, the gap is still -0.0046). "
     "Truncation retains each learner's most recent K interactions, so it lowers "
     "total interactions, time horizon and skill composition together; density "
     "is not isolated.",
     "- The from-scratch control now covers every K including 512 (0.6347, "
     "0.6476, 0.6423, 0.6523, 0.6586, 0.6597, 0.6676 at K=10..512; the K=512 "
     "cell added 2026-08-11, jobs 9081378-9081383, 6 seeds, 1366 train "
     "students, 30 epochs, per-seed 0.6653/0.6684/0.6686/0.6680/0.6667/0.6683). "
     "It is not monotone. From K=320 to K=512 every condition gains the same "
     "additional interactions, but scratch gains only +0.0079 while full gains "
     "+0.0327 and skill_only +0.0331, and correct_only gains +0.0056, less than "
     "scratch; at K=512 correct_only (0.6655) falls BELOW scratch (0.6676). A "
     "generic data-quantity explanation is therefore ruled out, because the "
     "control experiences the same change and does not reproduce the pattern. "
     "The skill_only minus correct_only gap is negative at every tested K "
     "through 320 (pps 3.14) and positive only at K=512 (pps 4.32): the sign "
     "change is bracketed between pps 3.14 and 4.32, NOT near 1.5 (at K=160, "
     "pps 1.57, the gap is still -0.0046). Truncation retains each learner's "
     "most recent K interactions, so it lowers total interactions, time horizon "
     "and skill composition together; the control isolates density from data "
     "quantity but not from every co-varying property.")

edit("5 honest limit, pps gap and LODO",
     "**Honest limit:** precise threshold fuzzy (empty gap 0.33-2.41; "
     "Algebra2006-07 at pps 2.414 sits inside the originally stated 0.33-2.79 "
     "span and closed its upper part). LODO does not generalize at n=6 "
     "(leave-one-dataset-out: multivariate 5/6 but overparameterized at n=6, "
     "not trustworthy; single features n_students 5/6, pps 3/6. No combo "
     "generalizes. NEGATIVE.).",
     "**Honest limit:** precise threshold fuzzy. The empty pps interval is "
     "0.325-2.414 on raw median length and narrows to 0.325-1.041 under the "
     "512-step model cap, which lowers Algebra2005 to 4.697, Bridge2006 to "
     "1.041 and Algebra2006 to 1.058; the ordering and the regime split are "
     "identical under both definitions, so state which one any interval claim "
     "uses. LODO does not generalize (analysis/lodo_regime.py, reproducible, "
     "all 7 folds): n_students 6/7, n_interactions 6/7, n_skills 6/7, pps 5/7 "
     "under both definitions, correct_rate 3/7. Four features separate all 7 "
     "in-sample, so in-sample separation is uninformative; nothing reaches 7/7; "
     "the 2 folds holding out a correctness-driven dataset leave a single such "
     "example in training. NEGATIVE.")


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(TARGET):
        sys.exit(f"ABORT: {TARGET} not found. Run from the repo root.")

    before = md5(TARGET)
    text = open(TARGET).read()
    applied, skipped, failed = 0, 0, []

    for name, old, new in EDITS:
        if new in text:
            print(f"  SKIP  {name} (already applied)")
            skipped += 1
            continue
        n = text.count(old)
        if n != 1:
            print(f"  FAIL  {name} ({n} matches, expected 1)")
            failed.append(name)
            continue
        text = text.replace(old, new, 1)
        print(f"  APPLY {name}")
        applied += 1

    if failed:
        sys.exit("\nABORT, nothing written. Unresolved: " + ", ".join(failed))

    if args.dry_run:
        print(f"\nDRY RUN. {applied} would apply, {skipped} already applied.")
        return

    open(TARGET, "w").write(text)
    print(f"\n{applied} applied, {skipped} already applied.")
    print(f"  before {before}\n  after  {md5(TARGET)}")
    if applied == 0:
        print("\nNo changes. The two md5 values above should match.")


if __name__ == "__main__":
    main()
