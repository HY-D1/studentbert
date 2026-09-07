#!/usr/bin/env python3
# Correct the blocks of RESULTS.md that the 270-job campaign superseded.
#
# Why this exists: RESULTS.md is the shared numbers file for four papers, and
# five of its statements are now known to be wrong or weaker than the data.
# The most consequential is Section 1, where DKT on ASSISTments 2017 is 0.690
# against a measured 0.6974, a difference large enough to flip which model
# leads that row.
#
# Idempotent: each edit is skipped if its replacement is already present, and
# every anchor must match exactly once or NOTHING is written. Run it twice and
# the second run reports "already applied" for all six.
#
#   python3 tools/patches/patch_results_base2.py --file RESULTS.md
#   python3 tools/patches/patch_results_base2.py --file RESULTS.md --dry-run

from __future__ import annotations

import argparse
import hashlib
import sys

EDITS = []


def edit(name, old, new):
    EDITS.append((name, old, new))


edit("section-1 table",
"""Knowledge-tracing test AUC. Columns: DKT / AKT / EduBERT-scratch / EduBERT-pretrained. (Recorded uniform table; EduBERT-pretrained = full-objective number for the 4 newer sets.)

| Dataset | DKT | AKT | scratch | EduBERT-pt |
|---|---|---|---|---|
| ASSIST2017 | 0.690 | 0.650 | 0.670 | **0.693** |
| EdNet | 0.680 | 0.672 | 0.678 | **0.685** |
| Junyi | 0.759 | 0.754 | 0.757 | **0.758** |
| Algebra2005 | 0.798 | 0.776 | 0.781 | **0.787** |
| Bridge2006 | 0.795 | 0.773 | 0.773 | **0.775** |
| ASSIST2009 | 0.876 | 0.863 | 0.870 | **0.870** |
| Algebra2006 | 0.803 | 0.775 | 0.787 | **0.790** |

_Read: DKT is the strongest simple baseline and wins on the 4 newer datasets; the pretraining edge concentrates on the original 3 (esp in-domain EdNet); AKT weakest. Pretraining is regime/scale specific, not universal._

**Log cross-check (newer datasets, parsed from logs):**

| Dataset | DKT (log) | AKT (log) | scratch (log) |
|---|---|---|---|
| algebra2005 | 0.7983 \u00b10.0007 (n=3) | 0.7756 \u00b10.0011 (n=3) | 0.7806 \u00b10.0144 (n=6) |
| bridge2006 | 0.7952 \u00b10.0005 (n=3) | 0.7728 \u00b10.0014 (n=3) | 0.7733 \u00b10.0011 (n=6) |
| assist2009 | 0.8762 \u00b10.0004 (n=3) | 0.8626 \u00b10.0007 (n=3) | 0.8698 \u00b10.0006 (n=6) |
| algebra2006 | 0.8034 \u00b10.0007 (n=3) | 0.7750 \u00b10.0012 (n=3) | 0.7868 \u00b10.0008 (n=6) |""",
"""HAND-PATCHED 2026-09-05 by tools/patches/patch_results_base2.py. `collect_all_results.py` does NOT regenerate this section and WILL overwrite it. Update the collector before running it again, or re-apply this patch afterwards.

Knowledge-tracing test AUC, mean over 6 seeds (42, 1, 2, 3, 4, 5), parsed from logs. Baselines come from the `base2_` campaign: one run per log, budget matched to the EduBERT row on that dataset (ASSIST2017 its full 1,366 training students, EdNet 20,000, Junyi 40,000, other four `--n_students 100000` which is their full split), 30 epochs, best checkpoint by validation. EduBERT-pretrained is in-domain for the original 3 and the EdNet-source full-objective encoder for the 4 newer sets. Final column is pretrained minus scratch paired by seed, 20,000-resample bootstrap CI, then seeds in the direction of the mean.

| Dataset | DKT | AKT | SAINT+ | scratch | EduBERT-pt | pt - scratch (95% CI) |
|---|---|---|---|---|---|---|
| ASSIST2017 | **0.6974** | 0.6519 | 0.6290 | 0.6697 | 0.6927 | +0.0231 [+0.0220,+0.0239] 6/6 |
| EdNet | 0.6799 | 0.6720 | **0.6860** | 0.6777 | 0.6846 | +0.0069 [+0.0065,+0.0075] 6/6 |
| Junyi | 0.7586 | 0.7534 | **0.7984** | 0.7572 | 0.7581 | +0.0010 [+0.0008,+0.0012] 6/6 |
| Algebra2005 | 0.7985 | 0.7756 | **0.8265** | 0.7806 | 0.7873 | +0.0067 [-0.0033,+0.0229] 2/6 |
| Bridge2006 | 0.7947 | 0.7731 | **0.8534** | 0.7733 | 0.7752 | +0.0018 [+0.0003,+0.0034] 4/6 |
| ASSIST2009 | **0.8761** | 0.8625 | 0.8406 | 0.8698 | 0.8699 | +0.0001 [-0.0011,+0.0016] 3/6 |
| Algebra2006 | 0.8028 | 0.7741 | **0.8748** | 0.7868 | 0.7897 | +0.0029 [+0.0016,+0.0044] 6/6 |

_Read: bold is the highest AUC in the row. SAINT+ is highest on 5 of 7, DKT on 2, EduBERT-pretrained on 0. Pretrained beats scratch on all 7 but only 5 of 7 intervals exclude zero: Algebra2005 (2/6 seeds, scratch pstdev 0.0144) and ASSIST2009 do not. AKT is lowest on 6 of 7, SAINT+ is lowest on ASSIST2017. Target budgets differ by row, so rows are comparable within themselves and not with each other. SUPERSEDES the earlier recorded table, in which DKT on ASSIST2017 read 0.690 against a measured 0.6974; that error flipped which model leads the row._

**Agreement with the earlier 3-seed cross-check:** the 12 cells it covered (DKT/AKT/scratch on the 4 newer datasets) reproduce here to within 0.0009, so the two campaigns agree and `base2_` is preferred only because it is 6 seeds, budget matched, and one run per log.

**SAINT+ leakage check, not assumed:** `src/models/saint_plus.py` lines 79-80 build the decoder response stream as START at position 0 and correct[t-1] at position t, with causal masks on encoder, decoder target and memory, so no position sees its own answer.""")

edit("section-2 central finding",
"""- **Central finding (recorded):** large sources (EdNet 442K, Junyi 61K) transfer well everywhere; small source (ASSIST 1.7K) transfers poorly everywhere. On the EdNet target, ASSIST (closest granularity) transfers WORST; Junyi (far granularity) helps more. Validated across 3 targets at budget-matched N=3000, 3 seeds.""",
"""- **Central finding (parsed from logs, 6 seeds):** ranking cross-dataset sources by pretraining corpus size reproduces the observed transfer ordering on all 3 targets. Source sizes are TRAINING-SPLIT students, since pretraining reads the train split only: ASSIST 1,366, Junyi 49,153, EdNet 353,597. On the EdNet target, ASSIST (closest granularity, 102 skills vs 142) gives -0.0004 CI [-0.0016,+0.0009] 2/6, so it does not transfer at all, while Junyi (1,326 skills, far granularity) gives +0.0043 6/6. On the Junyi target ASSIST gives -0.0012 CI [-0.0020,-0.0001] 1/6, so there it costs accuracy. CAVEAT: for the ASSIST2017 target the N=3000 condition is its FULL training split, since 1,366 < 3000, so that column is not a reduced-data condition.""")

edit("section-2 loss claim",
"""- **Loss != transfer (recorded):** Junyi has lower pretraining loss but worse KT transfer than EdNet.""",
"""- **Loss != transfer (parsed from pretrain logs):** best mlm_loss is EdNet 2.8784, Junyi 1.6294, ASSIST2017 1.1236, the exact inverse of the transfer ordering. CAVEAT: these losses are not directly comparable across datasets because the label spaces differ; chance-level skill loss ln(K+1) is 4.63 (ASSIST2017), 4.96 (EdNet), 7.19 (Junyi).""")

edit("section-8 dropout summary",
"""- **Dropout (recorded):** Junyi clean, in-domain best; ASSIST pretraining does NOT help (scratch best); EdNet high-variance/inconclusive at 3 seeds, but 8-seed PAIRED bootstrap found two real effects (in-domain K=5 +0.097 CI[+0.029,+0.164] 6/8; fromJunyi K=10 +0.095 CI[+0.057,+0.132] 8/8). Clean K is dataset-dependent (ASSIST K<=50, EdNet/Junyi K<=10); use --window_censor for high K.""",
"""- **Dropout (recomputed from per-seed logs, 20,000-resample paired bootstrap):** ASSIST2017 pretraining is WORSE than scratch at every clean K, not merely unhelpful. In-domain: -0.0376 [-0.0541,-0.0264] at K=5, -0.0456 [-0.0597,-0.0350] at K=10, -0.0390 [-0.0748,-0.0190] at K=20, -0.0312 [-0.0734,-0.0034] at K=50, all 0/3 seeds positive. Junyi is clean with in-domain best: +0.0371 [+0.0335,+0.0440] at K=5 and +0.0389 [+0.0326,+0.0428] at K=10, both 3/3. EdNet is mostly inconclusive at 8 seeds; only two of six cells clear (in-domain K=5 +0.0968 [+0.0295,+0.1637] 6/8; fromJunyi K=10 +0.0952 [+0.0569,+0.1315] 8/8). Clean K is dataset-dependent (ASSIST K<=50, EdNet/Junyi K<=10); use --window_censor for high K.""")

edit("section-8 next-skill junyi",
"""- **Next-skill, Junyi target (recorded):** frequency-saturated; plain top-1 flat, macro-top1 in-domain +0.007 (best all seeds), top-5 in-domain +0.055.""",
"""- **Next-skill, Junyi target (parsed from logs, n=3):** frequency-saturated; plain top-1 flat (in-domain 0.4903 vs scratch 0.4933), macro-top1 in-domain +0.0070 (0.4925 vs 0.4855), top-5 in-domain +0.0561 (0.8500 vs 0.7939).""")

edit("section-8.1 read line",
"""_Read: pretraining does NOT beat scratch on ASSIST dropout at any clean K (scratch best at K=5/10/20; tied at K=50). This is the quantitative version of the qualitative claim in section 8.""",
"""_Read: pretraining is WORSE than scratch on ASSIST dropout at every clean K, with in-domain intervals excluding zero at K=5/10/20/50 and 0/3 seeds positive each time. Write "produces a worse student-level classifier than random initialization", not "does not help".""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="RESULTS.md")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    raw = open(args.file, encoding="utf-8").read()
    print("before md5 %s" % hashlib.md5(raw.encode("utf-8")).hexdigest())

    plan, applied, skipped = raw, [], []
    for name, old, new in EDITS:
        if new in plan:
            skipped.append(name)
            continue
        n = plan.count(old)
        if n != 1:
            sys.exit("ABORT, no write performed. Anchor %r matched %d times, expected 1."
                     % (name, n))
        plan = plan.replace(old, new)
        applied.append(name)

    for name in applied:
        print("  apply   %s" % name)
    for name in skipped:
        print("  already %s" % name)

    if args.dry_run:
        print("dry run, nothing written")
        return
    if not applied:
        print("nothing to do")
        return
    open(args.file, "w", encoding="utf-8").write(plan)
    print("after  md5 %s" % hashlib.md5(plan.encode("utf-8")).hexdigest())
    print("wrote %s, %d edits applied" % (args.file, len(applied)))


if __name__ == "__main__":
    main()
