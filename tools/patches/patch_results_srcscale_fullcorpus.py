"""Update RESULTS.md section 11 with the three-draw full-corpus point.

The 353,597 row was written from a single encoder. Two further encoders were
built (jobs 10410464 and 10410468) and 24 transfer runs collected (jobs
10418184 to 10418706, all COMPLETED). Both target gains move DOWN, because the
pre-existing encoder turned out to be the weakest of the three on both targets.

Run from the repo root:
    PYTHONPATH=. python tools/patches/patch_results_srcscale_fullcorpus.py
"""

from __future__ import annotations

import pathlib
import sys

RESULTS = pathlib.Path("RESULTS.md")

OLD_ROW = "| 353,597 | 2.8784 (one draw only) | +0.0260 | +0.0064 | n/a |"
NEW_ROW = "| 353,597 | 2.8784 / 2.8872 / 2.8740 | +0.0250 | +0.0055 | 0.0034 |"

OLD_PERDRAW = """Per-draw gains at 150,000, assist2017: +0.0233 / +0.0245 / +0.0243.
Per-draw gains at 150,000, junyi: +0.0043 / +0.0042 / +0.0044."""

NEW_PERDRAW = """Per-draw gains at 150,000, assist2017: +0.0233 / +0.0245 / +0.0243.
Per-draw gains at 150,000, junyi: +0.0043 / +0.0042 / +0.0044.
Per-draw gains at 353,597, assist2017: +0.0233 / +0.0250 / +0.0267.
Per-draw gains at 353,597, junyi: +0.0043 / +0.0061 / +0.0060.
All eighteen 353,597 draw-by-target cells are positive on 6 of 6 seeds and
every per-draw bootstrap interval excludes zero.

THE 353,597 ROW WAS REVISED DOWNWARD ON 2026-09-18. It previously read +0.0260
and +0.0064 from a single encoder, `edubert_ednet_pretrain_full_encoder.pt`
(loss 2.8784, log `pretrain_ednet_7744629.log`, built 2026-06-20). Two further
encoders were built under the section-11 recipe, `..._n353597d1` (loss 2.8872,
job 10410464) and `..._n353597d2` (loss 2.8740, job 10410468), and their 24
transfer runs collected (jobs 10418184 to 10418706, all COMPLETED). The
pre-existing encoder is the WEAKEST of the three on both targets, so the
single-draw figures were low rather than high. Quote +0.0250 and +0.0055.

ONE PROPERTY OF THIS ROW DIFFERS FROM EVERY OTHER SIZE. At full corpus
`--n_students 353597` selects the entire training split, so there is nothing to
subsample and the three draws differ in the pretraining seed ONLY. At every
smaller size a draw varies both the sampled students and the initialization.
The 0.0034 and 0.0019 spreads here therefore bound initialization variance
alone and are not directly comparable to the spreads in the table above. Worth
noting that they are nonetheless the same order of magnitude, which suggests
initialization variance rather than sampling variance dominates encoder fit
across the whole ladder."""

OLD_READ_TAIL = """CAVEAT: the 353,597 row rests on ONE encoder and cannot get
further draws without re-pretraining the whole corpus._"""

NEW_READ_TAIL = """(c) The curve SATURATES at the top. Going from
150,000 to the full 353,597 students, a 2.4-fold increase in corpus, buys
+0.0010 on assist2017 (+0.0240 to +0.0250) and +0.0012 on junyi (+0.0043 to
+0.0055). Both are smaller than the draw spread at the larger size, 0.0034 and
0.0019, so the final step of the curve is NOT resolvable at three draws. The
defensible statement is that transfer rises steeply from 5,000 to 49,153,
continues to 150,000, and then flattens; do not read the top of the curve as
still climbing. This is the source-side analogue of the target-side saturation
in section 4._"""


def main() -> None:
    if not RESULTS.exists():
        sys.exit("ABORT: RESULTS.md not found. Run from the repo root.")
    s = RESULTS.read_text()

    if NEW_ROW in s and "REVISED DOWNWARD ON 2026-09-18" in s:
        print("already applied; nothing to do")
        return

    for name, old in (("table row", OLD_ROW),
                      ("per-draw block", OLD_PERDRAW),
                      ("read-line caveat", OLD_READ_TAIL)):
        n = s.count(old)
        if n != 1:
            sys.exit("ABORT (%s): matched %d, expected 1. RESULTS.md NOT modified."
                     % (name, n))

    s = s.replace(OLD_ROW, NEW_ROW, 1)
    s = s.replace(OLD_PERDRAW, NEW_PERDRAW, 1)
    s = s.replace(OLD_READ_TAIL, NEW_READ_TAIL, 1)
    RESULTS.write_text(s)

    print("ok: 353,597 row  +0.0260/+0.0064 -> +0.0250/+0.0055, losses and spread added")
    print("ok: per-draw gains and the downward-revision provenance recorded")
    print("ok: one-encoder caveat replaced by the saturation reading")

    s2 = RESULTS.read_text()
    print()
    for probe, want in (("+0.0260", 0), ("+0.0064", 0), ("+0.0250", 1),
                        ("+0.0055", 1), ("one draw only", 0),
                        ("2.8872", 1), ("2.8740", 1)):
        got = s2.count(probe)
        print("  %-16s %d  %s" % (probe, got, "ok" if got == want else "CHECK"))
    print("\n  sections: %d   end marker: %d (must be 1)"
          % (s2.count("\n## "), s2.count("_End of consolidated results._")))


if __name__ == "__main__":
    main()
