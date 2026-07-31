#!/usr/bin/env python3
"""Correct the truncation-control claim in RESULTS.md section 5.

The recorded line says the scratch control "rises smoothly with K" and puts the
crossover near pps 1.5. The sweep in the same section says otherwise:

  K:      10      20      40      80     160     320     512
  gap: -0.0121 -0.0027 -0.0089 -0.0059 -0.0046 -0.0010 +0.0265   (skill_only - correct_only)
  scr:  0.6347  0.6476  0.6423  0.6523  0.6586  0.6597       -

The gap is negative at every tested K through 320 (pps 3.14) and positive only at
K=512 (pps 5.02), so the sign change is bracketed between pps 3.14 and 5.02, not
near 1.5. At K=160, which is pps 1.57, the gap is still -0.0046. The scratch
column is also not monotone (it falls from 0.6476 at K=20 to 0.6423 at K=40) and
has no K=512 entry, so the longest endpoint is uncontrolled.

Run from the repo root:  python3 tools/patches/fix_crossover_claim.py
Idempotent.
"""
import re
import sys

PATH = "RESULTS.md"
OLD = "- Scratch control rises smoothly with K (rules out pure data-quantity). Crossover ~pps 1.5."
NEW = ("- A from-scratch control was run to K=320 (0.6347, 0.6476, 0.6423, 0.6523, 0.6586, 0.6597 "
       "at K=10..320). It is not monotone, it does not reproduce the endpoint reversal, and there "
       "is no scratch run at K=512, so the longest endpoint is uncontrolled. The skill_only minus "
       "correct_only gap is negative at every tested K through 320 (pps 3.14) and positive only at "
       "K=512 (pps 5.02): the sign change is bracketed between pps 3.14 and 5.02, NOT near 1.5 "
       "(at K=160, pps 1.57, the gap is still -0.0046). Truncation retains each learner's most "
       "recent K interactions, so it lowers total interactions, time horizon and skill composition "
       "together; density is not isolated.")

txt = open(PATH).read()
if "the sign change is bracketed between pps 3.14 and 5.02" in txt:
    sys.exit("ALREADY FIXED. Nothing done.")
if OLD not in txt:
    sys.exit(f"ERROR: anchor line not found verbatim in {PATH}.\n"
             "Check by hand: grep -n 'Crossover' RESULTS.md")

txt = txt.replace(OLD, NEW)
open(PATH, "w").write(txt)
print("PATCHED section 5: crossover claim replaced with the bracketed sign change.")

left = re.findall(r"[Cc]rossover\s*~?\s*pps\s*1\.5", txt)
print("remaining 'crossover ~pps 1.5' mentions:", len(left))
if left:
    print("  fix these by hand: grep -n 'rossover' RESULTS.md")
print("Verify: grep -n 'sign change is bracketed' RESULTS.md")
