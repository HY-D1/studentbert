#!/usr/bin/env python3
"""Correct the empty-pps-gap bounds in RESULTS.md.

Two lines state the empty gap as 0.33-2.79. That predates Algebra2006-07, which
sits at pps 2.410 and is listed in the section 5 dataset table, i.e. inside the
span those lines call empty. Sorted pps: junyi 0.066, ednet 0.211, assist2009
0.325, algebra2006 2.410, bridge2006 2.790, assist2017 4.330, algebra2005 5.330.
The largest gap with no dataset in it is therefore 0.33 to 2.41.

The poster already states 0.33-2.41; this brings RESULTS.md into line.

Run from the repo root:  python3 tools/patches/fix_pps_gap_text.py
Idempotent.
"""
import sys

PATH = "RESULTS.md"

SUBS = [
 ("**Honest limit:** precise threshold fuzzy (empty gap 0.33-2.79).",
  "**Honest limit:** precise threshold fuzzy (empty gap 0.33-2.41; Algebra2006-07 at pps 2.410 "
  "sits inside the originally stated 0.33-2.79 span and closed its upper part)."),
 ("- Regime threshold is fuzzy (empty pps gap 0.33-2.79); pps ordering holds, exact boundary not localized.",
  "- Regime threshold is fuzzy (empty pps gap 0.33-2.41, no dataset between assist2009 at 0.325 and "
  "algebra2006 at 2.410); pps ordering holds, exact boundary not localized."),
]

txt = open(PATH).read()
# Guard on a phrase that exists ONLY in the replacement. Checking "0.33-2.79"
# would never fire (the new text quotes it), and a prefix slice of the new text
# is unsafe because old and new share their first 60 characters.
SENTINEL = "sits inside the originally stated"
if SENTINEL in txt:
    sys.exit("ALREADY FIXED. Nothing done.")

done = 0
for old, new in SUBS:
    n = txt.count(old)
    if n == 1:
        txt = txt.replace(old, new)
        done += 1
    elif n > 1:
        sys.exit(f"ERROR: {old[:50]!r} appears {n} times; fix by hand")

open(PATH, "w").write(txt)
print(f"FIXED {done}/{len(SUBS)} occurrence(s). The one surviving '0.33-2.79' is the "
      "deliberate back-reference inside the corrected sentence.")
print("Verify: grep -n '0.33-2.41' RESULTS.md")
