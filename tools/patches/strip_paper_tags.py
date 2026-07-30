#!/usr/bin/env python3
"""Remove the venue/poster annotations from RESULTS.md sections 2.1, 6.1 and 8.2.

RESULTS.md records numbers and what they show; which paper a result is headed
for is tracked separately, and no other section in the file carries such a tag.
Sections 2.1/6.1/8.2 were added with tags by mistake; this strips them.

Run from the repo root:  python3 tools/patches/strip_paper_tags.py
Idempotent: exits cleanly if the tags are already gone.
"""
import sys

PATH = "RESULTS.md"

SUBS = [
 ("### 2.1 Verified source table at N=3000 (KT test AUC, parsed from logs) [EDM; poster R2]",
  "### 2.1 Source comparison at N=3000, all 3 targets (KT test AUC, parsed from logs)"),
 ("Quantitative backing for the scale>granularity claim and the poster R2 card._",
  "Quantitative backing for the scale-over-granularity claim in section 2._"),
 ("### 6.1 LogME vs the domain probe (w7_logme_8263344.log, scripts/compute_logme.py) "
  "[placement: ICLR or NeurIPS, advisor decision pending]",
  "### 6.1 LogME vs the domain probe (w7_logme_8263344.log, scripts/compute_logme.py)"),
 ("### 8.2 EdNet dropout at n3000: corrected full 8-seed grid [LAK]",
  "### 8.2 EdNet dropout at n3000: corrected full 8-seed grid"),
]

txt = open(PATH).read()
done = 0
for old, new in SUBS:
    n = txt.count(old)
    if n == 1:
        txt = txt.replace(old, new)
        done += 1
    elif n > 1:
        sys.exit(f"ERROR: {old[:60]!r} appears {n} times; fix by hand")

if done == 0:
    sys.exit("ALREADY CLEAN (no tagged headings found). Nothing done.")

open(PATH, "w").write(txt)
print(f"STRIPPED {done}/{len(SUBS)} annotation(s) from {PATH}.")

left = [t for t in ("[EDM", "[LAK", "[NeurIPS", "[ICLR") if t in txt]
print("remaining venue tags in file:", left if left else "none")
print("Verify: grep -n '^### 2.1\\|^### 6.1\\|^### 8.2' RESULTS.md")
