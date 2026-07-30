#!/usr/bin/env python3
"""Add section 3.2 (paired-by-seed next-skill gaps) to RESULTS.md.

Runs analysis/paired_bootstrap_lak.py and inserts its table verbatim, so every
number comes from nextskill_results_long.csv at run time and nothing is
transcribed by hand.

Run from the repo root:  python3 tools/patches/add_paired_seed_section.py
Idempotent. Requires nextskill_results_long.csv in the current directory.
"""
import os
import subprocess
import sys

PATH = "RESULTS.md"
ANALYSIS = "analysis/paired_bootstrap_lak.py"
CSV = "nextskill_results_long.csv"

for f in (PATH, ANALYSIS, CSV):
    if not os.path.exists(f):
        sys.exit(f"ERROR: {f} not found. Run from the repo root; regenerate the CSV with\n"
                 "  python analysis/parse_nextskill_full.py --dir .")

txt = open(PATH).read()
if "### 3.2 Paired-by-seed next-skill gaps" in txt:
    sys.exit("ALREADY PATCHED. Nothing done.")

out = subprocess.run([sys.executable, ANALYSIS, "--csv", CSV,
                      "--dataset", "assist2017", "--metric", "top1"],
                     capture_output=True, text=True)
if out.returncode != 0:
    sys.exit(f"ERROR running {ANALYSIS}:\n{out.stderr}")

body = out.stdout.strip()
for marker in ("| N | condition | seeds |", "CAVEAT, n="):
    if marker not in body:
        sys.exit(f"ERROR: expected {marker!r} in the analysis output; got:\n{body[:400]}")

# strip the script's own heading line; this section supplies its own
lines = [l for l in body.split("\n") if not l.startswith("#### ")]
table = "\n".join(lines).strip()

BLOCK = f"""
### 3.2 Paired-by-seed next-skill gaps, assist2017 target (top-1, recomputed from nextskill_results_long.csv)

Same runs as 3.1, but each pretrained condition is differenced against scratch at the
same N *within* a seed, since a seed fixes the target subsample (`first_n_students`
draws per seed). Regenerate with `python3 analysis/paired_bootstrap_lak.py`.

{table}

_Read: in-domain and Junyi sources are positive on 3/3 seeds at every budget. The
EdNet source is not: 2/3 seeds at N=25 with a per-seed spread of -0.0058 to +0.0852,
and 1/3 seeds at both N=50 and N=100 where its mean gap is negative (-0.0046 and
-0.0015). It only becomes consistently positive from N=200. So the largest source is
not merely noisy at small N, it is the wrong default there; low-N guidance should name
in-domain first, then Junyi. The gap is also not monotone in N: it collapses from
+0.092 at N=25 to +0.006 at N=100, then recovers to +0.012 to +0.015 by N=1000, with
tight per-seed spread at the high end, so the recovery is not seed noise. At n=3 seeds
these are per-seed evidence, not significance tests; see the caveat above._
"""

anchor = "\n## 4. "
assert txt.count(anchor) == 1, f"anchor {anchor!r} not unique; check: grep -n '^## ' {PATH}"
open(PATH, "w").write(txt.replace(anchor, BLOCK + anchor))
print("PATCHED. Inserted 3.2 before section 4.")
print("Verify: grep -n '### 3.2' RESULTS.md")
