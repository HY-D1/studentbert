#!/usr/bin/env python3
"""Add section 8.2 (EdNet dropout n3000, corrected full 8-seed grid) to RESULTS.md.
Run from the repo root: python3 tools/patches/patch_results_ednet_dropout.py

Idempotent: refuses to run twice. Numbers recomputed here from per-seed values
(comma-anchored extraction, July 30 2026; indomain k10 seed42 recovered from the
W&B run output.log). Asserts the two recorded headline paired effects reproduce
before writing anything.
"""
from statistics import mean, pstdev
import sys, os

PATH = "RESULTS.md"
SEEDS = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s42"]

# test AUC per seed; None = run died before eval (no banner in any log, not on W&B)
GRID = {
 ("scratch", 5):    [0.6623, 0.5238, 0.6342, 0.4995, 0.5000, 0.5185, 0.5196, 0.7029],
 ("scratch", 10):   [0.5124, 0.7183, 0.5180, 0.5977, 0.5264, 0.6752, 0.6348, 0.5378],
 ("indomain", 5):   [0.6949, 0.6970, 0.6118, 0.5172, 0.7156, 0.7110, 0.7047, 0.6834],
 ("indomain", 10):  [0.5802, None,   0.6817, 0.6852, 0.7273, 0.5861, 0.5405, 0.7261],
 ("fromassist", 5): [0.5164, 0.6053, 0.5000, 0.5218, 0.6753, 0.5364, 0.5130, 0.5122],
 ("fromassist", 10):[0.5229, 0.6912, 0.5043, 0.7367, 0.6113, 0.6456, 0.6081, 0.5463],
 ("fromjunyi", 5):  [0.5077, 0.7025, 0.5128, 0.6132, 0.5371, 0.5230, 0.6797, 0.6666],
 ("fromjunyi", 10): [0.6459, 0.7214, 0.6846, 0.7311, 0.5719, 0.7302, 0.7106, 0.6867],
}

def paired(pre, k):
    g = [p - s for p, s in zip(GRID[(pre, k)], GRID[("scratch", k)]) if p is not None]
    return mean(g), sum(x > 0 for x in g), len(g)

m5, p5, n5 = paired("indomain", 5)
m10, p10, n10 = paired("fromjunyi", 10)
assert round(m5, 3) == 0.097 and n5 == 8, (m5, n5)          # recorded +0.097
assert round(m10, 3) == 0.095 and p10 == 8, (m10, p10)      # recorded +0.095, 8/8
mi10, pi10, ni10 = paired("indomain", 10)

rows = []
for (c, k), v in GRID.items():
    vv = [x for x in v if x is not None]
    cells = " | ".join(f"{x:.4f}" if x is not None else "died" for x in v)
    rows.append(f"| {c} | {k} | {cells} | {mean(vv):.4f} ±{pstdev(vv):.4f} (n={len(vv)}) |")

BLOCK = f"""
### 8.2 EdNet dropout at n3000: corrected full 8-seed grid

Runs `edubert_ednet_drop_ednet_{{scratch|indomain|fromassist|fromjunyi}}_k{{5,10}}_n3000_seed{{1..7,42}}`, test AUC. Extraction MUST be comma-anchored (`grep -A4 -F "=== dropout (NAME,"`): a bare-name grep collides seed4 with seed42 (prefix) and silently duplicates values; 7 seed4 cells below were corrected this way on July 30 2026. indomain k10 seed42 = 0.7261 recovered from the W&B run output.log (run completed and synced; the local log lost its tail) - W&B and local logs agree exactly on the junyi indomain k10 seed2/seed42 cross-checks (0.6240 / 0.6259). indomain k10 seed2 died before eval (no banner locally, not on W&B): cell is n=7; a resubmit is optional and changes no claim.

| cond | K | {' | '.join(SEEDS)} | mean ±pstdev |
|---|---|{'---|' * 9}
{chr(10).join(rows)}

Paired-by-seed effects vs scratch on this corrected grid, matching the recorded 8-seed paired-bootstrap results exactly (so the original analysis used correct extraction):
- indomain k5: mean {m5:+.4f} ({p5}/8 seeds positive) - recorded +0.097, CI [+0.029, +0.164].
- fromjunyi k10: mean {m10:+.4f} ({p10}/8 positive) - recorded +0.095, CI [+0.057, +0.132].
- indomain k10 (n={ni10} paired): mean {mi10:+.4f}, per-seed range -0.094 to +0.201 - high variance, NO claim.
- All other cells: high variance, no claims beyond the two effects above. EdNet dropout remains mostly inconclusive; report it that way.
"""

txt = open(PATH).read()
if "### 8.2 EdNet dropout" in txt:
    sys.exit("ALREADY PATCHED ('### 8.2 EdNet dropout' present). Nothing done.")
anchor = "\n## 9. "
assert txt.count(anchor) == 1, f"anchor {anchor!r} not unique; run: grep -n '^## ' {PATH}"
open(PATH, "w").write(txt.replace(anchor, BLOCK + anchor))
print("PATCHED. Inserted 8.2 before section 9.")
print(f"  indomain k5 paired mean {m5:+.4f} ({p5}/8 pos) | fromjunyi k10 {m10:+.4f} ({p10}/8 pos) | indomain k10 n={ni10} mean {mi10:+.4f}")
