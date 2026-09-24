from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Why this exists: analysis/make_ce_figures.py carries the C&E truncation means and
# objective contrasts as constants taken from 4 dp log averages. Recomputed from the
# per-seed W&B values at full precision on 2026-09-23 (MRAP benchmark audit), nine
# of them move by 0.0001, and the K=512 scratch tie resolves to 0.6675, the value
# the paper's table prints. Contrast intervals come from paired_bootstrap_objective.py
# (rng seed 0) rerun on the full-precision values.

TARGET = Path("analysis/make_ce_figures.py")
EDITS = [
    ("    20:  (0.6410, 0.6399, 0.6426, 0.6476),", "    20:  (0.6410, 0.6399, 0.6427, 0.6476),", "K=20"),
    ("    40:  (0.6438, 0.6415, 0.6504, 0.6423),", "    40:  (0.6437, 0.6415, 0.6503, 0.6423),", "K=40"),
    ("    80:  (0.6496, 0.6450, 0.6509, 0.6523),", "    80:  (0.6497, 0.6450, 0.6509, 0.6523),", "K=80"),
    ("    512: (0.6941, 0.6920, 0.6655, 0.6676),", "    512: (0.6941, 0.6920, 0.6655, 0.6675),", "K=512 scratch"),
    ('"Junyi":        (  87.0, 1326, -0.0122, -0.0136, -0.0110,',
     '"Junyi":        (  87.0, 1326, -0.0122, -0.0137, -0.0110,', "Junyi"),
    ('"EdNet":        (  30.0,  142, -0.0069, -0.0078, -0.0055,',
     '"EdNet":        (  30.0,  142, -0.0068, -0.0078, -0.0054,', "EdNet"),
    ('"Bridge 2006":  (1373.0,  492, +0.0105, +0.0093, +0.0116,',
     '"Bridge 2006":  (1373.0,  492, +0.0106, +0.0093, +0.0116,', "Bridge 2006"),
    ('"Algebra 2006": (1168.5,  484, +0.0186, +0.0148, +0.0228,',
     '"Algebra 2006": (1168.5,  484, +0.0186, +0.0148, +0.0227,', "Algebra 2006"),
    ('"ASSIST 2017":  ( 441.0,  102, +0.0240, +0.0222, +0.0261,',
     '"ASSIST 2017":  ( 441.0,  102, +0.0240, +0.0222, +0.0262,', "ASSIST 2017"),
]


def apply(text: str, old: str, new: str, why: str) -> str:
    if new in text and old not in text:
        print(f"skip (already applied): {why}")
        return text
    n = text.count(old)
    if n != 1:
        sys.exit(f"ABORT ({why}): anchor matched {n} times, expected 1")
    print(f"ok: {why}")
    return text.replace(old, new, 1)


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    out = src
    for old, new, why in EDITS:
        out = apply(out, old, new, why)
    if out != src:
        TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
