from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Why this exists: collect_all_results.py hard-codes a few values and one label that the
# 2026-09-23 full-precision audit corrected in RESULTS.md (see patch_results_full_precision.py).
# The collector refuses to overwrite RESULTS.md while hand-maintained sections exist, but a
# regenerated draft (--out_md RESULTS_generated.md) should not reintroduce the old values.

TARGET = Path("analysis/collect_all_results.py")
EDITS = [
    ('"value": "+0.0186", "ci": "[+0.0149,+0.0228]", "seeds": "6/6",',
     '"value": "+0.0186", "ci": "[+0.0148,+0.0227]", "seeds": "6/6",', "RECORDED algebra2006 CI"),
    ('"value": "+0.0069", "ci": "[+0.0065,+0.0075]", "seeds": "6/6",',
     '"value": "+0.0070", "ci": "[+0.0066,+0.0075]", "seeds": "6/6",', "RECORDED EdNet full-scale gain"),
    ('"(EdNet-source encoders: full / skill_only / correct_only -> target), from logs:\\n")',
     '"(EdNet-source encoders: full / skill_only / correct_only -> target, except the ednet row, "\n'
     '      "whose encoders are Junyi-source), from logs:\\n")', "section 4 label"),
    ('("assist2017","+0.0240","[+0.0222, +0.0261]")', '("assist2017","+0.0240","[+0.0222, +0.0262]")', "contrast assist2017"),
    ('("ednet","-0.0069","[-0.0078, -0.0055]")', '("ednet","-0.0068","[-0.0078, -0.0054]")', "contrast ednet"),
    ('("junyi","-0.0122","[-0.0136, -0.0110]")', '("junyi","-0.0122","[-0.0137, -0.0110]")', "contrast junyi"),
    ('("bridge2006","+0.0105","[+0.0093, +0.0116]")', '("bridge2006","+0.0106","[+0.0093, +0.0116]")', "contrast bridge2006"),
    ('("algebra2006","+0.0186","[+0.0148, +0.0228]")', '("algebra2006","+0.0186","[+0.0148, +0.0227]")', "contrast algebra2006"),
    ("NOTE: ednet skill-correct -0.0069 is NOT the same quantity as the in-domain EdNet full-scale KT gain +0.0069 in section 3.",
     "NOTE: ednet skill-correct -0.0068 and the in-domain EdNet full-scale KT gain +0.0070 in section 3 are different quantities; the contrasts are full-precision reruns of paired_bootstrap_objective.py (2026-09-23).",
     "collision note"),
]


def apply(text: str, old: str, new: str, why: str) -> str:
    if new in text and (old not in text or old in new):
        print(f"skip (already applied): {why}")
        return text
    n = text.count(old)
    if n != 1:
        sys.exit(f"ABORT ({why}): anchor matched {n} times, expected 1; nothing written")
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
