from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Why this exists: patch_exposure_split.py sized exposure walltimes by dataset, on the assumption
# that a validation draw (about a tenth of the learners) scores in about a tenth of the time. That
# is wrong for the Task 2 scorer: both scorers cap features at 50,000 positions, and a validation
# draw of long-sequence learners (Algebra 2006, Bridge 2006, ASSISTments 2017) still reaches the
# cap, so H-score and NLEEP fit on as many positions as before. Walltimes are now set per scorer
# from the measured runs (sacct, 2026-09-22 to 24): logme7 1:48 to 4:23, so 1 hour everywhere;
# task2feat 11:44 to 18:30 on six targets, so 3 hours; Algebra 2006 task2feat 54:18, so 6 hours,
# the limit it completed under (10 times would exceed the 8-hour partition maximum).
# The queue had not been submitted when this was written. Harry, 2026-09-24.

TARGET = Path("slurm/generators/gen_tg1_jobs.sh")
EDITS = [
    ('''    case "$DS" in
      ednet|junyi) WALL=03:00:00 ;;
      *) WALL=01:00:00 ;;
    esac
''',
     '''    # Task 2 walltime from measured task2feat runs; its feature cap (50,000 positions) means a
    # validation draw is not proportionally faster. LogME measured under 5 minutes everywhere.
    case "$DS" in
      algebra2006) WALL_T2=06:00:00 ;;
      *) WALL_T2=03:00:00 ;;
    esac
''',
     "exposure: walltime per scorer"),
    ('emit "$Q" "tg1_exposure_${P}_logme_${DS}" "tg1_exposure_${P}_logme_${DS}" "$GRES" "$WALL" 32G',
     'emit "$Q" "tg1_exposure_${P}_logme_${DS}" "tg1_exposure_${P}_logme_${DS}" "$GRES" 01:00:00 32G',
     "exposure: LogME 1 hour"),
    ('emit "$Q" "tg1_exposure_${P}_task2_${DS}" "tg1_exposure_${P}_task2_${DS}" "$GRES" "$WALL" 48G',
     'emit "$Q" "tg1_exposure_${P}_task2_${DS}" "tg1_exposure_${P}_task2_${DS}" "$GRES" "$WALL_T2" 48G',
     "exposure: Task 2 3 or 6 hours"),
]


def apply(text: str, old: str, new: str, why: str) -> str:
    # Insertion-style edits keep their anchor, so "old in text" stays true after patching;
    # deciding by containment direction keeps re-runs as no-ops.
    if new in text and (old not in text or old in new):
        print(f"skip (already applied): {why}")
        return text
    n = text.count(old)
    if n != 1:
        sys.exit(f"ABORT ({why}): anchor matched {n} times, expected 1; nothing written "
                 "(is patch_exposure_split.py applied?)")
    print(f"ok: {why}")
    return text.replace(old, new, 1)


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    out = src
    for old, new, why in EDITS:
        out = apply(out, old, new, why)
    if out != src:
        TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
