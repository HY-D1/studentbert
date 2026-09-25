from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Why this exists: node d1020 killed two jobs with signal 9 and stalled job 10591596 for 1:03:23
# with zero output (hand-off 2026-09-24c, lesson 10). Every sbatch this generator writes now
# excludes it. Revisit if Research Computing reports the node fixed.

TARGET = Path("slurm/generators/gen_tg1_jobs.sh")
OLD = "    echo '#SBATCH --partition=gpu'\n    echo \"$gres\"\n"
NEW = "    echo '#SBATCH --partition=gpu'\n    echo '#SBATCH --exclude=d1020'\n    echo \"$gres\"\n"


def apply(text: str, old: str, new: str, why: str) -> str:
    if new in text and old not in text:
        print(f"skip (already applied): {why}")
        return text
    n = text.count(old)
    if n != 1:
        sys.exit(f"ABORT ({why}): anchor matched {n} times, expected 1; nothing written")
    print(f"ok: {why}")
    return text.replace(old, new, 1)


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    out = apply(src, OLD, NEW, "exclude node d1020 in every generated sbatch")
    if out != src:
        TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
