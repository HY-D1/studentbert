from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Why this exists: gen_tg1_jobs.sh gave the objdraws fine-tunes walltimes of 1 to 6 hours, but
# they run in 2 to 15 minutes. On 2026-09-23 six-hour Bridge 2006 requests could not backfill
# before a node reservation and sat pending for over an hour with the 7-job cap full, blocking
# every queue behind them. One hour is still about 4 times the slowest run (Junyi at N=1000).

TARGET = Path("slurm/generators/gen_tg1_jobs.sh")
EDITS = [
    ("junyi) BUD=n1000; NS=1000; WALL=04:00:00 ;;", "junyi) BUD=n1000; NS=1000; WALL=01:00:00 ;;", "junyi"),
    ("bridge2006) BUD=full; NS=100000; WALL=06:00:00 ;;", "bridge2006) BUD=full; NS=100000; WALL=01:00:00 ;;", "bridge2006"),
    ("algebra2006) BUD=full; NS=100000; WALL=05:00:00 ;;", "algebra2006) BUD=full; NS=100000; WALL=01:00:00 ;;", "algebra2006"),
    ("*) BUD=full; NS=100000; WALL=03:00:00 ;;", "*) BUD=full; NS=100000; WALL=01:00:00 ;;", "algebra2005 and assist2009"),
]


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
    out = src
    for old, new, why in EDITS:
        out = apply(out, old, new, why)
    if out != src:
        TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
