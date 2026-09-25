from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Why this exists: on 2026-09-24 `git status --short` on the cluster listed three files that are
# untracked by design (hand-off 2026-09-24c, section 1): the final-build zip, drip_multi's lock
# file and a dated audit report. Listing them keeps the status clean, so a real untracked change
# stands out.

TARGET = Path(".gitignore")
OLD = "tg1_exposure_*.jsonl\nexposure_*/\n"
NEW = (OLD + "# untracked by design: final-build zips, drip_multi lock, dated audit reports\n"
       "benchmark_*.zip\n.drip_multi.lock\naudit_20*.txt\n")


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
    out = apply(src, OLD, NEW, "ignore benchmark zips, drip lock, dated audits")
    if out != src:
        TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
