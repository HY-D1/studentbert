from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Why this exists: RESULTS.md gains a hand-rendered section 13 (the Task 4 cross-domain check,
# tools/patches/patch_results_section12_7_13.py). The collector's MANUAL list names the sections it
# does not regenerate, so an overwrite would refuse while they exist; section 13 joins the list so
# the list stays complete. Harry, 2026-09-29.

TARGET = Path("analysis/collect_all_results.py")
OLD = '              "### 8.2", "## 10.", "## 11.", "## 12."]\n'
NEW = '              "### 8.2", "## 10.", "## 11.", "## 12.", "## 13."]\n'


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    if NEW in src:
        print("skip (already applied): MANUAL lists ## 13.")
    else:
        n = src.count(OLD)
        if n != 1:
            sys.exit(f"ABORT: MANUAL anchor matched {n} times, expected 1; nothing written")
        TARGET.write_text(src.replace(OLD, NEW, 1), encoding="utf-8")
        print("ok: MANUAL lists ## 13.")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
