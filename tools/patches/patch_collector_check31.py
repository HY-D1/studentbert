from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Why this exists: the carried-over TODO (hand-off 2026-09-22, section 10.C) asked for RESULTS.md
# 3.1 to come from parse_nextskill_full.py output inside the collector. The collector refuses to
# overwrite RESULTS.md while hand-maintained sections exist, so the deliverable is a check mode:
# --check-3-1 <nextskill_results_long.csv> recomputes every 3.1 number (analysis/nextskill_31.py)
# and compares, then exits without collecting or writing anything. Added 2026-09-24.

TARGET = Path("analysis/collect_all_results.py")
OLD = """    args = ap.parse_args()

    # Sections this script does NOT regenerate."""
NEW = """    ap.add_argument("--check-3-1", dest="check_3_1", default=None, metavar="LONG_CSV",
                    help="compare RESULTS.md 3.1 (read from --out_md) with the per-seed "
                         "nextskill_results_long.csv, write nothing, exit 0 on PASS")
    args = ap.parse_args()

    if args.check_3_1:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from nextskill_31 import check as check_3_1
        sys.exit(check_3_1(args.out_md, args.check_3_1))

    # Sections this script does NOT regenerate."""


def apply(text: str, old: str, new: str, why: str) -> str:
    # Insertion-style edit: the anchor's tail survives inside NEW, so decide by containment.
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
    out = apply(src, OLD, NEW, "collector --check-3-1 mode")
    if out != src:
        TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
