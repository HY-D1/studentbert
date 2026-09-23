from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Why this exists: tools/drip_submit.sh promises "DRYRUN=1 ... show, submit nothing", but its
# dry branch moved every file into done/, so the real run that follows finds an empty queue.
# That silently dropped the three tg1 LogME jobs on 2026-09-22. The dry branch now lists the
# queue and exits; nothing moves.

TARGET = Path("tools/drip_submit.sh")
OLD = (
    '    if [ "$DRYRUN" = "1" ]; then\n'
    '      echo "[dry] would submit $(basename "$f")   (queue has $mine of my jobs)"\n'
    '      mv "$f" "$QDIR/done/"\n'
    '    else\n'
)
NEW = (
    '    if [ "$DRYRUN" = "1" ]; then\n'
    '      for g in "$QDIR"/*.sbatch; do\n'
    '        echo "[dry] would submit $(basename "$g")   (queue has $mine of my jobs)"\n'
    '      done\n'
    '      echo "DRYRUN: queue left untouched, $pending file(s) still pending"\n'
    '      exit 0\n'
    '    else\n'
)


def apply(text: str, old: str, new: str, why: str) -> str:
    # Replacement edits: once applied the old block is gone and the new one present.
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
    out = apply(src, OLD, NEW, "DRYRUN lists the queue and moves nothing")
    if out != src:
        TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
