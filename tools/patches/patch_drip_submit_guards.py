from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path

# Why this exists: "never run two loops over the same queue directory" was a comment, not a
# guard, and on 2026-09-22 a second loop was started on queue_tg1_objdraws_ft twice. Two loops
# can pick the same file between `ls | head -1` and `mv`, submitting it twice. The loop now
# takes an exclusive flock on the queue directory and a second loop exits at once. Dry runs
# skip the lock so a queue can always be inspected. The file is replaced atomically (new inode)
# because bash reads a running script by offset: rewriting it in place could corrupt a loop
# that is already running. Second fix: a dry run only listed the queue when there was spare
# capacity, so at a full cap (a running campaign) it waited forever; it now lists and exits
# before the submit loop.

TARGET = Path("tools/drip_submit.sh")
OLD = 'mkdir -p "$QDIR/done"\n'
NEW = (
    'mkdir -p "$QDIR/done"\n'
    'if [ "$DRYRUN" != "1" ] && command -v flock >/dev/null 2>&1; then\n'
    '  exec 9>"$QDIR/.drip.lock"\n'
    '  if ! flock -n 9; then\n'
    '    echo "another drip_submit loop already owns $QDIR; exiting, nothing submitted"\n'
    '    exit 1\n'
    '  fi\n'
    'fi\n'
)


OLD2 = '[ "$DRYRUN" = "1" ] && echo "DRYRUN: nothing will actually be submitted"\necho\n'
NEW2 = (
    OLD2
    + 'if [ "$DRYRUN" = "1" ]; then\n'
    + '  for g in "$QDIR"/*.sbatch; do echo "[dry] would submit $(basename "$g")"; done\n'
    + '  echo "DRYRUN: queue left untouched, $total file(s) pending"\n'
    + "  exit 0\n"
    + "fi\n"
)


def apply(text: str, old: str, new: str, why: str) -> str:
    # Insertion edit: the anchor survives, so decide by containment direction.
    if new in text and (old not in text or old in new):
        print(f"skip (already applied): {why}")
        return text
    n = text.count(old)
    if n != 1:
        sys.exit(f"ABORT ({why}): anchor matched {n} times, expected 1")
    print(f"ok: {why}")
    return text.replace(old, new, 1)


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    out = apply(src, OLD, NEW, "one drip loop per queue directory")
    out = apply(out, OLD2, NEW2, "dry run lists and exits even at a full cap")
    if out != src:
        mode = TARGET.stat().st_mode
        fd, tmp = tempfile.mkstemp(dir=TARGET.parent, prefix=".drip_submit.")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(out)
        os.chmod(tmp, mode)
        os.replace(tmp, TARGET)
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
