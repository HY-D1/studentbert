from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Why this exists: slurm/generators/gen_budget_control.sh empties its queue folders and rewrites
# every job whose encoder or result does not exist yet. Re-run while earlier jobs were submitted
# but not started (2026-10-09), it rewrote three pretraining jobs that drip_multi had already
# submitted, so the drip loop would have submitted them twice, and it would do the same to every
# fine-tune on its next run. tools/drip_multi.sh and drip_submit.sh move each submitted file into
# the queue's done/ folder, so a job whose file is in done/ has been submitted: this patch makes
# the generator skip those, in both the pretraining and the fine-tune section, and say how to
# resubmit one that failed (delete its file from done/). Exact anchors, all checked before any
# write; idempotent. Harry, 2026-10-09.

TARGET = Path("slurm/generators/gen_budget_control.sh")
PRE = "c19a664351218aa63d46fd41c97eafcc"

EDITS = [
    ("pcount=0; have=0; inprogress=0\n",
     "pcount=0; have=0; inprogress=0; queued=0\n",
     "pretraining counters"),
    ("    CK=\"../checkpoints/edubert_ednet_pretrain_ednet_${TAG}_encoder.pt\"\n"
     "    if [ -f \"$CODE/$CK\" ]; then\n",
     "    CK=\"../checkpoints/edubert_ednet_pretrain_ednet_${TAG}_encoder.pt\"\n"
     "    if [ ! -f \"$CODE/$CK\" ] && [ -f \"$PQDIR/done/budget_pretrain_ednet_${TAG}.sbatch\" ]; then\n"
     "      echo \"submitted earlier, not started yet: budget_pretrain_ednet_${TAG}\"\n"
     "      queued=$((queued + 1))\n"
     "      continue\n"
     "    fi\n"
     "    if [ -f \"$CODE/$CK\" ]; then\n",
     "pretraining: skip jobs already submitted"),
    ("fcount=0; waiting=0\n",
     "fcount=0; waiting=0; fsub=0\n",
     "fine-tune counters"),
    ("        NAME=\"kt_${TOK}_fromednet_${TAG}_bud_n${N}_seed${SEED}\"\n",
     "        NAME=\"kt_${TOK}_fromednet_${TAG}_bud_n${N}_seed${SEED}\"\n"
     "        if [ -f \"$FQDIR/done/${NAME}.sbatch\" ]; then\n"
     "          fsub=$((fsub + 1))\n"
     "          continue\n"
     "        fi\n",
     "fine-tunes: skip jobs already submitted"),
    ("echo \"encoders to run      : $pcount   (finished: $have, in progress: $inprogress)\"\n"
     "echo \"fine-tune jobs       : $fcount   (encoders not finished yet: $waiting)\"\n",
     "echo \"encoders to run      : $pcount   (finished: $have, in progress: $inprogress, \"\\\n"
     "\"submitted earlier: $queued)\"\n"
     "echo \"fine-tune jobs       : $fcount   (encoders not finished yet: $waiting, \"\\\n"
     "\"submitted earlier: $fsub)\"\n"
     "echo \"a job whose file is in a queue's done/ folder counts as submitted; to resubmit one\"\n"
     "echo \"that failed, delete its file from done/ and re-run this generator\"\n",
     "summary lines"),
]


def apply(text: str, old: str, new: str, why: str) -> tuple[str, bool]:
    if new in text and (old not in text or old in new):
        print(f"skip (already applied): {why}")
        return text, False
    n = text.count(old)
    if n != 1:
        sys.exit(f"ABORT ({why}): anchor matched {n} times, expected 1; nothing written")
    print(f"ok: {why}")
    return text.replace(old, new, 1), True


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    text, did = src, []
    for old, new, why in EDITS:
        text, d = apply(text, old, new, why)
        did.append(d)
    if any(did) and not all(did):
        sys.exit("ABORT: the generator is partly patched; nothing written")
    if all(did) and hashlib.md5(src.encode()).hexdigest() != PRE:
        sys.exit(f"ABORT: {TARGET} is not the file this patch was written for ({PRE}); nothing written")
    if text != src:
        TARGET.write_text(text, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
