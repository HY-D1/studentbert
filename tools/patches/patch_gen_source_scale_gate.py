"""Fix the encoder gate in slurm/generators/gen_source_scale.sh.

THE BUG. The gate is a pure file-existence test, `if [ -f "$CODE/$CK" ]`. But
pretrain_edubert.py writes a checkpoint on EVERY loss improvement, so the file
appears minutes into a run and stays there for the rest of it. On 2026-09-15
that let 24 fine-tune jobs load 150,000-student encoders about 70 minutes
before their pretraining jobs ended, producing a spurious draw spread of 0.0175
that collapsed to 0.0012 once the runs were repeated against finished encoders.
The file was there. The training was not done.

THE FIX. A checkpoint counts as usable only if its pretraining job also
FINISHED, evidenced by the "best mlm_loss" line that the script prints once at
the end. A checkpoint whose job is still running is reported as IN PROGRESS,
does not satisfy the gate, and no fine-tune job is emitted against it.

ALSO FIXED. The walltime ladder topped out at 03:00:00 for sizes at or above
100,000, but the 150,000 encoder took 1:20:49, so 353,597 needs more than three
hours. That was caught by hand on 2026-09-18 and is now in the ladder.

Run from the repo root:
    PYTHONPATH=. python tools/patches/patch_gen_source_scale_gate.py
"""

from __future__ import annotations

import pathlib
import sys

GEN = pathlib.Path("slurm/generators/gen_source_scale.sh")

OLD_GATE = '''    CK="../checkpoints/edubert_ednet_pretrain_ednet_${TAG}_encoder.pt"
    if [ -f "$CODE/$CK" ]; then
      echo "have encoder, skipping: $CK"
      have=$((have + 1))
      continue
    fi'''

NEW_GATE = '''    CK="../checkpoints/edubert_ednet_pretrain_ednet_${TAG}_encoder.pt"
    if [ -f "$CODE/$CK" ]; then
      if grep -l -q "best mlm_loss" srcscale_pretrain_ednet_${TAG}_*.log 2>/dev/null; then
        echo "have encoder, skipping: $CK"
        have=$((have + 1))
        continue
      fi
      echo "IN PROGRESS, not usable yet: $CK"
      echo "  A checkpoint exists but its pretraining job has not printed"
      echo "  'best mlm_loss', so training has not finished. pretrain_edubert.py"
      echo "  saves on every loss improvement, so the file appears long before"
      echo "  the run ends. Loading it now yields a half-trained encoder."
      echo "  Wait for COMPLETED, then re-run this generator:"
      echo "    sacct -X -n --starttime today --format=JobName%40,State | grep ${TAG}"
      inprogress=$((inprogress + 1))
      continue
    fi'''

OLD_WALL = '''    if [ "$SIZE" -ge 100000 ]; then
      WALL=03:00:00'''

NEW_WALL = '''    if [ "$SIZE" -ge 300000 ]; then
      WALL=06:00:00
    elif [ "$SIZE" -ge 100000 ]; then
      WALL=03:00:00'''

OLD_COUNTERS = '''pcount=0
have=0'''

NEW_COUNTERS = '''pcount=0
have=0
inprogress=0'''

OLD_SUMMARY = '''echo "encoders to run   : $pcount   (already on disk: $have)"'''

NEW_SUMMARY = '''echo "encoders to run   : $pcount   (finished on disk: $have)"
if [ "$inprogress" -gt 0 ]; then
  echo "encoders IN PROGRESS: $inprogress   <- checkpoint exists but the job has"
  echo "                        not finished. Nothing was emitted against these."
fi'''

OLD_COMMENT = '''# The second invocation is not a mistake: the fine-tune queue can only be
# written once the encoders it points at exist, and this script refuses to
# emit fine-tune jobs for a checkpoint that is not on disk.'''

NEW_COMMENT = '''# The second invocation is not a mistake: the fine-tune queue can only be
# written once the encoders it points at are FINISHED, and this script refuses
# to emit fine-tune jobs against an unfinished checkpoint.
#
# "Finished" means the pretraining job printed "best mlm_loss", not merely that
# the .pt file exists. pretrain_edubert.py saves on every loss improvement, so
# the file appears minutes into a run. On 2026-09-15 a file-existence gate let
# 24 fine-tune jobs load 150,000-student encoders about 70 minutes early, which
# produced a draw spread of 0.0175 that fell to 0.0012 on the valid rerun. Do
# not weaken this check back to [ -f ].'''

EDITS = [
    ("header comment", OLD_COMMENT, NEW_COMMENT),
    ("counters", OLD_COUNTERS, NEW_COUNTERS),
    ("encoder gate", OLD_GATE, NEW_GATE),
    ("walltime ladder", OLD_WALL, NEW_WALL),
    ("summary line", OLD_SUMMARY, NEW_SUMMARY),
]


def main() -> None:
    if not GEN.exists():
        sys.exit("ABORT: %s not found. Run from the repo root." % GEN)
    s = GEN.read_text()

    if "IN PROGRESS, not usable yet" in s and "inprogress=0" in s:
        print("already applied; nothing to do")
        return

    for name, old, new in EDITS:
        n = s.count(old)
        if n != 1:
            sys.exit("ABORT (%s): matched %d, expected 1. File NOT modified." % (name, n))

    for name, old, new in EDITS:
        s = s.replace(old, new, 1)
        print("ok: %s" % name)

    GEN.write_text(s)
    print("\nwrote %s" % GEN)

    s2 = GEN.read_text()
    print()
    for probe, want in (('if [ -f "$CODE/$CK" ]; then', 1),
                        ('best mlm_loss', 1),
                        ('IN PROGRESS, not usable yet', 1),
                        ('inprogress=0', 1),
                        ('WALL=06:00:00', 1),
                        ('WALL=03:00:00', 1)):
        got = s2.count(probe)
        print("  %-30s %d  %s" % (probe[:30], got, "ok" if got == want else "CHECK"))


if __name__ == "__main__":
    main()
