#!/usr/bin/env bash
# =============================================================================
# drip_submit.sh
#
# Submits sbatch files from a queue directory a few at a time, respecting the
# ~8-job QOS submit cap. Idempotent: a submitted file is moved to done/, so
# rerunning after a dropped connection never resubmits the same job.
#
#   bash tools/drip_submit.sh queue_mt1
#   bash tools/drip_submit.sh queue_probe
#   CAP=6 SLEEP=90 bash tools/drip_submit.sh queue_mt1
#   DRYRUN=1 bash tools/drip_submit.sh queue_mt1     # show, submit nothing
#
# The loop dies if the login node changes (explorer-01 vs -02). That is fine:
# check `squeue -u $USER`, then rerun this exact command. Never run two loops
# over the same queue directory.
# =============================================================================
set -uo pipefail

QDIR="${1:-}"
CAP="${CAP:-7}"
SLEEP="${SLEEP:-60}"
DRYRUN="${DRYRUN:-0}"
ME="${USER:-$(id -un)}"

if [ -z "$QDIR" ] || [ ! -d "$QDIR" ]; then
  echo "usage: bash tools/drip_submit.sh <queue-dir>"
  echo "  e.g. bash tools/drip_submit.sh queue_mt1"
  exit 1
fi

mkdir -p "$QDIR/done"
total=$(ls "$QDIR"/*.sbatch 2>/dev/null | wc -l)
if [ "$total" -eq 0 ]; then
  echo "nothing to submit: $QDIR holds no *.sbatch (already drained?)"
  echo "submitted so far: $(ls "$QDIR/done" 2>/dev/null | wc -l)"
  exit 0
fi

echo "queue      : $QDIR"
echo "to submit  : $total"
echo "cap        : $CAP concurrent, polling every ${SLEEP}s"
echo "node       : $(hostname)"
[ "$DRYRUN" = "1" ] && echo "DRYRUN: nothing will actually be submitted"
echo

while :; do
  pending=$(ls "$QDIR"/*.sbatch 2>/dev/null | wc -l)
  [ "$pending" -eq 0 ] && break

  mine=$(squeue -u "$ME" -h 2>/dev/null | wc -l)
  if [ "$mine" -lt "$CAP" ]; then
    f=$(ls "$QDIR"/*.sbatch 2>/dev/null | head -1)
    [ -z "$f" ] && break
    if [ "$DRYRUN" = "1" ]; then
      echo "[dry] would submit $(basename "$f")   (queue has $mine of my jobs)"
      mv "$f" "$QDIR/done/"
    else
      out=$(sbatch "$f" 2>&1)
      rc=$?
      if [ $rc -eq 0 ]; then
        echo "$(date +%H:%M:%S)  $out   <- $(basename "$f")"
        mv "$f" "$QDIR/done/"
      else
        echo "$(date +%H:%M:%S)  SUBMIT FAILED for $(basename "$f"): $out"
        echo "  left in the queue; fix the cause and rerun this script"
        sleep "$SLEEP"
      fi
    fi
  else
    echo "$(date +%H:%M:%S)  $mine jobs queued or running (cap $CAP), waiting"
    sleep "$SLEEP"
  fi
done

echo
echo "queue drained. submitted: $(ls "$QDIR/done" | wc -l)"
echo "watch with : squeue -u $ME"
echo "after they finish, check states with:"
echo "  sacct -X --format=JobID,JobName%22,State,Elapsed,ExitCode -S \$(date +%F)"
