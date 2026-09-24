#!/usr/bin/env bash
# Round-robin submitter over several queue directories, so a slow or blocked queue never
# starves the others (tools/drip_submit.sh drains one queue at a time).
#
#   bash tools/drip_multi.sh queue_a queue_b ...
#
# Environment:
#   CAP=7          jobs kept submitted in PARTITION; the gpu QOS allows 8 per user, and one slot
#                  stays free for an interactive srun
#   PARTITION=gpu  only jobs in this partition count toward CAP
#   SLEEP=60       seconds between polls when the cap is full
#   ONCE=1         fill up to CAP once and exit (for scrontab)
#   DRYRUN=1       list what would be submitted; change nothing
#   DONE_CMD="..." run once, after every queue is drained AND no job of yours remains in
#                  PARTITION (for example the final benchmark build)
#
# Safety: only one drip_multi runs at a time (lock in $DRIP_LOCK), and a queue still owned by an
# old drip_submit.sh loop (its .drip.lock) is skipped rather than shared, so nothing can be
# submitted twice.
set -u
CAP="${CAP:-7}"
PARTITION="${PARTITION:-gpu}"
SLEEP="${SLEEP:-60}"
ONCE="${ONCE:-0}"
DRYRUN="${DRYRUN:-0}"
DONE_CMD="${DONE_CMD:-}"
LOCK="${DRIP_LOCK:-$PWD/.drip_multi.lock}"

if [ "$#" -lt 1 ]; then
  echo "usage: bash tools/drip_multi.sh QUEUE_DIR [QUEUE_DIR ...]"
  exit 2
fi
for q in "$@"; do
  if [ ! -d "$q" ]; then
    echo "no such queue directory: $q"
    exit 2
  fi
  mkdir -p "$q/done"
done

pending() { ls "$1"/*.sbatch 2>/dev/null | wc -l; }
mine() { squeue -u "$USER" -h -p "$PARTITION" 2>/dev/null | wc -l; }
stamp() { date +%T; }

if [ "$DRYRUN" = "1" ]; then
  for q in "$@"; do
    echo "$q: $(pending "$q") pending"
    ls "$q"/*.sbatch 2>/dev/null | head -3 | sed 's/^/  next: /'
  done
  echo "DRYRUN: nothing submitted, nothing moved; $(mine) of your jobs in $PARTITION (cap $CAP)"
  exit 0
fi

exec 8>"$LOCK"
if ! flock -n 8; then
  echo "$(stamp) another drip_multi already runs from this directory; exiting"
  exit 0
fi
echo "$(stamp) drip_multi: $# queue(s), cap $CAP in partition $PARTITION, node $(hostname -s)"

while :; do
  total=0
  for q in "$@"; do total=$((total + $(pending "$q"))); done
  if [ "$total" -eq 0 ]; then
    if [ -z "$DONE_CMD" ]; then
      echo "$(stamp) all queues drained"
      exit 0
    fi
    if [ "$(mine)" -eq 0 ]; then
      echo "$(stamp) all queues drained and no jobs left in $PARTITION; running DONE_CMD"
      bash -c "$DONE_CMD"
      echo "$(stamp) DONE_CMD exited with status $?"
      exit 0
    fi
    [ "$ONCE" = "1" ] && exit 0
    sleep "$SLEEP"
    continue
  fi
  submitted=0
  for q in "$@"; do
    f=$(ls "$q"/*.sbatch 2>/dev/null | head -1)
    [ -n "$f" ] || continue
    [ "$(mine)" -lt "$CAP" ] || break
    exec 9>"$q/.drip.lock"
    if ! flock -n 9; then
      exec 9>&-
      echo "$(stamp) $q is owned by another loop; skipping it this round"
      continue
    fi
    if out=$(sbatch "$f" 2>&1); then
      mv "$f" "$q/done/"
      echo "$(stamp) $out   <- $q/$(basename "$f")"
      submitted=1
    else
      echo "$(stamp) SUBMIT FAILED for $f: $out"
    fi
    exec 9>&-
  done
  # Capacity may remain after a full round; go round again at once instead of sleeping.
  [ "$submitted" = "1" ] && continue
  if [ "$ONCE" = "1" ]; then
    echo "$(stamp) single pass done: cap reached or nothing submittable"
    exit 0
  fi
  sleep "$SLEEP"
done
