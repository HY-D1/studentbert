#!/usr/bin/env bash
# Safe, idempotent cleanup for the StudentBERT workspace.
#
#   * DRY-RUN by default. It only PRINTS what it would do.
#     Set APPLY=1 to actually delete checkpoints and archive duplicate logs.
#   * HARD GUARD: pretraining encoders (*_pretrain_*_encoder.pt) are NEVER
#     matched for deletion. Run this only AFTER your AUC/next-skill numbers are
#     collected (e.g. after parse_nextskill_full.py has written its table).
#   * Duplicate job logs are MOVED to logs_archive/ (reversible), not deleted,
#     so run provenance is preserved.
#
# Usage:
#   bash cleanup_sb.sh                          # dry run, report only
#   APPLY=1 bash cleanup_sb.sh                  # actually clean
#   CODE_DIR=/projects/algl/dai.hany/studentbert/code APPLY=1 bash cleanup_sb.sh
#   LOG_GLOBS="w5_nextskill_sweep_*.log w5_nsauc_*.log" APPLY=1 bash cleanup_sb.sh
set -euo pipefail
shopt -s nullglob

CODE_DIR="${CODE_DIR:-$(pwd)}"
CKPT_DIR="${CKPT_DIR:-$CODE_DIR/../checkpoints}"
APPLY="${APPLY:-0}"
ARCHIVE="${ARCHIVE:-$CODE_DIR/logs_archive}"
# Which log families to consider for de-duplication. Narrow by default to the
# W5 next-skill families relevant to the current task. Widen deliberately.
LOG_GLOBS="${LOG_GLOBS:-w5_nextskill_sweep_*.log w5_nextskill_auc_sweep_*.log w5_nsauc_*.log}"

say(){ printf '%s\n' "$*"; }
act(){ if [ "$APPLY" = "1" ]; then eval "$@"; else say "  [dry-run] $*"; fi; }

say "======== StudentBERT cleanup (APPLY=$APPLY) ========"
say "code dir : $CODE_DIR"
say "ckpt dir : $CKPT_DIR"
if [ "$APPLY" != "1" ]; then
  say "NOTE: DRY RUN. Nothing is changed. Re-run with APPLY=1 to act."
fi

# ---------- 0. disk snapshot (before) ----------
say ""; say "--- disk usage before ---"
du -sh "$CKPT_DIR" 2>/dev/null || say "  (ckpt dir not found)"
du -sh "$CODE_DIR" 2>/dev/null || true

# ---------- 1. fine-tuned *_best.pt (guarded) ----------
say ""; say "--- fine-tuned checkpoints to remove  (GUARD: *_pretrain_*_encoder.pt kept) ---"
best=()
while IFS= read -r -d '' f; do best+=("$f"); done < <(
  find "$CODE_DIR" "$CKPT_DIR" -type f -name '*_best.pt' ! -name '*pretrain*' -print0 2>/dev/null | sort -z
)
if [ "${#best[@]}" -eq 0 ]; then
  say "  (none found)"
else
  for f in "${best[@]}"; do say "  $(du -h "$f" 2>/dev/null | cut -f1)  $f"; done
  say "  ---- total: $(du -ch "${best[@]}" 2>/dev/null | tail -1 | cut -f1) in ${#best[@]} files ----"
  for f in "${best[@]}"; do act "rm -f \"$f\""; done
fi

# ---------- 1b. safety readout: encoders that MUST survive ----------
say ""; say "--- pretraining encoders (MUST be kept; never in the delete set above) ---"
enc=()
while IFS= read -r -d '' f; do enc+=("$f"); done < <(
  find "$CKPT_DIR" -type f -name '*_pretrain_*_encoder.pt' -print0 2>/dev/null | sort -z
)
if [ "${#enc[@]}" -eq 0 ]; then
  say "  WARNING: no *_pretrain_*_encoder.pt found under $CKPT_DIR (check the path)."
else
  for f in "${enc[@]}"; do say "  KEEP  $f"; done
fi

# ---------- 2. duplicate / resubmit job logs -> archive (reversible) ----------
say ""; say "--- duplicate job logs (same sbatch family, >1 job id) ---"
say "    families considered: $LOG_GLOBS"
logs=()
for g in $LOG_GLOBS; do
  for f in "$CODE_DIR"/$g; do logs+=("$f"); done
done
declare -A fam
for f in "${logs[@]}"; do
  b=$(basename "$f")
  key=$(printf '%s' "$b" | sed -E 's/_[0-9]+\.log$//')   # strip trailing _<jobid>.log
  [ "$key" = "$b" ] && continue                          # no jobid suffix -> skip
  fam["$key"]+="$f"$'\n'
done
found_dup=0
for key in $(printf '%s\n' "${!fam[@]}" | sort); do
  mapfile -t fs < <(printf '%s' "${fam[$key]}" | sed '/^$/d')
  [ "${#fs[@]}" -lt 2 ] && continue
  found_dup=1
  # sort by size desc; keep the largest (most complete), archive the rest
  mapfile -t sorted < <(for p in "${fs[@]}"; do printf '%s\t%s\n' "$(stat -c%s "$p" 2>/dev/null || echo 0)" "$p"; done | sort -rn | cut -f2-)
  say "  family: $key"
  say "    KEEP    $(basename "${sorted[0]}")  ($(du -h "${sorted[0]}" | cut -f1))"
  for p in "${sorted[@]:1}"; do
    say "    ARCHIVE $(basename "$p")  ($(du -h "$p" | cut -f1))"
    act "mkdir -p \"$ARCHIVE\" && mv \"$p\" \"$ARCHIVE\"/"
  done
done
[ "$found_dup" -eq 0 ] && say "  (no families with duplicate job ids)"

# ---------- 3. quick audit-grep pass over the log dir ----------
say ""; say "--- audit greps (over $CODE_DIR/*.log) ---"
cd "$CODE_DIR"
say "  CPU fallback (files WITHOUT device=cuda; empty is good):"
grep -L "device=cuda" *.log 2>/dev/null | sed 's/^/    /' || true
say "  crashes (Traceback/Killed/OOM; FutureWarning 'error' is benign):"
grep -lE "Traceback|Killed|OOM" *.log 2>/dev/null | sed 's/^/    /' || say "    (none)"
say "  NaN / AUC=1.0000 leakage signature:"
grep -liE "nan|auc.*: 1\.0000|auc 1\.0000" *.log 2>/dev/null | sed 's/^/    /' || say "    (none)"

# ---------- 4. disk snapshot (after, only meaningful under APPLY=1) ----------
say ""; say "--- disk usage after ---"
du -sh "$CKPT_DIR" 2>/dev/null || true
du -sh "$CODE_DIR" 2>/dev/null || true

say ""
if [ "$APPLY" = "1" ]; then
  say "Done. Deleted ${#best[@]} fine-tuned checkpoints; archived duplicate logs to $ARCHIVE."
else
  say "Dry run complete. Re-run with:  APPLY=1 bash cleanup_sb.sh"
fi
