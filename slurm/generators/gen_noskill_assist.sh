#!/usr/bin/env bash
# =============================================================================
# gen_noskill_assist.sh
#
# Re-runs the ASSISTments 2017 next-skill cells with --exclude_label noskill so
# the sensitivity analysis excluding the placeholder label can be reported.
# New GPU runs are unavoidable: the original sweep logged aggregate metrics
# only, per-example predictions were never written, and the _best.pt fine-tune
# checkpoints were deleted to reclaim quota, so nothing can be recomputed
# offline. Training is IDENTICAL to the original runs; only the reported
# metrics change, and --save_preds now writes the predictions so no future
# metric variant ever needs GPU time again.
#
# Default grid: N = 25 and 1000, matching the existing macro-top-1 appendix.
#   6 jobs = 3 seeds x 2 budgets, 4 conditions each = 24 runs, about 1 GPU-hour.
# Full sweep instead (72 runs, about 3 GPU-hours):
#   NLIST="25 50 100 200 500 1000" bash slurm/generators/gen_noskill_assist.sh
#
# Copied from the verified slurm/generators/gen_macrotop1_assist.sh, keeping the
# three settings learned the hard way: no "source ~/.bashrc" and no conda
# activate (absolute interpreter instead), --cpus-per-task=8, --time=08:00:00.
#
# run_type uses the token "nsk", NOT "nsauc" or "nextskill" or "mt1", so
# analysis/parse_nextskill_full.py ignores these and the section 3.1 and 3.3
# tables cannot be merged with or overwritten by them.
#
# Seeds, samples and epochs are unchanged, so the plain test/top1 these emit
# should reproduce the existing values. That is a free correctness check: if
# top1 drifts more than about 0.01, stop and investigate before reporting.
#
#   bash slurm/generators/gen_noskill_assist.sh
# Writes: queue_noskill/*.sbatch  (submit with: bash tools/drip_submit.sh queue_noskill)
# =============================================================================
set -euo pipefail

CODE=/projects/algl/dai.hany/studentbert/code
PY=/projects/algl/dai.hany/envs/sb/bin/python
QUEUE="$CODE/queue_noskill"
NLIST="${NLIST:-25 1000}"
mkdir -p "$QUEUE"

ASSIST=../checkpoints/edubert_assist2017_pretrain_full_encoder.pt
EDNET=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
JUNYI=../checkpoints/edubert_junyi_pretrain_full_encoder.pt

n=0
for SEED in 1 2 42; do
  for N in $NLIST; do
    # EdNet used 60 epochs at N<=50 in the original sweep (convergence-fair)
    if [ "$N" -le 50 ]; then EDEP=60; else EDEP=30; fi
    f="$QUEUE/w10_nsk_n${N}_s${SEED}.sbatch"
    cat > "$f" <<EOF
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --job-name=nsk_n${N}_s${SEED}
#SBATCH --output=w10_nsk_n${N}_s${SEED}_%j.log
cd $CODE

PYTHONPATH=. $PY scripts/downstream_edubert.py --task next_skill \\
  --processed_dir ../processed/assist2017 --init scratch \\
  --n_students ${N} --seed ${SEED} --epochs 30 \\
  --exclude_label noskill --save_preds \\
  --run_type scratch_nsk_n${N}_seed${SEED} --wandb

PYTHONPATH=. $PY scripts/downstream_edubert.py --task next_skill \\
  --processed_dir ../processed/assist2017 --init pretrained --encoder_ckpt ${ASSIST} \\
  --n_students ${N} --seed ${SEED} --epochs 30 \\
  --exclude_label noskill --save_preds \\
  --run_type indomain_nsk_n${N}_seed${SEED} --wandb

PYTHONPATH=. $PY scripts/downstream_edubert.py --task next_skill \\
  --processed_dir ../processed/assist2017 --init pretrained --encoder_ckpt ${EDNET} \\
  --n_students ${N} --seed ${SEED} --epochs ${EDEP} \\
  --exclude_label noskill --save_preds \\
  --run_type ednet_nsk_n${N}_seed${SEED} --wandb

PYTHONPATH=. $PY scripts/downstream_edubert.py --task next_skill \\
  --processed_dir ../processed/assist2017 --init pretrained --encoder_ckpt ${JUNYI} \\
  --n_students ${N} --seed ${SEED} --epochs 30 \\
  --exclude_label noskill --save_preds \\
  --run_type junyi_nsk_n${N}_seed${SEED} --wandb
EOF
    n=$((n + 1))
  done
done

echo "wrote $n sbatch files to $QUEUE  (N list: $NLIST)"
echo "runs = $((n * 4))"
echo
echo "INSPECT ONE BEFORE SUBMITTING:"
echo "  cat $QUEUE/$(ls "$QUEUE" | head -1)"
echo
echo "encoders these expect (must all exist):"
for c in "$ASSIST" "$EDNET" "$JUNYI"; do
  if [ -f "$CODE/$c" ]; then echo "  ok      $c"; else echo "  MISSING $c"; fi
done
echo
echo "then: bash tools/drip_submit.sh queue_noskill"
