#!/usr/bin/env bash
# =============================================================================
# gen_macrotop1_assist.sh
#
# Generates the macro_top1 backfill for the ASSISTments 2017 next-skill sweep.
# RESULTS.md section 9 records that macro_top1 was never logged for this sweep
# (the metric arrived in W7, Junyi only), so the 24 cells read NOT FOUND.
#
# 6 jobs = 3 seeds x 2 budgets, each running the 4 conditions in sequence.
# Built from slurm/w5_nsauc_s1.sbatch, with three changes learned the hard way:
#   - no "source ~/.bashrc" (that file no longer exists) and no conda activate;
#     the interpreter is called by absolute path instead
#   - --cpus-per-task=8, because a 1-CPU draw made the loader the bottleneck
#     (6.3 steps/s on a V100 node vs 59-92 on a fast node)
#   - --time=08:00:00, because the same sbatch can take 0.5h or 7.3h
#
# run_type uses the token "mt1", NOT "nsauc" or "nextskill". That keeps these
# runs out of analysis/parse_nextskill_full.py, so the existing section 3.1
# table cannot be silently merged with or overwritten by them.
#
# Because seeds and data are unchanged, the top-1 these produce should
# REPRODUCE the 3.1 values exactly. That is a free reproducibility check.
#
#   bash slurm/generators/gen_macrotop1_assist.sh
# Writes: queue_mt1/*.sbatch   (submit with tools/drip_submit.sh queue_mt1)
# =============================================================================
set -euo pipefail

CODE=/projects/algl/dai.hany/studentbert/code
PY=/projects/algl/dai.hany/envs/sb/bin/python
QUEUE="$CODE/queue_mt1"
mkdir -p "$QUEUE"

ASSIST=../checkpoints/edubert_assist2017_pretrain_full_encoder.pt
EDNET=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
JUNYI=../checkpoints/edubert_junyi_pretrain_full_encoder.pt

n=0
for SEED in 1 2 42; do
  for N in 25 1000; do
    # EdNet used 60 epochs at N<=50 in the original sweep (convergence-fair)
    if [ "$N" -le 50 ]; then EDEP=60; else EDEP=30; fi
    f="$QUEUE/w9_mt1_n${N}_s${SEED}.sbatch"
    cat > "$f" <<EOF
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --job-name=mt1_n${N}_s${SEED}
#SBATCH --output=w9_mt1_n${N}_s${SEED}_%j.log
cd $CODE

PYTHONPATH=. $PY scripts/downstream_edubert.py --task next_skill \\
  --processed_dir ../processed/assist2017 --init scratch \\
  --n_students ${N} --seed ${SEED} --epochs 30 \\
  --run_type scratch_mt1_n${N}_seed${SEED} --wandb

PYTHONPATH=. $PY scripts/downstream_edubert.py --task next_skill \\
  --processed_dir ../processed/assist2017 --init pretrained --encoder_ckpt ${ASSIST} \\
  --n_students ${N} --seed ${SEED} --epochs 30 \\
  --run_type indomain_mt1_n${N}_seed${SEED} --wandb

PYTHONPATH=. $PY scripts/downstream_edubert.py --task next_skill \\
  --processed_dir ../processed/assist2017 --init pretrained --encoder_ckpt ${EDNET} \\
  --n_students ${N} --seed ${SEED} --epochs ${EDEP} \\
  --run_type ednet_mt1_n${N}_seed${SEED} --wandb

PYTHONPATH=. $PY scripts/downstream_edubert.py --task next_skill \\
  --processed_dir ../processed/assist2017 --init pretrained --encoder_ckpt ${JUNYI} \\
  --n_students ${N} --seed ${SEED} --epochs 30 \\
  --run_type junyi_mt1_n${N}_seed${SEED} --wandb
EOF
    n=$((n + 1))
  done
done

echo "wrote $n sbatch files to $QUEUE"
echo
echo "INSPECT ONE BEFORE SUBMITTING:"
echo "  cat $QUEUE/w9_mt1_n25_s1.sbatch"
echo
echo "encoders these expect (must all exist):"
for c in "$ASSIST" "$EDNET" "$JUNYI"; do
  if [ -f "$CODE/$c" ]; then echo "  ok      $c"; else echo "  MISSING $c"; fi
done
echo
echo "then: bash tools/drip_submit.sh queue_mt1"
