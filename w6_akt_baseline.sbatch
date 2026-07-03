#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --output=w6_akt_baseline_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
for DS in assist2017 ednet junyi; do
  for SEED in 42 1 2; do
    PYTHONPATH=. python scripts/train_baseline.py --model akt --processed_dir ../processed/$DS --epochs 100 --warmup_frac 0.1 --seed $SEED --run_type akt_baseline_seed${SEED} --wandb
  done
done
