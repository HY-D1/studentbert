#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --mem=48G
#SBATCH --output=w6_akt_junyi_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
for SEED in 42 1 2; do
  PYTHONPATH=. python scripts/train_baseline.py --model akt --processed_dir ../processed/junyi --n_students 3000 --epochs 50 --warmup_frac 0.1 --seed $SEED --run_type akt_junyi_n3000_seed${SEED} --wandb
done
