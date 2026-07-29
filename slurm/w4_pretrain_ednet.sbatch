#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=pretrain_ednet_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
PYTHONPATH=. python scripts/pretrain_edubert.py \
  --processed_dir ../processed/ednet \
  --epochs 10 --batch_size 128 --lr 1e-3 --warmup_frac 0.05 \
  --dropout 0.1 --run_type pretrain_full --wandb
