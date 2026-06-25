#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=saint100_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
PYTHONPATH=. python scripts/train_baseline.py \
  --model saint --processed_dir ../processed/assist2017 \
  --epochs 100 --batch_size 64 --lr 1e-3 --warmup_frac 0.1 \
  --dropout 0.1 --run_type baseline --wandb
