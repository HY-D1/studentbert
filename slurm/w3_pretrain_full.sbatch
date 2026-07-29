#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=pretrainfull_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
PYTHONPATH=. python scripts/pretrain_edubert.py \
  --processed_dir ../processed/assist2017 \
  --epochs 100 --batch_size 64 --lr 1e-3 --warmup_frac 0.1 \
  --dropout 0.1 --run_type pretrain_full --wandb
