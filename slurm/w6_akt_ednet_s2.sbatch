#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --output=w6_akt_ednet_s2_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
PYTHONPATH=. python scripts/train_baseline.py --model akt --processed_dir ../processed/ednet --n_students 3000 --epochs 50 --warmup_frac 0.1 --seed 2 --run_type akt_ednet_n3000_seed2 --wandb
