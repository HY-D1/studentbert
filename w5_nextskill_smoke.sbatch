#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=w5_ns_smoke_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill \
  --processed_dir ../processed/assist2017 --init scratch \
  --n_students 100 --seed 42 --epochs 5 --run_type smoke_ns_n100_seed42 --wandb
