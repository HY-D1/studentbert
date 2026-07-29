#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --mem=32G
#SBATCH --output=ds_smoke_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill \
  --processed_dir ../processed/assist2017 --init scratch \
  --epochs 5 --run_type smoke_nextskill --wandb
PYTHONPATH=. python scripts/downstream_edubert.py --task dropout \
  --processed_dir ../processed/assist2017 --init scratch \
  --epochs 5 --run_type smoke_dropout --wandb
