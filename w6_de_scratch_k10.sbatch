#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --output=w6_de_scratch_k10_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
for SEED in 42 1 2; do
  PYTHONPATH=. python scripts/downstream_edubert.py --task dropout --processed_dir ../processed/ednet --init scratch --k_prefix 10 --n_students 3000 --seed $SEED --epochs 15 --run_type drop_ednet_scratch_k10_n3000_seed${SEED} --wandb
done
