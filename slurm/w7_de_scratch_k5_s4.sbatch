#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --output=w7_de_scratch_k5_s4_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
PYTHONPATH=. python scripts/downstream_edubert.py --task dropout --processed_dir ../processed/ednet --init scratch  --k_prefix 5 --n_students 3000 --seed 4 --epochs 15 --run_type drop_ednet_scratch_k5_n3000_seed4 --wandb
