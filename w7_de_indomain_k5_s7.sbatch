#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --output=w7_de_indomain_k5_s7_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
PYTHONPATH=. python scripts/downstream_edubert.py --task dropout --processed_dir ../processed/ednet --init pretrained --encoder_ckpt ../checkpoints/edubert_ednet_pretrain_full_encoder.pt --k_prefix 5 --n_students 3000 --seed 7 --epochs 15 --run_type drop_ednet_indomain_k5_n3000_seed7 --wandb
