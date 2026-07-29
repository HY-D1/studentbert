#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=w4smoke_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
PYTHONPATH=. python scripts/pretrain_edubert.py \
  --processed_dir ../processed/ednet \
  --max_interactions 20000 --epochs 3 --batch_size 128 \
  --run_type pretrain_smoke --wandb
PYTHONPATH=. python scripts/finetune_edubert.py \
  --processed_dir ../processed/assist2017 --init pretrained \
  --encoder_ckpt ../checkpoints/edubert_ednet_pretrain_smoke_encoder.pt \
  --n_students 100 --epochs 5 --run_type ednet_smoke --wandb
