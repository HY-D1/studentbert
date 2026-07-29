#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=w3tiny_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code

# 1) pretrain on ~5,000 interactions
PYTHONPATH=. python scripts/pretrain_edubert.py \
  --processed_dir ../processed/assist2017 \
  --max_interactions 5000 --epochs 50 --batch_size 32 \
  --run_type pretrain_tiny --wandb

# 2) fine-tune the pretrained encoder for KT
PYTHONPATH=. python scripts/finetune_edubert.py \
  --processed_dir ../processed/assist2017 --init pretrained \
  --encoder_ckpt ../checkpoints/edubert_assist2017_pretrain_tiny_encoder.pt \
  --epochs 20 --batch_size 32 --run_type pretrain_tiny --wandb

# 3) scratch control (same KT fine-tune, random init)
PYTHONPATH=. python scripts/finetune_edubert.py \
  --processed_dir ../processed/assist2017 --init scratch \
  --epochs 20 --batch_size 32 --run_type scratch_tiny --wandb
