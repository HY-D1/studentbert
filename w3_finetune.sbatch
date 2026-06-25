#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=w3finetune_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code

# === TASK 3: full fine-tune, pretrained vs scratch (4-model table) ===
# pretrained EduBERT on full ASSISTments
PYTHONPATH=. python scripts/finetune_edubert.py \
  --processed_dir ../processed/assist2017 --init pretrained \
  --encoder_ckpt ../checkpoints/edubert_assist2017_pretrain_full_encoder.pt \
  --epochs 20 --batch_size 64 --run_type pretrain_full_ft --wandb

# scratch EduBERT on full ASSISTments
PYTHONPATH=. python scripts/finetune_edubert.py \
  --processed_dir ../processed/assist2017 --init scratch \
  --epochs 20 --batch_size 64 --run_type scratch_full_ft --wandb

# === TASK 4: cold-start — pretrained on 50 students vs scratch on 500 ===
PYTHONPATH=. python scripts/finetune_edubert.py \
  --processed_dir ../processed/assist2017 --init pretrained \
  --encoder_ckpt ../checkpoints/edubert_assist2017_pretrain_full_encoder.pt \
  --n_students 50 --epochs 30 --batch_size 32 --run_type pretrain_cold50 --wandb

PYTHONPATH=. python scripts/finetune_edubert.py \
  --processed_dir ../processed/assist2017 --init scratch \
  --n_students 500 --epochs 30 --batch_size 32 --run_type scratch_cold500 --wandb
