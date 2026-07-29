#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=dropout_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
for COND in scratch indomain ednet junyi; do :; done
PYTHONPATH=. python scripts/downstream_edubert.py --task dropout \
  --processed_dir ../processed/assist2017 --init scratch \
  --k_prefix 20 --epochs 30 --run_type scratch_dropout --wandb
PYTHONPATH=. python scripts/downstream_edubert.py --task dropout \
  --processed_dir ../processed/assist2017 --init pretrained \
  --encoder_ckpt ../checkpoints/edubert_assist2017_pretrain_full_encoder.pt \
  --k_prefix 20 --epochs 30 --run_type indomain_dropout --wandb
PYTHONPATH=. python scripts/downstream_edubert.py --task dropout \
  --processed_dir ../processed/assist2017 --init pretrained \
  --encoder_ckpt ../checkpoints/edubert_ednet_pretrain_full_encoder.pt \
  --k_prefix 20 --epochs 30 --run_type ednet_dropout --wandb
PYTHONPATH=. python scripts/downstream_edubert.py --task dropout \
  --processed_dir ../processed/assist2017 --init pretrained \
  --encoder_ckpt ../checkpoints/edubert_junyi_pretrain_full_encoder.pt \
  --k_prefix 20 --epochs 30 --run_type junyi_dropout --wandb
