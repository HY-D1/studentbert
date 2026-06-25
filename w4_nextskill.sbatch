#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=nextskill_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
# scratch
PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill \
  --processed_dir ../processed/assist2017 --init scratch \
  --epochs 30 --run_type scratch_nextskill --wandb
# in-domain
PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill \
  --processed_dir ../processed/assist2017 --init pretrained \
  --encoder_ckpt ../checkpoints/edubert_assist2017_pretrain_full_encoder.pt \
  --epochs 30 --run_type indomain_nextskill --wandb
# EdNet
PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill \
  --processed_dir ../processed/assist2017 --init pretrained \
  --encoder_ckpt ../checkpoints/edubert_ednet_pretrain_full_encoder.pt \
  --epochs 30 --run_type ednet_nextskill --wandb
# Junyi
PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill \
  --processed_dir ../processed/assist2017 --init pretrained \
  --encoder_ckpt ../checkpoints/edubert_junyi_pretrain_full_encoder.pt \
  --epochs 30 --run_type junyi_nextskill --wandb
