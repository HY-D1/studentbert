#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --output=w6_probe_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
ASSIST=../checkpoints/edubert_assist2017_pretrain_full_encoder.pt
EDNET=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
JUNYI=../checkpoints/edubert_junyi_pretrain_full_encoder.pt
for SEED in 42 1 2; do
  PYTHONPATH=. python scripts/probe_edubert.py --processed_dir ../processed/assist2017 --init scratch --seed $SEED --epochs 20 --run_type probe_scratch_seed${SEED} --wandb
  PYTHONPATH=. python scripts/probe_edubert.py --processed_dir ../processed/assist2017 --init pretrained --encoder_ckpt $ASSIST --seed $SEED --epochs 20 --run_type probe_indomain_seed${SEED} --wandb
  PYTHONPATH=. python scripts/probe_edubert.py --processed_dir ../processed/assist2017 --init pretrained --encoder_ckpt $EDNET --seed $SEED --epochs 20 --run_type probe_ednet_seed${SEED} --wandb
  PYTHONPATH=. python scripts/probe_edubert.py --processed_dir ../processed/assist2017 --init pretrained --encoder_ckpt $JUNYI --seed $SEED --epochs 20 --run_type probe_junyi_seed${SEED} --wandb
done
