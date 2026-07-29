#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=w5_ednet_lowN_ep60_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
EDNET=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
for SEED in 42 1 2; do
  for N in 25 50 100; do
    PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill \
      --processed_dir ../processed/assist2017 --init pretrained --encoder_ckpt $EDNET \
      --n_students $N --seed $SEED --epochs 60 \
      --run_type ednet_nextskill_n${N}_ep60_seed${SEED} --wandb
  done
done
