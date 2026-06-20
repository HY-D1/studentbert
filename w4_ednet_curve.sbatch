#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=ednet_curve_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
ENC=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
for SEED in 42 1 2; do
  for N in 25 50 100 200 500 1000; do
    PYTHONPATH=. python scripts/finetune_edubert.py \
      --processed_dir ../processed/assist2017 --init pretrained \
      --encoder_ckpt $ENC --n_students $N --seed $SEED \
      --epochs 30 --batch_size 32 --run_type ednet_cold_pre_n${N}_s${SEED} --wandb
  done
done
