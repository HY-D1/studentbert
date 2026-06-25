#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=coldcurve_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
for N in 50 100 200 500; do
  PYTHONPATH=. python scripts/finetune_edubert.py \
    --processed_dir ../processed/assist2017 --init pretrained \
    --encoder_ckpt ../checkpoints/edubert_assist2017_pretrain_full_encoder.pt \
    --n_students $N --epochs 30 --batch_size 32 --run_type pretrain_cold$N --wandb
  PYTHONPATH=. python scripts/finetune_edubert.py \
    --processed_dir ../processed/assist2017 --init scratch \
    --n_students $N --epochs 30 --batch_size 32 --run_type scratch_cold$N --wandb
done
