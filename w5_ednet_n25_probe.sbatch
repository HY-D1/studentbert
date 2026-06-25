#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --output=w5_ednet_n25_probe_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
EDNET=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
for SEED in 3 4 5 6 7; do
  PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill \
    --processed_dir ../processed/assist2017 --init pretrained --encoder_ckpt $EDNET \
    --n_students 25 --seed $SEED --epochs 30 \
    --run_type ednet_nextskill_n25_seed${SEED} --wandb
  PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill \
    --processed_dir ../processed/assist2017 --init scratch \
    --n_students 25 --seed $SEED --epochs 30 \
    --run_type scratch_nextskill_n25_seed${SEED} --wandb
done
