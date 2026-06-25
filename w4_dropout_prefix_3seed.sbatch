#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=dropout_prefix3_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
# clean-K prefix sweep, 3 seeds, scratch vs EdNet
# (drop K=100 from the K list if Step 1 shows it leaks)
for SEED in 42 1 2; do
  for K in 20 50 100; do
    PYTHONPATH=. python scripts/downstream_edubert.py --task dropout \
      --processed_dir ../processed/assist2017 --init scratch \
      --k_prefix $K --seed $SEED --epochs 30 \
      --run_type scratch_dropout_k${K}_s${SEED} --wandb
    PYTHONPATH=. python scripts/downstream_edubert.py --task dropout \
      --processed_dir ../processed/assist2017 --init pretrained \
      --encoder_ckpt ../checkpoints/edubert_ednet_pretrain_full_encoder.pt \
      --k_prefix $K --seed $SEED --epochs 30 \
      --run_type ednet_dropout_k${K}_s${SEED} --wandb
  done
done
