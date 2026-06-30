#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --output=w5_nextskill_auc_sweep_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code

ASSIST=../checkpoints/edubert_assist2017_pretrain_full_encoder.pt
EDNET=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
JUNYI=../checkpoints/edubert_junyi_pretrain_full_encoder.pt

for SEED in 42 1 2; do
  for N in 25 50 100 200 500 1000; do
    # scratch (30 ep)
    PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill \
      --processed_dir ../processed/assist2017 --init scratch \
      --n_students $N --seed $SEED --epochs 30 \
      --run_type scratch_nsauc_n${N}_seed${SEED} --wandb
    # in-domain (30 ep)
    PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill \
      --processed_dir ../processed/assist2017 --init pretrained --encoder_ckpt $ASSIST \
      --n_students $N --seed $SEED --epochs 30 \
      --run_type indomain_nsauc_n${N}_seed${SEED} --wandb
    # EdNet (60 ep at N<=50 for convergence-fairness, else 30)
    if [ $N -le 50 ]; then EDEP=60; else EDEP=30; fi
    PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill \
      --processed_dir ../processed/assist2017 --init pretrained --encoder_ckpt $EDNET \
      --n_students $N --seed $SEED --epochs $EDEP \
      --run_type ednet_nsauc_n${N}_seed${SEED} --wandb
    # Junyi (30 ep)
    PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill \
      --processed_dir ../processed/assist2017 --init pretrained --encoder_ckpt $JUNYI \
      --n_students $N --seed $SEED --epochs 30 \
      --run_type junyi_nsauc_n${N}_seed${SEED} --wandb
  done
done
