#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --output=w5_nsauc_s1_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
ASSIST=../checkpoints/edubert_assist2017_pretrain_full_encoder.pt
EDNET=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
JUNYI=../checkpoints/edubert_junyi_pretrain_full_encoder.pt
SEED=1
for N in 25 50 100 200 500 1000; do
  PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill --processed_dir ../processed/assist2017 --init scratch --n_students $N --seed $SEED --epochs 30 --run_type scratch_nsauc_n${N}_seed${SEED} --wandb
  PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill --processed_dir ../processed/assist2017 --init pretrained --encoder_ckpt $ASSIST --n_students $N --seed $SEED --epochs 30 --run_type indomain_nsauc_n${N}_seed${SEED} --wandb
  if [ $N -le 50 ]; then EDEP=60; else EDEP=30; fi
  PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill --processed_dir ../processed/assist2017 --init pretrained --encoder_ckpt $EDNET --n_students $N --seed $SEED --epochs $EDEP --run_type ednet_nsauc_n${N}_seed${SEED} --wandb
  PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill --processed_dir ../processed/assist2017 --init pretrained --encoder_ckpt $JUNYI --n_students $N --seed $SEED --epochs 30 --run_type junyi_nsauc_n${N}_seed${SEED} --wandb
done
