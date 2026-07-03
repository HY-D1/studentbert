#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --output=w6_dropoutK_s2_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
ASSIST=../checkpoints/edubert_assist2017_pretrain_full_encoder.pt
EDNET=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
JUNYI=../checkpoints/edubert_junyi_pretrain_full_encoder.pt
SEED=2
for K in 5 10 20 50 100 200; do
  PYTHONPATH=. python scripts/downstream_edubert.py --task dropout --processed_dir ../processed/assist2017 --init scratch --k_prefix $K --seed $SEED --epochs 30 --run_type scratch_dropout_k${K}_seed${SEED} --wandb
  PYTHONPATH=. python scripts/downstream_edubert.py --task dropout --processed_dir ../processed/assist2017 --init pretrained --encoder_ckpt $ASSIST --k_prefix $K --seed $SEED --epochs 30 --run_type indomain_dropout_k${K}_seed${SEED} --wandb
  PYTHONPATH=. python scripts/downstream_edubert.py --task dropout --processed_dir ../processed/assist2017 --init pretrained --encoder_ckpt $EDNET --k_prefix $K --seed $SEED --epochs 30 --run_type ednet_dropout_k${K}_seed${SEED} --wandb
  PYTHONPATH=. python scripts/downstream_edubert.py --task dropout --processed_dir ../processed/assist2017 --init pretrained --encoder_ckpt $JUNYI --k_prefix $K --seed $SEED --epochs 30 --run_type junyi_dropout_k${K}_seed${SEED} --wandb
done
