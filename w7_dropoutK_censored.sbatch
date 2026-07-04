#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --output=w7_dropoutK_censored_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
ASSIST=../checkpoints/edubert_assist2017_pretrain_full_encoder.pt
EDNET=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
JUNYI=../checkpoints/edubert_junyi_pretrain_full_encoder.pt
T=../processed/assist2017
for K in 100 200; do
  for SEED in 42 1 2; do
    PYTHONPATH=. python scripts/downstream_edubert.py --task dropout --processed_dir $T --init scratch --k_prefix $K --window_censor --seed $SEED --epochs 30 --run_type drop_assist_scratch_cens_k${K}_seed${SEED} --wandb
    PYTHONPATH=. python scripts/downstream_edubert.py --task dropout --processed_dir $T --init pretrained --encoder_ckpt $ASSIST --k_prefix $K --window_censor --seed $SEED --epochs 30 --run_type drop_assist_indomain_cens_k${K}_seed${SEED} --wandb
    PYTHONPATH=. python scripts/downstream_edubert.py --task dropout --processed_dir $T --init pretrained --encoder_ckpt $EDNET --k_prefix $K --window_censor --seed $SEED --epochs 30 --run_type drop_assist_fromednet_cens_k${K}_seed${SEED} --wandb
    PYTHONPATH=. python scripts/downstream_edubert.py --task dropout --processed_dir $T --init pretrained --encoder_ckpt $JUNYI --k_prefix $K --window_censor --seed $SEED --epochs 30 --run_type drop_assist_fromjunyi_cens_k${K}_seed${SEED} --wandb
  done
done
