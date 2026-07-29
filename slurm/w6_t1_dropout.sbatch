#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --output=w6_t1_dropout_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
ASSIST=../checkpoints/edubert_assist2017_pretrain_full_encoder.pt
EDNET=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
JUNYI=../checkpoints/edubert_junyi_pretrain_full_encoder.pt
for K in 5 10; do
  for SEED in 42 1 2; do
    # EdNet target
    PYTHONPATH=. python scripts/downstream_edubert.py --task dropout --processed_dir ../processed/ednet --init scratch --k_prefix $K --seed $SEED --epochs 30 --run_type drop_ednet_scratch_k${K}_seed${SEED} --wandb
    PYTHONPATH=. python scripts/downstream_edubert.py --task dropout --processed_dir ../processed/ednet --init pretrained --encoder_ckpt $EDNET --k_prefix $K --seed $SEED --epochs 30 --run_type drop_ednet_indomain_k${K}_seed${SEED} --wandb
    PYTHONPATH=. python scripts/downstream_edubert.py --task dropout --processed_dir ../processed/ednet --init pretrained --encoder_ckpt $ASSIST --k_prefix $K --seed $SEED --epochs 30 --run_type drop_ednet_fromassist_k${K}_seed${SEED} --wandb
    PYTHONPATH=. python scripts/downstream_edubert.py --task dropout --processed_dir ../processed/ednet --init pretrained --encoder_ckpt $JUNYI --k_prefix $K --seed $SEED --epochs 30 --run_type drop_ednet_fromjunyi_k${K}_seed${SEED} --wandb
    # Junyi target
    PYTHONPATH=. python scripts/downstream_edubert.py --task dropout --processed_dir ../processed/junyi --init scratch --k_prefix $K --seed $SEED --epochs 30 --run_type drop_junyi_scratch_k${K}_seed${SEED} --wandb
    PYTHONPATH=. python scripts/downstream_edubert.py --task dropout --processed_dir ../processed/junyi --init pretrained --encoder_ckpt $JUNYI --k_prefix $K --seed $SEED --epochs 30 --run_type drop_junyi_indomain_k${K}_seed${SEED} --wandb
    PYTHONPATH=. python scripts/downstream_edubert.py --task dropout --processed_dir ../processed/junyi --init pretrained --encoder_ckpt $ASSIST --k_prefix $K --seed $SEED --epochs 30 --run_type drop_junyi_fromassist_k${K}_seed${SEED} --wandb
    PYTHONPATH=. python scripts/downstream_edubert.py --task dropout --processed_dir ../processed/junyi --init pretrained --encoder_ckpt $EDNET --k_prefix $K --seed $SEED --epochs 30 --run_type drop_junyi_fromednet_k${K}_seed${SEED} --wandb
  done
done
