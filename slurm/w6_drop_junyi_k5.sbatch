#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --output=w6_drop_junyi_k5_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
ASSIST=../checkpoints/edubert_assist2017_pretrain_full_encoder.pt
EDNET=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
JUNYI=../checkpoints/edubert_junyi_pretrain_full_encoder.pt
K=5
N=3000
T=../processed/junyi
for SEED in 42 1 2; do
  PYTHONPATH=. python scripts/downstream_edubert.py --task dropout --processed_dir $T --init scratch --k_prefix $K --n_students $N --seed $SEED --epochs 30 --run_type drop_junyi_scratch_k${K}_n${N}_seed${SEED} --wandb
  PYTHONPATH=. python scripts/downstream_edubert.py --task dropout --processed_dir $T --init pretrained --encoder_ckpt $JUNYI --k_prefix $K --n_students $N --seed $SEED --epochs 30 --run_type drop_junyi_indomain_k${K}_n${N}_seed${SEED} --wandb
  PYTHONPATH=. python scripts/downstream_edubert.py --task dropout --processed_dir $T --init pretrained --encoder_ckpt $ASSIST --k_prefix $K --n_students $N --seed $SEED --epochs 30 --run_type drop_junyi_fromassist_k${K}_n${N}_seed${SEED} --wandb
  PYTHONPATH=. python scripts/downstream_edubert.py --task dropout --processed_dir $T --init pretrained --encoder_ckpt $EDNET --k_prefix $K --n_students $N --seed $SEED --epochs 30 --run_type drop_junyi_fromednet_k${K}_n${N}_seed${SEED} --wandb
done
