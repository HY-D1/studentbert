#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --mem=48G
#SBATCH --output=w6_kt_assist_n3000_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
ASSIST=../checkpoints/edubert_assist2017_pretrain_full_encoder.pt
EDNET=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
JUNYI=../checkpoints/edubert_junyi_pretrain_full_encoder.pt
T=../processed/assist2017
N=3000
for SEED in 42 1 2; do
  PYTHONPATH=. python scripts/finetune_edubert.py --processed_dir $T --init scratch --n_students $N --seed $SEED --epochs 20 --run_type kt_assist_scratch_n${N}_seed${SEED} --wandb
  PYTHONPATH=. python scripts/finetune_edubert.py --processed_dir $T --init pretrained --encoder_ckpt $ASSIST --n_students $N --seed $SEED --epochs 20 --run_type kt_assist_indomain_n${N}_seed${SEED} --wandb
  PYTHONPATH=. python scripts/finetune_edubert.py --processed_dir $T --init pretrained --encoder_ckpt $EDNET --n_students $N --seed $SEED --epochs 20 --run_type kt_assist_fromednet_n${N}_seed${SEED} --wandb
  PYTHONPATH=. python scripts/finetune_edubert.py --processed_dir $T --init pretrained --encoder_ckpt $JUNYI --n_students $N --seed $SEED --epochs 20 --run_type kt_assist_fromjunyi_n${N}_seed${SEED} --wandb
done
