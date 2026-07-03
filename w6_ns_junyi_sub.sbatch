#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --output=w6_ns_junyi_sub_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
ASSIST=../checkpoints/edubert_assist2017_pretrain_full_encoder.pt
EDNET=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
JUNYI=../checkpoints/edubert_junyi_pretrain_full_encoder.pt
T=../processed/junyi
N=3000
for SEED in 42 1 2; do
  PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill --processed_dir $T --init scratch --n_students $N --seed $SEED --epochs 20 --run_type ns_junyi_scratch_n${N}_seed${SEED} --wandb
  PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill --processed_dir $T --init pretrained --encoder_ckpt $JUNYI --n_students $N --seed $SEED --epochs 20 --run_type ns_junyi_indomain_n${N}_seed${SEED} --wandb
  PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill --processed_dir $T --init pretrained --encoder_ckpt $ASSIST --n_students $N --seed $SEED --epochs 20 --run_type ns_junyi_fromassist_n${N}_seed${SEED} --wandb
  PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill --processed_dir $T --init pretrained --encoder_ckpt $EDNET --n_students $N --seed $SEED --epochs 20 --run_type ns_junyi_fromednet_n${N}_seed${SEED} --wandb
done
