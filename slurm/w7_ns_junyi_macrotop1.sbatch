#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --output=w7_ns_junyi_macrotop1_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
ASSIST=../checkpoints/edubert_assist2017_pretrain_full_encoder.pt
EDNET=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
JUNYI=../checkpoints/edubert_junyi_pretrain_full_encoder.pt
T=../processed/junyi
for SEED in 42 1 2; do
  PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill --processed_dir $T --init scratch --n_students 3000 --seed $SEED --epochs 20 --run_type ns_junyi_scratch_mt1_seed${SEED} --wandb
  PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill --processed_dir $T --init pretrained --encoder_ckpt $JUNYI --n_students 3000 --seed $SEED --epochs 20 --run_type ns_junyi_indomain_mt1_seed${SEED} --wandb
  PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill --processed_dir $T --init pretrained --encoder_ckpt $ASSIST --n_students 3000 --seed $SEED --epochs 20 --run_type ns_junyi_fromassist_mt1_seed${SEED} --wandb
  PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill --processed_dir $T --init pretrained --encoder_ckpt $EDNET --n_students 3000 --seed $SEED --epochs 20 --run_type ns_junyi_fromednet_mt1_seed${SEED} --wandb
done
