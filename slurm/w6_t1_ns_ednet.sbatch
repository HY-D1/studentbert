#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --output=w6_t1_ns_ednet_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
ASSIST=../checkpoints/edubert_assist2017_pretrain_full_encoder.pt
EDNET=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
JUNYI=../checkpoints/edubert_junyi_pretrain_full_encoder.pt
T=../processed/ednet
for SEED in 42 1 2; do
  PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill --processed_dir $T --init scratch --seed $SEED --epochs 20 --run_type ns_ednet_scratch_seed${SEED} --wandb
  PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill --processed_dir $T --init pretrained --encoder_ckpt $EDNET --seed $SEED --epochs 20 --run_type ns_ednet_indomain_seed${SEED} --wandb
  PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill --processed_dir $T --init pretrained --encoder_ckpt $ASSIST --seed $SEED --epochs 20 --run_type ns_ednet_fromassist_seed${SEED} --wandb
  PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill --processed_dir $T --init pretrained --encoder_ckpt $JUNYI --seed $SEED --epochs 20 --run_type ns_ednet_fromjunyi_seed${SEED} --wandb
done
