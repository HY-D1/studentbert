#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --mem=48G
#SBATCH --output=w6_probe2_targets_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
ASSIST=../checkpoints/edubert_assist2017_pretrain_full_encoder.pt
EDNET=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
JUNYI=../checkpoints/edubert_junyi_pretrain_full_encoder.pt
for SEED in 42 1 2; do
  # EdNet target: scratch, in-domain(ednet), from-assist, from-junyi
  PYTHONPATH=. python scripts/probe_edubert_v2.py --processed_dir ../processed/ednet --init scratch --seed $SEED --epochs 20 --run_type probe2_ednet_scratch_seed${SEED} --wandb
  PYTHONPATH=. python scripts/probe_edubert_v2.py --processed_dir ../processed/ednet --init pretrained --encoder_ckpt $EDNET --seed $SEED --epochs 20 --run_type probe2_ednet_indomain_seed${SEED} --wandb
  PYTHONPATH=. python scripts/probe_edubert_v2.py --processed_dir ../processed/ednet --init pretrained --encoder_ckpt $ASSIST --seed $SEED --epochs 20 --run_type probe2_ednet_fromassist_seed${SEED} --wandb
  PYTHONPATH=. python scripts/probe_edubert_v2.py --processed_dir ../processed/ednet --init pretrained --encoder_ckpt $JUNYI --seed $SEED --epochs 20 --run_type probe2_ednet_fromjunyi_seed${SEED} --wandb
  # Junyi target
  PYTHONPATH=. python scripts/probe_edubert_v2.py --processed_dir ../processed/junyi --init scratch --seed $SEED --epochs 20 --run_type probe2_junyi_scratch_seed${SEED} --wandb
  PYTHONPATH=. python scripts/probe_edubert_v2.py --processed_dir ../processed/junyi --init pretrained --encoder_ckpt $JUNYI --seed $SEED --epochs 20 --run_type probe2_junyi_indomain_seed${SEED} --wandb
  PYTHONPATH=. python scripts/probe_edubert_v2.py --processed_dir ../processed/junyi --init pretrained --encoder_ckpt $ASSIST --seed $SEED --epochs 20 --run_type probe2_junyi_fromassist_seed${SEED} --wandb
  PYTHONPATH=. python scripts/probe_edubert_v2.py --processed_dir ../processed/junyi --init pretrained --encoder_ckpt $EDNET --seed $SEED --epochs 20 --run_type probe2_junyi_fromednet_seed${SEED} --wandb
done
