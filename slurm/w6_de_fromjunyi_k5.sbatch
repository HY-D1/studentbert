#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --output=w6_de_fromjunyi_k5_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
for SEED in 42 1 2; do
  PYTHONPATH=. python scripts/downstream_edubert.py --task dropout --processed_dir ../processed/ednet --init pretrained --encoder_ckpt ../checkpoints/edubert_junyi_pretrain_full_encoder.pt --k_prefix 5 --n_students 3000 --seed $SEED --epochs 15 --run_type drop_ednet_fromjunyi_k5_n3000_seed${SEED} --wandb
done
