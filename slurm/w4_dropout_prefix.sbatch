#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --output=dropout_prefix_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code

# Sweep prefix length K x {scratch, ednet} to isolate horizon vs objective-mismatch.
# scratch = baseline; ednet = best cross-dataset encoder (the one that underperformed
# scratch on dropout at K=20). If the scratch-minus-ednet gap shrinks as K grows ->
# short-prefix/OOD effect; if it persists at K=200 -> deeper objective mismatch.
for K in 20 50 100 200; do
  PYTHONPATH=. python scripts/downstream_edubert.py --task dropout \
    --processed_dir ../processed/assist2017 --init scratch \
    --k_prefix $K --epochs 30 --run_type scratch_dropout_k${K} --wandb
  PYTHONPATH=. python scripts/downstream_edubert.py --task dropout \
    --processed_dir ../processed/assist2017 --init pretrained \
    --encoder_ckpt ../checkpoints/edubert_ednet_pretrain_full_encoder.pt \
    --k_prefix $K --epochs 30 --run_type ednet_dropout_k${K} --wandb
done
