#!/bin/bash
# TASK 1: ASSIST synthetic sequence-length manipulation.
# Truncate ASSIST to max_seq_len K (holds n_skills=102, n_students=1708 fixed;
# base rate drifts only 0.40-0.44, immaterial + already-refuted variable).
# Vary practice-per-skill = K/102 across the threshold, run objective ablation,
# check if the regime FLIPS. Includes scratch as a data-quantity control.
# Run ON CLUSTER from code dir: bash gen_task1_truncation.sh   (avoid paste mangling)
cd /projects/algl/dai.hany/studentbert/code
FULL=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
CORR=../checkpoints/edubert_ednet_pretrain_ednet_correct_only_encoder.pt
SKIL=../checkpoints/edubert_ednet_pretrain_ednet_skill_only_encoder.pt

rm -f w8_trunc_*.sbatch

for K in 10 20 40 80 160 320; do
  for S in 42 1 2 3 4 5; do
    # three pretrained-objective conditions
    for OBJ in full correct_only skill_only; do
      case $OBJ in
        full) CK=$FULL ;;
        correct_only) CK=$CORR ;;
        skill_only) CK=$SKIL ;;
      esac
      cat > w8_trunc_${OBJ}_k${K}_s${S}.sbatch <<SBATCH
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --output=w8_trunc_${OBJ}_k${K}_s${S}_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
PYTHONPATH=. python scripts/finetune_edubert.py --processed_dir ../processed/assist2017 --init pretrained --encoder_ckpt ${CK} --max_seq_len ${K} --seed ${S} --epochs 30 --run_type trunc_${OBJ}_k${K}_seed${S} --wandb
SBATCH
    done
    # scratch control (data-quantity baseline, no objective variant)
    cat > w8_trunc_scratch_k${K}_s${S}.sbatch <<SBATCH
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --output=w8_trunc_scratch_k${K}_s${S}_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
PYTHONPATH=. python scripts/finetune_edubert.py --processed_dir ../processed/assist2017 --init scratch --max_seq_len ${K} --seed ${S} --epochs 30 --run_type trunc_scratch_k${K}_seed${S} --wandb
SBATCH
  done
done
echo "wrote $(ls w8_trunc_*.sbatch 2>/dev/null | wc -l) jobs (want 144: 4 cond x 6 K x 6 seeds)"
echo "=== sanity ==="
tail -1 w8_trunc_full_k80_s42.sbatch
head -1 w8_trunc_full_k80_s42.sbatch
