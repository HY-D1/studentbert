#!/bin/bash
# Task 2 ablation: EdNet-source objective encoders -> ASSISTments2009, 6 seeds.
# Tests the correctness-driven prediction (pps=0.325 < 1.41 boundary).
# 2495 train students; n_students large = use all. Run: bash gen_a09_ablation.sh
cd /projects/algl/dai.hany/studentbert/code
FULL=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
CORR=../checkpoints/edubert_ednet_pretrain_ednet_correct_only_encoder.pt
SKIL=../checkpoints/edubert_ednet_pretrain_ednet_skill_only_encoder.pt

rm -f w8_a09abl_*.sbatch

for S in 42 1 2 3 4 5; do
  for OBJ in full correct_only skill_only; do
    case $OBJ in
      full) CK=$FULL ;;
      correct_only) CK=$CORR ;;
      skill_only) CK=$SKIL ;;
    esac
    cat > w8_a09abl_${OBJ}_s${S}.sbatch <<SBATCH
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --output=w8_a09abl_${OBJ}_s${S}_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
PYTHONPATH=. python scripts/finetune_edubert.py --processed_dir ../processed/assist2009 --init pretrained --encoder_ckpt ${CK} --n_students 100000 --seed ${S} --epochs 30 --run_type a09abl_${OBJ}_seed${S} --wandb
SBATCH
  done
done
echo "wrote $(ls w8_a09abl_*.sbatch 2>/dev/null | wc -l) jobs (want 18)"
tail -1 w8_a09abl_full_s42.sbatch
head -1 w8_a09abl_full_s42.sbatch
