#!/bin/bash
# 7th-dataset ablation: EdNet-source objective encoders -> Algebra2006-2007, 6 seeds.
# Tests the skill-driven prediction (pps 2.41 > boundary 0.27).
# 1048 train students; n_students large = use all. Run: bash gen_alg06_ablation.sh
cd /projects/algl/dai.hany/studentbert/code
FULL=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
CORR=../checkpoints/edubert_ednet_pretrain_ednet_correct_only_encoder.pt
SKIL=../checkpoints/edubert_ednet_pretrain_ednet_skill_only_encoder.pt

rm -f w8_alg06abl_*.sbatch

for S in 42 1 2 3 4 5; do
  for OBJ in full correct_only skill_only; do
    case $OBJ in
      full) CK=$FULL ;;
      correct_only) CK=$CORR ;;
      skill_only) CK=$SKIL ;;
    esac
    cat > w8_alg06abl_${OBJ}_s${S}.sbatch <<SBATCH
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --mem=32G
#SBATCH --output=w8_alg06abl_${OBJ}_s${S}_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
PYTHONPATH=. python scripts/finetune_edubert.py --processed_dir ../processed/algebra2006 --init pretrained --encoder_ckpt ${CK} --n_students 100000 --seed ${S} --epochs 30 --run_type alg06abl_${OBJ}_seed${S} --wandb
SBATCH
  done
done
echo "wrote $(ls w8_alg06abl_*.sbatch 2>/dev/null | wc -l) jobs (want 18)"
tail -1 w8_alg06abl_full_s42.sbatch
head -1 w8_alg06abl_full_s42.sbatch
