#!/bin/bash
# Step 3b: objective ablation into Algebra2005 to test the correctness-driven prediction.
# EdNet-source objective encoders (full/skill_only/correct_only) -> Algebra2005, 6 seeds.
# Algebra2005 has only 567 students (453 train), so we use ALL train students
# (n_students larger than the set = no subsampling, uses everything).
# Run ON CLUSTER from code dir: bash gen_algebra_ablation.sh   (avoids paste mangling)
cd /projects/algl/dai.hany/studentbert/code
FULL=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
CORR=../checkpoints/edubert_ednet_pretrain_ednet_correct_only_encoder.pt
SKIL=../checkpoints/edubert_ednet_pretrain_ednet_skill_only_encoder.pt

rm -f w8_algabl_*.sbatch

for S in 42 1 2 3 4 5; do
  for OBJ in full correct_only skill_only; do
    case $OBJ in
      full) CK=$FULL ;;
      correct_only) CK=$CORR ;;
      skill_only) CK=$SKIL ;;
    esac
    cat > w8_algabl_${OBJ}_s${S}.sbatch <<SBATCH
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=w8_algabl_${OBJ}_s${S}_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
PYTHONPATH=. python scripts/finetune_edubert.py --processed_dir ../processed/algebra2005 --init pretrained --encoder_ckpt ${CK} --n_students 100000 --seed ${S} --epochs 30 --run_type algabl_${OBJ}_seed${S} --wandb
SBATCH
  done
done
echo "wrote $(ls w8_algabl_*.sbatch 2>/dev/null | wc -l) jobs (want 18)"
echo "=== sanity: last line of one sbatch (must be full python command) ==="
tail -1 w8_algabl_full_s42.sbatch
echo "=== first line must be #!/bin/bash ==="
head -1 w8_algabl_full_s42.sbatch
