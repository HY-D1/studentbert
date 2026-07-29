#!/bin/bash
# Task 1 ANCHOR: K=512 (native ASSIST length, no truncation) to complete the flip.
# This is the high-practice-per-skill skill-driven endpoint the sweep was missing.
# full/skill_only/correct_only, 6 seeds = 18 jobs. Run: bash gen_task1_anchor.sh
cd /projects/algl/dai.hany/studentbert/code
FULL=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
CORR=../checkpoints/edubert_ednet_pretrain_ednet_correct_only_encoder.pt
SKIL=../checkpoints/edubert_ednet_pretrain_ednet_skill_only_encoder.pt

rm -f w8_trunc_*_k512_*.sbatch

for S in 42 1 2 3 4 5; do
  for OBJ in full correct_only skill_only; do
    case $OBJ in
      full) CK=$FULL ;;
      correct_only) CK=$CORR ;;
      skill_only) CK=$SKIL ;;
    esac
    cat > w8_trunc_${OBJ}_k512_s${S}.sbatch <<SBATCH
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --output=w8_trunc_${OBJ}_k512_s${S}_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
PYTHONPATH=. python scripts/finetune_edubert.py --processed_dir ../processed/assist2017 --init pretrained --encoder_ckpt ${CK} --max_seq_len 512 --seed ${S} --epochs 30 --run_type trunc_${OBJ}_k512_seed${S} --wandb
SBATCH
  done
done
echo "wrote $(ls w8_trunc_*_k512_*.sbatch 2>/dev/null | wc -l) anchor jobs (want 18)"
tail -1 w8_trunc_full_k512_s42.sbatch
head -1 w8_trunc_full_k512_s42.sbatch
