#!/bin/bash
# Bridge2006 objective ablation to test the correctness-driven prediction (skill-count hypothesis).
# EdNet-source objective encoders (full/skill_only/correct_only) -> Bridge2006, 6 seeds.
# Bridge2006 has 1130 students (904 train); n_students large = use all (no subsampling).
# Run ON CLUSTER from code dir: bash gen_bridge_ablation.sh   (avoids paste mangling)
cd /projects/algl/dai.hany/studentbert/code
FULL=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
CORR=../checkpoints/edubert_ednet_pretrain_ednet_correct_only_encoder.pt
SKIL=../checkpoints/edubert_ednet_pretrain_ednet_skill_only_encoder.pt

rm -f w8_bridgeabl_*.sbatch

for S in 42 1 2 3 4 5; do
  for OBJ in full correct_only skill_only; do
    case $OBJ in
      full) CK=$FULL ;;
      correct_only) CK=$CORR ;;
      skill_only) CK=$SKIL ;;
    esac
    cat > w8_bridgeabl_${OBJ}_s${S}.sbatch <<SBATCH
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --output=w8_bridgeabl_${OBJ}_s${S}_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
PYTHONPATH=. python scripts/finetune_edubert.py --processed_dir ../processed/bridge2006 --init pretrained --encoder_ckpt ${CK} --n_students 100000 --seed ${S} --epochs 30 --run_type bridgeabl_${OBJ}_seed${S} --wandb
SBATCH
  done
done
echo "wrote $(ls w8_bridgeabl_*.sbatch 2>/dev/null | wc -l) jobs (want 18)"
echo "=== sanity: last line + first line of one sbatch ==="
tail -1 w8_bridgeabl_full_s42.sbatch
head -1 w8_bridgeabl_full_s42.sbatch
