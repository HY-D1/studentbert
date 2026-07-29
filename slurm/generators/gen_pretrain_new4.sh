#!/bin/bash
# OPTION B: pretrain EduBERT (full objective) on the 4 new datasets so each has
# its OWN learned skill embedding for the embedding analysis (regime split).
# Uses data ALREADY on disk (../processed/<ds>) - NO downloads needed.
# Produces edubert_<ds>_pretrain_full_encoder.pt for each.
# Run: bash gen_pretrain_new4.sh
cd /projects/algl/dai.hany/studentbert/code

rm -f w8_pretrain_new_*.sbatch

# walltime by dataset size (algebra2006 is big, 547MB / 1.8M interactions)
for DS in algebra2005 bridge2006 assist2009 algebra2006; do
  case $DS in
    algebra2006) WALL="07:00:00"; MEM="48G" ;;   # largest
    bridge2006)  WALL="06:00:00"; MEM="48G" ;;
    *)           WALL="05:00:00"; MEM="32G" ;;
  esac
  cat > w8_pretrain_new_${DS}.sbatch <<SB
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=${WALL}
#SBATCH --mem=${MEM}
#SBATCH --output=w8_pretrain_new_${DS}_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
PYTHONPATH=. python scripts/pretrain_edubert.py --processed_dir ../processed/${DS} --epochs 50 --run_type pretrain_full --wandb
SB
done
echo "wrote $(ls w8_pretrain_new_*.sbatch 2>/dev/null | wc -l) jobs (want 4)"
echo "produces: edubert_{algebra2005,bridge2006,assist2009,algebra2006}_pretrain_full_encoder.pt"
tail -1 w8_pretrain_new_algebra2005.sbatch
head -1 w8_pretrain_new_algebra2005.sbatch
