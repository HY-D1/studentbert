#!/bin/bash
# EduBERT-scratch on the 3 new datasets for a uniform 6-dataset head-to-head table.
# --init scratch (no encoder), 6 seeds each. Run: bash gen_scratch_new.sh
cd /projects/algl/dai.hany/studentbert/code

rm -f w8_scratch_*.sbatch

for DS in algebra2005 bridge2006 assist2009; do
  for S in 42 1 2 3 4 5; do
    cat > w8_scratch_${DS}_s${S}.sbatch <<SBATCH
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=w8_scratch_${DS}_s${S}_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
PYTHONPATH=. python scripts/finetune_edubert.py --processed_dir ../processed/${DS} --init scratch --n_students 100000 --seed ${S} --epochs 30 --run_type scratch_${DS}_seed${S} --wandb
SBATCH
  done
done
echo "wrote $(ls w8_scratch_*.sbatch 2>/dev/null | wc -l) jobs (want 18: 3 datasets x 6 seeds)"
tail -1 w8_scratch_algebra2005_s42.sbatch
head -1 w8_scratch_algebra2005_s42.sbatch
