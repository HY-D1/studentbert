#!/bin/bash
# DKT + AKT baselines on the 3 NEW datasets (Algebra2005, Bridge2006, ASSISTments2009)
# for the NeurIPS 6-dataset table. Same splits, 3 seeds each.
# Run: bash gen_baselines_new.sh
cd /projects/algl/dai.hany/studentbert/code

rm -f w8_base_*.sbatch

for DS in algebra2005 bridge2006 assist2009; do
  for MODEL in dkt akt; do
    for S in 42 1 2; do
      cat > w8_base_${MODEL}_${DS}_s${S}.sbatch <<SBATCH
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=w8_base_${MODEL}_${DS}_s${S}_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
PYTHONPATH=. python scripts/train_baseline.py --model ${MODEL} --processed_dir ../processed/${DS} --seed ${S} --epochs 30 --run_type base_${MODEL}_${DS}_seed${S} --wandb
SBATCH
    done
  done
done
echo "wrote $(ls w8_base_*.sbatch 2>/dev/null | wc -l) jobs (want 18: 2 models x 3 datasets x 3 seeds)"
echo "=== sanity ==="
tail -1 w8_base_dkt_algebra2005_s42.sbatch
head -1 w8_base_dkt_algebra2005_s42.sbatch
