#!/bin/bash
# DKT + AKT + EduBERT-scratch on Algebra2006-2007 (the 7th dataset) so it can
# join the main baseline table. 3 seeds baselines, 6 seeds scratch. Run: bash gen_baselines_alg06.sh
cd /projects/algl/dai.hany/studentbert/code
rm -f w8_base_*algebra2006*.sbatch w8_scratch_algebra2006_*.sbatch
for MODEL in dkt akt; do
  for S in 42 1 2; do
    cat > w8_base_${MODEL}_algebra2006_s${S}.sbatch <<SB
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --mem=32G
#SBATCH --output=w8_base_${MODEL}_algebra2006_s${S}_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
PYTHONPATH=. python scripts/train_baseline.py --model ${MODEL} --processed_dir ../processed/algebra2006 --seed ${S} --epochs 30 --run_type base_${MODEL}_algebra2006_seed${S} --wandb
SB
  done
done
for S in 42 1 2 3 4 5; do
  cat > w8_scratch_algebra2006_s${S}.sbatch <<SB
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --mem=32G
#SBATCH --output=w8_scratch_algebra2006_s${S}_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
PYTHONPATH=. python scripts/finetune_edubert.py --processed_dir ../processed/algebra2006 --init scratch --n_students 100000 --seed ${S} --epochs 30 --run_type scratch_algebra2006_seed${S} --wandb
SB
done
echo "wrote $(ls w8_base_*algebra2006*.sbatch w8_scratch_algebra2006_*.sbatch 2>/dev/null | wc -l) jobs (want 12: 2 baselines x 3 + scratch x 6)"
