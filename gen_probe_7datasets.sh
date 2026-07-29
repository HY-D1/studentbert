#!/bin/bash
# Extend the masked-skill probe to all 7 datasets (NeurIPS evidence base).
# Probes EdNet full-objective encoder + scratch on each target, 3 seeds.
# (Set INCLUDE_OBJ=1 to also probe skill_only/correct_only encoders for the
#  mechanism angle - does skill_only decode skill better.)
# Run: bash gen_probe_7datasets.sh
cd /projects/algl/dai.hany/studentbert/code
FULL=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
SKIL=../checkpoints/edubert_ednet_pretrain_ednet_skill_only_encoder.pt
CORR=../checkpoints/edubert_ednet_pretrain_ednet_correct_only_encoder.pt
INCLUDE_OBJ=${INCLUDE_OBJ:-0}

rm -f w8_probe7_*.sbatch

for DS in assist2017 ednet junyi algebra2005 bridge2006 assist2009 algebra2006; do
  for S in 42 1 2; do
    # pretrained (full) probe
    cat > w8_probe7_${DS}_full_s${S}.sbatch <<SB
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --output=w8_probe7_${DS}_full_s${S}_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
PYTHONPATH=. python scripts/probe_edubert_v2.py --processed_dir ../processed/${DS} --init pretrained --encoder_ckpt ${FULL} --seed ${S} --epochs 20 --run_type probe7_${DS}_full_s${S} --wandb
SB
    # scratch probe (floor)
    cat > w8_probe7_${DS}_scratch_s${S}.sbatch <<SB
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --output=w8_probe7_${DS}_scratch_s${S}_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
PYTHONPATH=. python scripts/probe_edubert_v2.py --processed_dir ../processed/${DS} --init scratch --seed ${S} --epochs 20 --run_type probe7_${DS}_scratch_s${S} --wandb
SB
    if [ "$INCLUDE_OBJ" = "1" ]; then
      for OBJ in skill_only correct_only; do
        case $OBJ in skill_only) CK=$SKIL;; correct_only) CK=$CORR;; esac
        cat > w8_probe7_${DS}_${OBJ}_s${S}.sbatch <<SB
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --output=w8_probe7_${DS}_${OBJ}_s${S}_%j.log
source ~/.bashrc
conda activate /projects/algl/dai.hany/envs/sb
cd /projects/algl/dai.hany/studentbert/code
PYTHONPATH=. python scripts/probe_edubert_v2.py --processed_dir ../processed/${DS} --init pretrained --encoder_ckpt ${CK} --seed ${S} --epochs 20 --run_type probe7_${DS}_${OBJ}_s${S} --wandb
SB
      done
    fi
  done
done
echo "wrote $(ls w8_probe7_*.sbatch 2>/dev/null | wc -l) jobs (base: 2 cond x 7 ds x 3 seeds = 42; with INCLUDE_OBJ=1: 84)"
tail -1 w8_probe7_assist2017_full_s42.sbatch
head -1 w8_probe7_assist2017_full_s42.sbatch
