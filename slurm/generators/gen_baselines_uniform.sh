#!/usr/bin/env bash
# Rebuild the DKT and AKT baseline campaign so that Table 2 is uniform.
#
# Why: the current baseline table cannot be defended as it stands.
#   1. Two AKT campaigns exist and both print "=== AKT on <ds> ===", one at
#      full scale / 100 epochs (akt_baseline_seed*) and one at N=3000 /
#      50 epochs (akt_<ds>_n3000_seed*). Any grep over the logs silently
#      averages them.
#   2. Neither AKT campaign matches the EduBERT budget on EdNet (20,000) or
#      Junyi (40,000), so those rows were never budget-matched.
#   3. DKT on ASSISTments 2017 has two runs, one with no recoverable seed.
#   4. Baselines ran 3 seeds while EduBERT ran 6 on most rows.
#
# This runs every baseline cell once, under a fresh run_type prefix (base2_)
# that cannot collide with either old campaign, at the budget that matches the
# EduBERT row, at 6 seeds, one run per sbatch so each log holds one banner.
#
# Budgets are matched to the EduBERT runs already in the logs:
#   assist2017  full training split (1,366), EduBERT ran at N=3000 which is
#               the whole split, so no --n_students here either
#   ednet       20,000, matching edubert_ednet_ktfull_*_n20000
#   junyi       40,000, matching edubert_junyi_ktfull_*_n40000
#   the other four   --n_students 100000, matching gen_scratch_new.sh and the
#                    *abl_full runs, which is the full split on all four
#
# Writes 84 files into queue_base2/. Run:
#   bash slurm/generators/gen_baselines_uniform.sh
#   bash tools/drip_submit.sh queue_base2

set -uo pipefail

CODE=/projects/algl/dai.hany/studentbert/code
PY=/projects/algl/dai.hany/envs/sb/bin/python
QDIR="$CODE/queue_base2"

cd "$CODE" || exit 1
mkdir -p "$QDIR"
rm -f "$QDIR"/*.sbatch

SEEDS="42 1 2 3 4 5"
EPOCHS=30

count=0
for DS in assist2017 ednet junyi algebra2005 bridge2006 assist2009 algebra2006; do
  case "$DS" in
    assist2017) BUDGET="" ;;
    ednet)      BUDGET="--n_students 20000" ;;
    junyi)      BUDGET="--n_students 40000" ;;
    *)          BUDGET="--n_students 100000" ;;
  esac
  if [ ! -d "$CODE/../processed/$DS" ]; then
    echo "MISSING DATASET DIR: ../processed/$DS"
    exit 1
  fi
  for MODEL in dkt akt; do
    for SEED in $SEEDS; do
      NAME="base2_${MODEL}_${DS}_seed${SEED}"
      F="$QDIR/${NAME}.sbatch"
      {
        echo '#!/bin/bash'
        echo '#SBATCH --partition=gpu'
        echo '#SBATCH --gres=gpu:1'
        echo '#SBATCH --cpus-per-task=8'
        echo '#SBATCH --time=08:00:00'
        echo '#SBATCH --mem=48G'
        echo "#SBATCH --output=${NAME}_%j.log"
        echo "cd $CODE"
        echo "PYTHONPATH=. $PY scripts/train_baseline.py --model $MODEL --processed_dir ../processed/$DS $BUDGET --seed $SEED --epochs $EPOCHS --run_type $NAME --wandb"
      } > "$F"
      count=$((count + 1))
    done
  done
done

echo "wrote $count sbatch files to $QDIR"
echo "budgets: assist2017 full, ednet 20000, junyi 40000, other four 100000"
echo "submit with: bash tools/drip_submit.sh queue_base2"
