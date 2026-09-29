#!/bin/bash
# Failure-review D6 jobs: one CPU job per target on short (IdleBot never sees it), scratch and
# all 7 full encoders, 3 seeds, the estimators' own 3,000-learner draw and 50,000-position cap.
# scripts/frozen_gold_kt.py skips finished (candidate, seed) pairs, so a resubmitted job only
# redoes what is missing. Runtime is unmeasured; the FROZEN lines in each log time every pair.
#
#   bash slurm/generators/gen_frozen_gold_jobs.sh
#   ls queue_frozen_gold/

set -euo pipefail
ROOT=/projects/algl/dai.hany
CODE=$ROOT/studentbert/code
OUT=$ROOT/frozen_gold
SBPY=$ROOT/envs/sb/bin/python
DATASETS7="assist2017 ednet junyi algebra2005 bridge2006 assist2009 algebra2006"
Q=${Q:-queue_frozen_gold}
mkdir -p "$Q" "$OUT/logs"

CKS=""
for SRC in $DATASETS7; do
  ck="../checkpoints/edubert_${SRC}_pretrain_full_encoder.pt"
  if [ ! -f "$CODE/$ck" ]; then
    echo "MISSING ENCODER: $ck"
    exit 1
  fi
  CKS="$CKS $ck"
done

for DS in $DATASETS7; do
  f="$Q/fg_${DS}.sbatch"
  {
    echo '#!/bin/bash'
    echo "#SBATCH --job-name=fg_${DS}.sbatch"
    echo '#SBATCH --partition=short'
    echo '#SBATCH --nodes=1'
    echo '#SBATCH --cpus-per-task=8'
    echo '#SBATCH --mem=32G'
    echo '#SBATCH --time=08:00:00'
    echo "#SBATCH --output=$OUT/logs/%x_%j.log"
    echo 'set -euo pipefail'
    echo "cd $CODE"
    echo 'export PYTHONPATH=. OMP_NUM_THREADS=8 MKL_NUM_THREADS=8'
    echo "$SBPY scripts/frozen_gold_kt.py --target_dir ../processed/$DS --candidates scratch$CKS --n_students 3000 --seeds 42 1 2 --device cpu --out $OUT/frozen_gold_kt_${DS}.jsonl"
  } > "$f"
done

echo "wrote $(ls "$Q"/*.sbatch | wc -l) jobs in $Q"
