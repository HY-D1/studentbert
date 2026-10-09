#!/usr/bin/env bash
# LAYER AND SKILL-TABLE SCORES ON ALL 7 KT TARGETS (reversal-aware pipeline, step 4; 2026-10-08).
#
# Step 4 of the 2026-10-06 task needs, for every held-out target and candidate, H-score at every
# layer and the score with the target-specific skill table reset. scripts/diagnose_logme.py
# computes exactly that (D4, RESULTS.md 12.9) but was run with all 7 encoders on Algebra 2006
# only. This emits one GPU job per target with the same arguments as D4: scratch plus the 7
# full-corpus encoders, the Track B learner draw (--n_students 3000), estimator seeds 42 1 2.
# Algebra 2006 is included again under the new name, so all 7 files share one provenance and the
# rerun can be checked against tg1_logmediag_kt_algebra2006.jsonl.
#
# Output: tg1_layerdiag7_kt_<target>.jsonl in the code root. A target whose output file exists is
# skipped, never overwritten.
#
#   bash slurm/generators/gen_layerdiag7.sh
#   bash tools/drip_multi.sh queue_layerdiag7 [other queues]

set -uo pipefail

CODE="${CODE:-/projects/algl/dai.hany/studentbert/code}"
PY="${PY:-/projects/algl/dai.hany/envs/sb/bin/python}"
Q="$CODE/queue_layerdiag7"
TARGETS="${TARGETS:-assist2017 ednet junyi algebra2005 bridge2006 assist2009 algebra2006}"
SOURCES="assist2017 ednet junyi algebra2005 bridge2006 assist2009 algebra2006"

cd "$CODE" || exit 1
CKS=""
for SRC in $SOURCES; do
  CK="../checkpoints/edubert_${SRC}_pretrain_full_encoder.pt"
  if [ ! -f "$CODE/$CK" ]; then
    echo "MISSING ENCODER: $CK"
    exit 1
  fi
  CKS="$CKS $CK"
done

mkdir -p "$Q"
rm -f "$Q"/*.sbatch
n=0
for DS in $TARGETS; do
  if [ ! -d "$CODE/../processed/$DS" ]; then
    echo "MISSING TARGET: ../processed/$DS"
    exit 1
  fi
  OUT="tg1_layerdiag7_kt_${DS}.jsonl"
  if [ -f "$CODE/$OUT" ]; then
    echo "have $OUT, skipping"
    continue
  fi
  NAME="tg1_layerdiag7_${DS}"
  {
    echo '#!/bin/bash'
    echo '#SBATCH --partition=gpu'
    echo '#SBATCH --gres=gpu:1'
    echo '#SBATCH --exclude=d1020'
    echo '#SBATCH --cpus-per-task=8'
    echo '#SBATCH --time=04:00:00'
    echo '#SBATCH --mem=48G'
    echo "#SBATCH --output=${NAME}_%j.log"
    echo "cd $CODE"
    echo "PYTHONPATH=. $PY scripts/diagnose_logme.py --target_dir ../processed/$DS --candidates scratch$CKS --n_students 3000 --seeds 42 1 2 --out $OUT"
  } > "$Q/${NAME}.sbatch"
  n=$((n + 1))
done
echo "layer-diagnostic jobs written: $n  (queue $Q)"
