#!/usr/bin/env bash
# =============================================================================
# gen_probe_indomain.sh
#
# Adds the in-domain column to the 7-dataset probe table. Every existing probe7
# run loaded ../checkpoints/edubert_ednet_pretrain_full_encoder.pt (verified in
# the logs), so section 6 is "EdNet-full encoder vs scratch" with the source held
# fixed. There is no in-domain probe for the 4 newer datasets.
#
# Default: 4 jobs, one per newer dataset, 3 seeds inside each.
#   bash slurm/generators/gen_probe_indomain.sh
# All seven under identical probe7 conditions (7 jobs, 21 runs):
#   ALL7=1 bash slurm/generators/gen_probe_indomain.sh
#
# Built from slurm/w8_probe7_algebra2005_full_s1.sbatch with the same three
# changes as the macro_top1 generator: no .bashrc, absolute interpreter,
# --cpus-per-task=8 and a generous walltime.
#
# run_type uses "indom" so these never collide with the existing
# probe7_<ds>_full_s<S> (EdNet source) or probe7_<ds>_scratch_s<S> runs.
#
# Writes: queue_probe/*.sbatch   (submit with tools/drip_submit.sh queue_probe)
# =============================================================================
set -euo pipefail

CODE=/projects/algl/dai.hany/studentbert/code
PY=/projects/algl/dai.hany/envs/sb/bin/python
QUEUE="$CODE/queue_probe"
mkdir -p "$QUEUE"

if [ "${ALL7:-0}" = "1" ]; then
  SETS="algebra2005 algebra2006 assist2009 bridge2006 assist2017 ednet junyi"
  echo "ALL7=1: generating all seven targets (21 runs)"
else
  SETS="algebra2005 algebra2006 assist2009 bridge2006"
  echo "default: the 4 newer targets only (12 runs); the original 3 already have"
  echo "in-domain probes from W6. Use ALL7=1 for a fully uniform 7-target table."
fi

n=0
missing=0
for DS in $SETS; do
  CKPT="../checkpoints/edubert_${DS}_pretrain_full_encoder.pt"
  if [ ! -f "$CODE/$CKPT" ]; then
    echo "  MISSING encoder for $DS: $CKPT  (skipped)"
    missing=$((missing + 1))
    continue
  fi
  f="$QUEUE/w9_probe_indom_${DS}.sbatch"
  {
    echo "#!/bin/bash"
    echo "#SBATCH --partition=gpu"
    echo "#SBATCH --gres=gpu:1"
    echo "#SBATCH --cpus-per-task=8"
    echo "#SBATCH --time=06:00:00"
    echo "#SBATCH --mem=32G"
    echo "#SBATCH --job-name=probe_indom_${DS}"
    echo "#SBATCH --output=w9_probe_indom_${DS}_%j.log"
    echo "cd $CODE"
    echo ""
    for S in 1 2 42; do
      echo "PYTHONPATH=. $PY scripts/probe_edubert_v2.py \\"
      echo "  --processed_dir ../processed/${DS} --init pretrained \\"
      echo "  --encoder_ckpt ${CKPT} \\"
      echo "  --seed ${S} --epochs 20 --run_type probe7_${DS}_indom_s${S} --wandb"
      echo ""
    done
  } > "$f"
  n=$((n + 1))
done

echo
echo "wrote $n sbatch files to $QUEUE ($((n * 3)) runs)"
[ "$missing" -gt 0 ] && echo "$missing dataset(s) skipped for a missing encoder"
echo
echo "INSPECT ONE BEFORE SUBMITTING:"
echo "  cat $QUEUE/$(ls "$QUEUE" | head -1)"
echo
echo "then: bash tools/drip_submit.sh queue_probe"
