#!/bin/bash
# MRAP Task 4 jobs. One GPU job per encoder runs its 4 tasks x 3 seeds of feature extraction
# (measured 2026-09-28: 8.9 s for RTE and 18.6 s for 10,000 MNLI examples with bert-base, plus
# model and dataset loading), so the 2-hour limit is about 10 times the expected run. The
# extractor skips files that already exist, so a resubmitted job only redoes what is missing.
# Scoring runs afterwards as one CPU job on short, which IdleBot never sees.
#
#   bash slurm/generators/gen_task4_jobs.sh
#   ls queue_task4_extract/ queue_task4_score/
# The Airline extension (2026-09-30; SciERC excluded) builds its jobs with
#   TASKS=airline QX=queue_task4_airline_x QS=queue_task4_airline_s bash \
#       slurm/generators/gen_task4_jobs.sh

set -euo pipefail
ROOT=/projects/algl/dai.hany
CODE=$ROOT/studentbert/code
OUT=$ROOT/task4
T4PY=$ROOT/envs/t4/bin/python
SBPY=$ROOT/envs/sb/bin/python
MODELS="bert-base-uncased roberta-base distilbert-base-uncased emilyalsentzer/Bio_ClinicalBERT dmis-lab/biobert-v1.1 cardiffnlp/twitter-roberta-base allenai/scibert_scivocab_uncased"
TASKS=${TASKS:-"agnews mnli qnli rte"}
SEEDS="42 1 2"
QX=${QX:-queue_task4_extract}
QS=${QS:-queue_task4_score}
mkdir -p "$QX" "$QS"

for model in $MODELS; do
  slug=${model//\//__}
  f="$QX/t4x_${slug}.sbatch"
  {
    echo '#!/bin/bash'
    echo "#SBATCH --job-name=t4x_${slug}.sbatch"
    echo '#SBATCH --partition=gpu'
    echo '#SBATCH --exclude=d1020'
    echo '#SBATCH --gres=gpu:1'
    echo '#SBATCH --cpus-per-task=8'
    echo '#SBATCH --mem=32G'
    echo '#SBATCH --time=02:00:00'
    echo "#SBATCH --output=$OUT/logs/%x_%j.log"
    echo 'set -euo pipefail'
    echo "cd $CODE"
    echo "export HF_HOME=$ROOT/hf_cache PYTHONPATH=."
    echo "for task in $TASKS; do"
    echo "  for seed in $SEEDS; do"
    echo "    $T4PY scripts/task4_extract.py --model $model --task \$task --seed \$seed --out_dir $OUT/features"
    echo '  done'
    echo 'done'
    echo "echo \"done: $model\""
  } > "$f"
done

f="$QS/t4s_score.sbatch"
{
  echo '#!/bin/bash'
  echo '#SBATCH --job-name=t4s_score.sbatch'
  echo '#SBATCH --partition=short'
  echo '#SBATCH --cpus-per-task=4'
  echo '#SBATCH --mem=16G'
  echo '#SBATCH --time=01:00:00'
  echo "#SBATCH --output=$OUT/logs/%x_%j.log"
  echo 'set -euo pipefail'
  echo "cd $CODE"
  echo 'export PYTHONPATH=.'
  echo "$SBPY scripts/task4_score.py --features $OUT/features --out $OUT/task4_scores.jsonl"
} > "$f"

echo "wrote $(ls "$QX"/*.sbatch | wc -l) extraction jobs in $QX and $(ls "$QS"/*.sbatch | wc -l) scoring job in $QS"
