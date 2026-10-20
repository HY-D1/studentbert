#!/bin/bash
# OPTION 1: NLP EARLY FINE-TUNES (approved by Prof. Hazra; design in docs/option1_nlp_design.md).
#
# PROBE=1 writes the 5-run timing and protocol check (bert-base-uncased, [CLS] pooling, seed 42,
# one run per task). Without PROBE it writes the full set: one GPU job per encoder and task, each
# running both poolings on the 3 estimator seeds (6 runs of 3 epochs), 35 jobs and 210 runs. The
# fine-tune skips an output that already exists, and this generator never rewrites a job whose
# sbatch file is in the queue's done/ folder, so a re-run never submits twice. Submit with
# tools/drip_multi.sh in tmux.
#
#   PROBE=1 bash slurm/generators/gen_task4_early_ft.sh
#   bash slurm/generators/gen_task4_early_ft.sh

set -euo pipefail
ROOT=${ROOT:-/projects/algl/dai.hany}
CODE=$ROOT/studentbert/code
OUT=$ROOT/task4/early_ft
T4PY=$ROOT/envs/t4/bin/python
MODELS="bert-base-uncased roberta-base distilbert-base-uncased emilyalsentzer/Bio_ClinicalBERT dmis-lab/biobert-v1.1 cardiffnlp/twitter-roberta-base allenai/scibert_scivocab_uncased"
TASKS="agnews mnli qnli rte airline"
if [ "${PROBE:-0}" = "1" ]; then
  Q=$CODE/queue_t4ft_probe; MODELS_RUN="bert-base-uncased"; POOLS="cls"; SEEDS="42"; WALL=01:00:00
else
  Q=$CODE/queue_t4ft; MODELS_RUN="$MODELS"; POOLS="cls mean"; SEEDS="42 1 2"; WALL=03:00:00
fi
mkdir -p "$Q" "$Q/done" "$OUT/logs"
rm -f "$Q"/*.sbatch
n=0; skipped=0
for model in $MODELS_RUN; do
  slug=${model//\//__}
  for task in $TASKS; do
    name="t4ft_${slug}_${task}"
    [ "${PROBE:-0}" = "1" ] && name="t4ftprobe_${task}"
    if [ -f "$Q/done/${name}.sbatch" ]; then
      skipped=$((skipped + 1))
      continue
    fi
    {
      echo '#!/bin/bash'
      echo "#SBATCH --job-name=${name}.sbatch"
      echo '#SBATCH --partition=gpu'
      echo '#SBATCH --exclude=d1020'
      echo '#SBATCH --gres=gpu:v100-sxm2:1'
      echo '#SBATCH --cpus-per-task=8'
      echo '#SBATCH --mem=32G'
      echo "#SBATCH --time=${WALL}"
      echo "#SBATCH --output=$OUT/logs/%x_%j.log"
      echo 'set -euo pipefail'
      echo "cd $CODE"
      echo "export HF_HOME=$ROOT/hf_cache PYTHONPATH=."
      echo "for pooling in $POOLS; do"
      echo "  for seed in $SEEDS; do"
      echo "    $T4PY scripts/task4_early_ft.py --model $model --task $task --pooling \$pooling --seed \$seed --out_dir $OUT"
      echo '  done'
      echo 'done'
      echo "echo \"done: $model $task\""
    } > "$Q/${name}.sbatch"
    n=$((n + 1))
  done
done
echo "jobs written: $n in $Q (skipped, already submitted: $skipped)"
