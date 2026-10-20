#!/bin/bash
# PHASE 2 JOBS (docs/phase2_complete_rule_design.md): fresh estimator seeds 3, 4 and 5.
#
# STAGE=extract writes one GPU job per encoder (5 tasks x 3 fresh seeds of frozen features) and one
# CPU scoring job; STAGE=ft writes the early fine-tunes, one GPU job per encoder and task (both
# poolings x 3 fresh seeds, 3 epochs), 35 jobs and 210 runs. Everything goes to separate folders
# (features_fresh, task4_scores_fresh.jsonl, early_ft_fresh), so the frozen results are not
# touched. The scripts skip outputs that already exist, and a job whose sbatch file is in its
# queue's done/ folder is never rewritten. Submit with tools/drip_multi.sh in tmux.
#
#   STAGE=extract bash slurm/generators/gen_phase2_jobs.sh
#   STAGE=ft bash slurm/generators/gen_phase2_jobs.sh

set -euo pipefail
ROOT=${ROOT:-/projects/algl/dai.hany}
CODE=$ROOT/studentbert/code
T4=$ROOT/task4
T4PY=$ROOT/envs/t4/bin/python
SBPY=$ROOT/envs/sb/bin/python
MODELS="bert-base-uncased roberta-base distilbert-base-uncased emilyalsentzer/Bio_ClinicalBERT dmis-lab/biobert-v1.1 cardiffnlp/twitter-roberta-base allenai/scibert_scivocab_uncased"
TASKS="agnews mnli qnli rte airline"
SEEDS="3 4 5"
STAGE=${STAGE:?set STAGE=extract or STAGE=ft}
mkdir -p "$T4/logs"

header() {
  echo '#!/bin/bash'
  echo "#SBATCH --job-name=$1.sbatch"
  echo "#SBATCH --partition=$2"
  [ "$2" = "gpu" ] && echo '#SBATCH --exclude=d1020' && echo "#SBATCH --gres=$3"
  echo "#SBATCH --cpus-per-task=$4"
  echo "#SBATCH --mem=$5"
  echo "#SBATCH --time=$6"
  echo "#SBATCH --output=$T4/logs/%x_%j.log"
  echo 'set -euo pipefail'
  echo "cd $CODE"
  echo "export HF_HOME=$ROOT/hf_cache PYTHONPATH=."
}

n=0; skipped=0
if [ "$STAGE" = "extract" ]; then
  Q=$CODE/queue_p2_extract; QS=$CODE/queue_p2_score
  mkdir -p "$Q/done" "$QS/done"
  rm -f "$Q"/*.sbatch "$QS"/*.sbatch
  for model in $MODELS; do
    slug=${model//\//__}
    name="p2x_${slug}"
    if [ -f "$Q/done/${name}.sbatch" ]; then skipped=$((skipped + 1)); continue; fi
    {
      header "$name" gpu gpu:1 8 32G 02:00:00
      echo "for task in $TASKS; do"
      echo "  for seed in $SEEDS; do"
      echo "    $T4PY scripts/task4_extract.py --model $model --task \$task --seed \$seed --out_dir $T4/features_fresh"
      echo '  done'
      echo 'done'
      echo "echo \"done: $model\""
    } > "$Q/${name}.sbatch"
    n=$((n + 1))
  done
  if [ ! -f "$QS/done/p2s_score.sbatch" ]; then
    {
      header p2s_score short "" 4 16G 01:00:00
      echo "$SBPY scripts/task4_score.py --features $T4/features_fresh --out $T4/task4_scores_fresh.jsonl"
    } > "$QS/p2s_score.sbatch"
    n=$((n + 1))
  else
    skipped=$((skipped + 1))
  fi
elif [ "$STAGE" = "ft" ]; then
  Q=$CODE/queue_p2_ft
  mkdir -p "$Q/done"
  rm -f "$Q"/*.sbatch
  for model in $MODELS; do
    slug=${model//\//__}
    for task in $TASKS; do
      name="p2ft_${slug}_${task}"
      if [ -f "$Q/done/${name}.sbatch" ]; then skipped=$((skipped + 1)); continue; fi
      {
        header "$name" gpu gpu:v100-sxm2:1 8 32G 03:00:00
        echo "for pooling in cls mean; do"
        echo "  for seed in $SEEDS; do"
        echo "    $T4PY scripts/task4_early_ft.py --model $model --task $task --pooling \$pooling --seed \$seed --out_dir $T4/early_ft_fresh"
        echo '  done'
        echo 'done'
        echo "echo \"done: $model $task\""
      } > "$Q/${name}.sbatch"
      n=$((n + 1))
    done
  done
else
  echo "unknown STAGE $STAGE" >&2; exit 1
fi
echo "jobs written: $n (skipped, already submitted: $skipped)"
