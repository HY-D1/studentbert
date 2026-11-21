#!/bin/bash
# NLP EARLY FINE-TUNES TO 10 EPOCHS (docs/nlp_budget_extension_design.md, asked for by Prof. Hazra).
#
# The 630 runs of the three earlier NLP sets (gen_task4_early_ft.sh, gen_phase2_jobs.sh and
# gen_phase3_jobs.sh), rerun with scripts/task4_early_ft.py unchanged and --epochs 10 into new
# folders, so the recorded 3-epoch runs are not touched. Same encoders, tasks, poolings, seeds,
# GPU type, environment and defaults.
#
# PROBE=1 writes the 5-run protocol check (bert-base-uncased, [CLS] pooling, seed 42, one job per
# task) into early_ft10. Without PROBE it writes the full set: one GPU job per encoder, task and
# pooling, running that pooling's 3 seeds, for each seed group (42 1 2 -> early_ft10, 3 4 5 ->
# early_ft10_fresh, 6 7 8 -> early_ft10_p3): 210 jobs and 630 runs. The fine-tune skips an output
# that already exists (so the probe's 5 runs are not repeated), and this generator never rewrites a
# job whose sbatch file is in its queue's done/ folder, so a re-run never submits twice. Submit with
# tools/drip_multi.sh in tmux.
#
#   PROBE=1 bash slurm/generators/gen_nlp_ft10_jobs.sh
#   bash slurm/generators/gen_nlp_ft10_jobs.sh

set -euo pipefail
ROOT=${ROOT:-/projects/algl/dai.hany}
CODE=$ROOT/studentbert/code
T4=$ROOT/task4
T4PY=$ROOT/envs/t4/bin/python
EPOCHS=10
MODELS="bert-base-uncased roberta-base distilbert-base-uncased emilyalsentzer/Bio_ClinicalBERT dmis-lab/biobert-v1.1 cardiffnlp/twitter-roberta-base allenai/scibert_scivocab_uncased"
TASKS="agnews mnli qnli rte airline"

write_job() {
  local q=$1 name=$2 model=$3 task=$4 pools=$5 seeds=$6 out=$7 wall=$8
  {
    echo '#!/bin/bash'
    echo "#SBATCH --job-name=${name}.sbatch"
    echo '#SBATCH --partition=gpu'
    echo '#SBATCH --exclude=d1020'
    echo '#SBATCH --gres=gpu:v100-sxm2:1'
    echo '#SBATCH --cpus-per-task=8'
    echo '#SBATCH --mem=32G'
    echo "#SBATCH --time=${wall}"
    echo "#SBATCH --output=$out/logs/%x_%j.log"
    echo 'set -euo pipefail'
    echo "cd $CODE"
    echo "export HF_HOME=$ROOT/hf_cache PYTHONPATH=."
    echo "for pooling in $pools; do"
    echo "  for seed in $seeds; do"
    echo "    $T4PY scripts/task4_early_ft.py --model $model --task $task --pooling \$pooling --seed \$seed --epochs $EPOCHS --out_dir $out"
    echo '  done'
    echo 'done'
    echo "echo \"done: $model $task $pools\""
  } > "$q/${name}.sbatch"
}

n=0; skipped=0
if [ "${PROBE:-0}" = "1" ]; then
  Q=$CODE/queue_ft10_probe; OUT=$T4/early_ft10
  mkdir -p "$Q" "$Q/done" "$OUT/logs"
  rm -f "$Q"/*.sbatch
  for task in $TASKS; do
    name="t4ft10probe_${task}"
    if [ -f "$Q/done/${name}.sbatch" ]; then skipped=$((skipped + 1)); continue; fi
    write_job "$Q" "$name" bert-base-uncased "$task" cls 42 "$OUT" 01:30:00
    n=$((n + 1))
  done
  echo "jobs written: $n in $Q (skipped, already submitted: $skipped)"
  exit 0
fi

for group in "ft10:42 1 2:early_ft10" "ft10f:3 4 5:early_ft10_fresh" "ft10p3:6 7 8:early_ft10_p3"; do
  tag=${group%%:*}; rest=${group#*:}; seeds=${rest%%:*}; OUT=$T4/${rest#*:}
  Q=$CODE/queue_$tag
  mkdir -p "$Q" "$Q/done" "$OUT/logs"
  rm -f "$Q"/*.sbatch
  for model in $MODELS; do
    slug=${model//\//__}
    for task in $TASKS; do
      for pooling in cls mean; do
        name="t4${tag}_${slug}_${task}_${pooling}"
        if [ -f "$Q/done/${name}.sbatch" ]; then skipped=$((skipped + 1)); continue; fi
        write_job "$Q" "$name" "$model" "$task" "$pooling" "$seeds" "$OUT" 04:00:00
        n=$((n + 1))
      done
    done
  done
done
echo "jobs written: $n in $CODE/queue_ft10, queue_ft10f, queue_ft10p3 (skipped, already submitted: $skipped)"
