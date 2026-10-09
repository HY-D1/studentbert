#!/usr/bin/env bash
# END-TO-END REVERSAL PIPELINE, CPU PART (step 10 of the 2026-10-06 task).
#
# candidate checkpoints -> features -> reversal-risk prediction -> recommendation and confidence
# -> evaluation. The GPU part, layer and skill-table scores computed from the checkpoints, is
# slurm/generators/gen_layerdiag7.sh; its 7 output files must exist before this runs. Everything
# here reads logs and files only, runs the tests first, and stops at the first failure.
#
#   sbatch --wait -p short --exclude=c0615 --time=01:00:00 --cpus-per-task=4 --mem=16G \
#       --output=reversal_pipeline_%j.log --wrap="bash tools/reversal_pipeline.sh"

set -euo pipefail

CODE="${CODE:-/projects/algl/dai.hany/studentbert/code}"
PY="${PY:-/projects/algl/dai.hany/envs/sb/bin/python}"
BF=benchmark_final
cd "$CODE"
export PYTHONPATH=.

n=$(ls tg1_layerdiag7_kt_*.jsonl 2>/dev/null | wc -l)
if [ "$n" -ne 7 ]; then
  echo "ABORT: $n of 7 layer files; run slurm/generators/gen_layerdiag7.sh first"
  exit 1
fi

"$PY" tests/test_reversal_table.py
"$PY" tests/test_reversal_model.py
"$PY" analysis/early_ft_epochs.py --executions "$BF/executions.tsv" --logdir . \
  --wandb wandb_by_id.jsonl --out "$BF/early_ft_epochs.tsv"
"$PY" analysis/reversal_table.py --executions "$BF/executions.tsv" \
  --gold-cells "$BF/gold_cells.tsv" --early "$BF/early_ft_epochs.tsv" \
  --layers tg1_layerdiag7_kt_*.jsonl --out "$BF/reversal_pairs.tsv"
"$PY" analysis/reversal_model.py --pairs "$BF/reversal_pairs.tsv" --out "$BF/reversal_model"
md5sum "$BF/early_ft_epochs.tsv" "$BF/reversal_pairs.tsv" "$BF/reversal_pairs_targets.tsv" \
  "$BF/reversal_model_pairs.tsv" "$BF/reversal_model_recs.tsv" "$BF/reversal_model_report.md"
echo "reversal pipeline: done"
