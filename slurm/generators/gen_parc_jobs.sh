#!/bin/bash
# PARC on the StudentBERT benchmark: one GPU job per target, scoring scratch and the seven full
# encoders at estimator seeds 42, 1 and 2 with scripts/score_parc.py (settings fixed in
# docs/parc_design.md). The NLP side runs on the CPU with scripts/task4_score_parc.py and needs no
# queue. A job whose sbatch file is already in the queue's done/ folder is never written again.
#
#   bash slurm/generators/gen_parc_jobs.sh
#   tmux new-session -d -s parc "bash tools/drip_multi.sh queue_parc_kt > drip_parc.log 2>&1"
set -euo pipefail
CODE="${CODE:-/projects/algl/dai.hany/studentbert/code}"
PY="${PY:-/projects/algl/dai.hany/envs/sb/bin/python}"
DATASETS7="assist2017 ednet junyi algebra2005 bridge2006 assist2009 algebra2006"
cd "$CODE"
CKS=""
for SRC in $DATASETS7; do
  CK="../checkpoints/edubert_${SRC}_pretrain_full_encoder.pt"
  if [ ! -f "$CK" ]; then
    echo "MISSING ENCODER: $CK"
    exit 1
  fi
  CKS="$CKS $CK"
done
Q="$CODE/queue_parc_kt"
mkdir -p "$Q/done"
written=0
skipped=0
for DS in $DATASETS7; do
  NAME="parc_kt_${DS}"
  if [ -f "$Q/done/$NAME.sbatch" ]; then
    skipped=$((skipped + 1))
    continue
  fi
  {
    echo '#!/bin/bash'
    echo '#SBATCH --partition=gpu'
    echo '#SBATCH --exclude=d1020'
    echo '#SBATCH --gres=gpu:1'
    echo '#SBATCH --cpus-per-task=8'
    echo '#SBATCH --time=03:00:00'
    echo '#SBATCH --mem=48G'
    echo "#SBATCH --output=${NAME}_%j.log"
    echo "cd $CODE"
    echo "PYTHONPATH=. $PY scripts/score_parc.py --target_dir ../processed/$DS --candidates scratch$CKS --n_students 3000 --seeds 42 1 2 --out tg1_parc_kt_${DS}.jsonl"
  } > "$Q/$NAME.sbatch"
  written=$((written + 1))
done
echo "jobs written: $written (skipped, already submitted: $skipped)"
