#!/usr/bin/env bash
# TG1: gap-fill and estimator jobs for the transfer-guidance benchmark (MRAP Tasks 1 to 3).
#
# Three queues, each written only when named in QUEUES, each refusing to guess a setting that the
# evidence has to supply:
#   scratch  18 KT jobs: the matched scratch control at N=1000 that Track A lacks on assist2017,
#            ednet and junyi (seeds 42 1 2 3 4 5, 30 epochs). TA_BATCH and TA_GPU come from the
#            builder report: the original objabl jobs left no committed job file, and a scratch
#            run on another batch size or GPU pool would not pair with them.
#   probe    42 probe jobs: the masked-skill probe of the EdNet skill_only and correct_only
#            encoders on all 7 targets x seeds 42 1 2, same settings as w8_probe7 (20 epochs,
#            default batch). PROBE_GPU should match the existing probe7 runs (builder report).
#   logme    3 jobs: the leakage-safe KT LogME on Track B, one per target, scoring scratch and
#            the three full encoders at seeds 42 1 2 on the N=3000 learner draw.
#
# Log prefix tg1_ and run-name token tg_ are new on purpose: w9_ and w10_ are already taken
# (w9_mt1_, w9_probe_indom_, w10_nsk_), and a shared run name is what let two campaigns collide
# in the N=3000 grid. A job is not written if any existing log already holds its run name, or if
# drip_submit has already moved it to done/. GPU value "any" means unpinned (--gres=gpu:1).
# No `source ~/.bashrc` and no `conda activate`: the file is gone and has killed jobs before.
#
#   QUEUES=probe PROBE_GPU=v100-sxm2 bash slurm/generators/gen_tg1_jobs.sh
#   TIMING=1 QUEUES=scratch TA_BATCH=64 TA_GPU=v100-sxm2 bash slurm/generators/gen_tg1_jobs.sh
#   DRYRUN=1 bash tools/drip_submit.sh queue_tg1_probe
#   bash tools/drip_submit.sh queue_tg1_probe

set -uo pipefail

CODE="${CODE:-/projects/algl/dai.hany/studentbert/code}"
PY="${PY:-/projects/algl/dai.hany/envs/sb/bin/python}"
QUEUES="${QUEUES:-}"
TIMING="${TIMING:-0}"
DATASETS7="assist2017 ednet junyi algebra2005 bridge2006 assist2009 algebra2006"

if [ -z "$QUEUES" ]; then
  echo "set QUEUES to one or more of: scratch probe logme"
  exit 1
fi
cd "$CODE" || exit 1

KNOWN="$(mktemp)"
trap 'rm -f "$KNOWN"' EXIT
grep -h -o -E "run=[^ ]+" ./*.log 2>/dev/null | sort -u > "$KNOWN"
echo "run names already present in logs: $(wc -l < "$KNOWN")"

written=0
skip_log=0
skip_done=0

gres_line() {
  if [ "$1" = "any" ]; then
    echo "#SBATCH --gres=gpu:1"
    return 0
  fi
  if ! sinfo -p gpu -h -o "%G" | grep -q "gpu:$1:"; then
    echo "GPU type '$1' is not in any gpu GRES on this cluster" >&2
    return 1
  fi
  echo "#SBATCH --gres=gpu:$1:1"
}

emit() {
  local q="$1" name="$2" run="$3" gres="$4" wall="$5" mem="$6" cmd="$7"
  if grep -qxF "run=$run" "$KNOWN"; then
    skip_log=$((skip_log + 1))
    echo "  skip, a log already holds $run"
    return 0
  fi
  if [ -f "$q/done/$name.sbatch" ]; then
    skip_done=$((skip_done + 1))
    return 0
  fi
  {
    echo '#!/bin/bash'
    echo '#SBATCH --partition=gpu'
    echo "$gres"
    echo '#SBATCH --cpus-per-task=8'
    echo "#SBATCH --time=$wall"
    echo "#SBATCH --mem=$mem"
    echo "#SBATCH --output=${name}_%j.log"
    echo "cd $CODE"
    echo "$cmd"
  } > "$q/$name.sbatch"
  written=$((written + 1))
}

seeds() {
  if [ "$TIMING" = "1" ]; then echo "42"; else echo "$1"; fi
}

wants() {
  case " $QUEUES " in *" $1 "*) return 0 ;; *) return 1 ;; esac
}

if wants scratch; then
  : "${TA_BATCH:?set TA_BATCH from the builder report (Track A fine-tune batch size)}"
  : "${TA_GPU:?set TA_GPU from the builder report (a GRES type such as v100-sxm2, or any)}"
  GRES="$(gres_line "$TA_GPU")" || exit 1
  Q="$CODE/queue_tg1_scratch"
  mkdir -p "$Q"
  rm -f "$Q"/*.sbatch
  for pair in assist2017:tg_objabl ednet:tg_regime_ednet junyi:tg_objabl2; do
    DS="${pair%%:*}"
    STEM="${pair##*:}"
    case "$DS" in
      assist2017) WALL=01:00:00 ;;
      junyi) WALL=04:00:00 ;;
      *) WALL=06:00:00 ;;
    esac
    for S in $(seeds "42 1 2 3 4 5"); do
      RT="${STEM}_scratch_n1000_seed${S}"
      emit "$Q" "tg1_scratch_${DS}_n1000_seed${S}" "edubert_${DS}_${RT}" "$GRES" "$WALL" 24G \
        "PYTHONPATH=. $PY scripts/finetune_edubert.py --processed_dir ../processed/$DS --init scratch --n_students 1000 --seed $S --epochs 30 --batch_size $TA_BATCH --run_type $RT --wandb"
    done
  done
  echo "scratch queue: $Q"
fi

if wants probe; then
  : "${PROBE_GPU:?set PROBE_GPU from the builder report (GPU of the existing probe7 runs, or any)}"
  GRES="$(gres_line "$PROBE_GPU")" || exit 1
  for OBJ in skill_only correct_only; do
    if ! grep -l -q "best mlm_loss" w7_pretrain_ednet_${OBJ}_*.log 2>/dev/null; then
      echo "the EdNet $OBJ encoder has no finished pretraining log; refusing to probe it"
      exit 1
    fi
  done
  Q="$CODE/queue_tg1_probe"
  mkdir -p "$Q"
  rm -f "$Q"/*.sbatch
  if [ "$TIMING" = "1" ]; then SET="ednet junyi"; else SET="$DATASETS7"; fi
  for DS in $SET; do
    case "$DS" in
      ednet) WALL=06:00:00 ;;
      junyi) WALL=03:00:00 ;;
      *) WALL=02:00:00 ;;
    esac
    for OBJ in skill_only correct_only; do
      CK="../checkpoints/edubert_ednet_pretrain_ednet_${OBJ}_encoder.pt"
      for S in $(seeds "42 1 2"); do
        RT="tg_probe7_${DS}_${OBJ}_s${S}"
        emit "$Q" "tg1_probe7_${DS}_${OBJ}_s${S}" "edubert_${DS}_${RT}" "$GRES" "$WALL" 32G \
          "PYTHONPATH=. $PY scripts/probe_edubert_v2.py --processed_dir ../processed/$DS --init pretrained --encoder_ckpt $CK --seed $S --epochs 20 --run_type $RT --wandb"
      done
    done
  done
  echo "probe queue: $Q"
fi

if wants logme; then
  : "${LOGME_GPU:?set LOGME_GPU (any is fine: scoring is deterministic per seed and light)}"
  GRES="$(gres_line "$LOGME_GPU")" || exit 1
  CKS=""
  for SRC in assist2017 ednet junyi; do
    CK="../checkpoints/edubert_${SRC}_pretrain_full_encoder.pt"
    if [ ! -f "$CK" ]; then
      echo "MISSING ENCODER: $CK"
      exit 1
    fi
    CKS="$CKS $CK"
  done
  Q="$CODE/queue_tg1_logme"
  mkdir -p "$Q"
  rm -f "$Q"/*.sbatch
  for DS in assist2017 ednet junyi; do
    emit "$Q" "tg1_logme_${DS}" "tg1_logme_${DS}" "$GRES" 02:00:00 32G \
      "PYTHONPATH=. $PY scripts/score_transferability.py --target_dir ../processed/$DS --candidates scratch$CKS --n_students 3000 --seeds $(seeds "42 1 2") --out tg1_logme_kt_${DS}.jsonl"
  done
  echo "logme queue: $Q"
fi

echo "written $written   skipped (log exists) $skip_log   skipped (already submitted) $skip_done"
echo "timing probe: $TIMING"
