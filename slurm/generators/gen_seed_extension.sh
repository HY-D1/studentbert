#!/usr/bin/env bash
# Extend the N=3000 source-comparison grid from seeds {42,1,2} to {42,1,2,3,4,5}.
#
# Why: with 3 seeds the exact one-sided sign test floors at 0.125 and cannot
# reach p<0.05 however unanimous the result. At 6 seeds the floor is 1/64 =
# 0.0156, and a paired bootstrap over 6 seeds gives a real 95% CI. This is the
# single largest credibility upgrade available to sections 6.2 and 6.3.
#
# One run per sbatch on purpose: a TIMEOUT or NODE_FAIL then costs one cell,
# not twelve, and each log holds exactly one banner so log attribution is
# unambiguous.
#
# No `source ~/.bashrc` and no `conda activate`: that file no longer exists on
# this cluster and its absence has already killed jobs at wandb.init. The
# interpreter is called by absolute path.
#
# Writes 72 files into queue_seeds456/. Run:
#   bash slurm/generators/gen_seed_extension.sh
#   bash tools/drip_submit.sh queue_seeds456

set -uo pipefail

CODE=/projects/algl/dai.hany/studentbert/code
PY=/projects/algl/dai.hany/envs/sb/bin/python
QDIR="$CODE/queue_seeds456"

cd "$CODE" || exit 1
mkdir -p "$QDIR"
rm -f "$QDIR"/*.sbatch

ASSIST=../checkpoints/edubert_assist2017_pretrain_full_encoder.pt
EDNET=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
JUNYI=../checkpoints/edubert_junyi_pretrain_full_encoder.pt
N=3000
NEW_SEEDS="3 4 5"

for CK in "$ASSIST" "$EDNET" "$JUNYI"; do
  if [ ! -f "$CODE/$CK" ]; then
    echo "MISSING ENCODER: $CK"
    exit 1
  fi
done

count=0
for TARGET in assist2017 ednet junyi; do
  case "$TARGET" in
    assist2017) TOK=assist; INDOM="$ASSIST"; C1=fromednet;  K1="$EDNET";  C2=fromjunyi; K2="$JUNYI" ;;
    ednet)      TOK=ednet;  INDOM="$EDNET";  C1=fromassist; K1="$ASSIST"; C2=fromjunyi; K2="$JUNYI" ;;
    junyi)      TOK=junyi;  INDOM="$JUNYI";  C1=fromassist; K1="$ASSIST"; C2=fromednet; K2="$EDNET" ;;
  esac
  for SEED in $NEW_SEEDS; do
    for TASK in kt ns; do
      for COND in scratch indomain "$C1" "$C2"; do
        case "$COND" in
          scratch)  INIT="--init scratch" ;;
          indomain) INIT="--init pretrained --encoder_ckpt $INDOM" ;;
          "$C1")    INIT="--init pretrained --encoder_ckpt $K1" ;;
          "$C2")    INIT="--init pretrained --encoder_ckpt $K2" ;;
        esac
        if [ "$TASK" = "kt" ]; then
          SCRIPT="scripts/finetune_edubert.py"
          EXTRA=""
        else
          SCRIPT="scripts/downstream_edubert.py"
          EXTRA="--task next_skill"
        fi
        NAME="${TASK}_${TOK}_${COND}_n${N}_seed${SEED}"
        F="$QDIR/${NAME}.sbatch"
        {
          echo '#!/bin/bash'
          echo '#SBATCH --partition=gpu'
          echo '#SBATCH --gres=gpu:1'
          echo '#SBATCH --cpus-per-task=8'
          echo '#SBATCH --time=08:00:00'
          echo '#SBATCH --mem=48G'
          echo "#SBATCH --output=${NAME}_%j.log"
          echo "cd $CODE"
          echo "PYTHONPATH=. $PY $SCRIPT $EXTRA --processed_dir ../processed/$TARGET $INIT --n_students $N --seed $SEED --epochs 20 --run_type $NAME --wandb"
        } > "$F"
        count=$((count + 1))
      done
    done
  done
done

echo "wrote $count sbatch files to $QDIR"
ls "$QDIR" | head -6
echo "..."
echo "submit with: bash tools/drip_submit.sh queue_seeds456"
