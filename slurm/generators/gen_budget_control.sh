#!/usr/bin/env bash
# BUDGET-MATCHED SOURCE-SCALE CONTROLS (Paper A, Prof. Hazra's items 4 and 5, 2026-10-08).
#
# WHY. Every encoder of the section-11 ladder trains for 10 epochs at batch 128, so the number of
# optimizer updates grows with the corpus (ceil(learners / 128) per epoch: 400 updates at 5,000
# learners, 3,850 at 49,153, 27,630 at 353,597). The original Junyi encoder trained for 20 epochs,
# 7,700 updates, against 3,850 for EdNet cut to the same 49,153 learners. Both comparisons mix
# data with optimization budget. This generator builds encoders whose update count is matched:
#   49153:20  EdNet 49,153 learners for 20 epochs = 7,700 updates, exactly Junyi's (item 5)
#   15000:33  15,000 learners for 33 epochs = 3,894 updates, the 49,153 rung's 3,850 (item 4)
#   5000:96   5,000 learners for 96 epochs = 3,840 updates (item 4, optional)
# Everything else is the section-11 recipe: batch 128, lr 1e-3, warm-up 5% of all updates, the
# same draw seeds (42, 1, 2, so the same sampled students as the ladder), fine-tuning at N=3000
# for 20 epochs with seeds 42 1 2 3 4 5 on v100-sxm2, paired against the existing scratch runs.
#
# NAMES. Encoders are edubert_ednet_pretrain_ednet_n<SIZE>e<EPOCHS>[d<DRAW>]_encoder.pt and
# fine-tunes kt_<tok>_fromednet_n<SIZE>e<EPOCHS>[d<DRAW>]_bud_n3000_seed<SEED>. The "e<EPOCHS>"
# and "_bud_" tokens keep them out of the ladder parser (RX_S11 in build_transfer_benchmark.py)
# and can never overwrite a ladder encoder.
#
# GATE. Fine-tunes are written only for encoders whose pretraining log printed "best mlm_loss";
# a checkpoint appears on every loss improvement, long before the run ends (2026-09-15 lesson).
#
# USAGE (from the code root)
#   bash slurm/generators/gen_budget_control.sh                    # item 5: 3 encoders
#   bash tools/drip_submit.sh queue_budget_pretrain
#   ... after "best mlm_loss" in all three logs, re-run the generator, then ...
#   bash tools/drip_submit.sh queue_budget_ft                      # 18 fine-tunes
#   SPECS="15000:33" TARGETS="assist2017 junyi" bash slurm/generators/gen_budget_control.sh
#                                                                  # item 4, once approved

set -uo pipefail

CODE="${CODE:-/projects/algl/dai.hany/studentbert/code}"
PY="${PY:-/projects/algl/dai.hany/envs/sb/bin/python}"
PQDIR="$CODE/queue_budget_pretrain"
FQDIR="$CODE/queue_budget_ft"
GPUTYPE="${GPUTYPE:-v100-sxm2}"
SEEDS="${SEEDS:-42 1 2 3 4 5}"
DRAWS="${DRAWS:-42 1 2}"
SPECS="${SPECS:-49153:20}"
TARGETS="${TARGETS:-assist2017}"
N=3000

cd "$CODE" || exit 1

if ! sinfo -p gpu -h -o "%G" | grep -q "gpu:${GPUTYPE}:"; then
  echo "GPUTYPE '$GPUTYPE' does not appear in any gpu GRES on this cluster; available:"
  sinfo -p gpu -h -o "%G" | sort -u
  exit 1
fi

for SPEC in $SPECS; do
  case "$SPEC" in
    *:*) ;;
    *) echo "bad SPEC '$SPEC' (expected SIZE:EPOCHS)"; exit 1 ;;
  esac
done
for TARGET in $TARGETS; do
  case "$TARGET" in
    assist2017|junyi) ;;
    *) echo "bad TARGET '$TARGET' (EdNet is foreign only on assist2017 and junyi)"; exit 1 ;;
  esac
done

mkdir -p "$PQDIR" "$FQDIR"
rm -f "$PQDIR"/*.sbatch "$FQDIR"/*.sbatch

tag_of() {
  if [ "$3" = "42" ]; then echo "n$1e$2"; else echo "n$1e$2d$3"; fi
}

pcount=0; have=0; inprogress=0; queued=0
for SPEC in $SPECS; do
  SIZE="${SPEC%%:*}"; EPOCHS="${SPEC##*:}"
  UPDATES=$(( (SIZE + 127) / 128 * EPOCHS ))
  # The ladder's 49,153 x 10 encoder (3,850 updates) finished inside 01:30; allow about twice
  # the scaled time so a slow node draw does not hit the walltime.
  if [ "$UPDATES" -le 4500 ]; then WALL=03:00:00
  elif [ "$UPDATES" -le 9000 ]; then WALL=06:00:00
  else WALL=08:00:00; fi
  for DRAW in $DRAWS; do
    TAG=$(tag_of "$SIZE" "$EPOCHS" "$DRAW")
    CK="../checkpoints/edubert_ednet_pretrain_ednet_${TAG}_encoder.pt"
    if [ ! -f "$CODE/$CK" ] && [ -f "$PQDIR/done/budget_pretrain_ednet_${TAG}.sbatch" ]; then
      echo "submitted earlier, not started yet: budget_pretrain_ednet_${TAG}"
      queued=$((queued + 1))
      continue
    fi
    if [ -f "$CODE/$CK" ]; then
      if grep -l -q "best mlm_loss" budget_pretrain_ednet_${TAG}_*.log 2>/dev/null; then
        echo "have encoder, skipping: $CK"
        have=$((have + 1))
      else
        echo "IN PROGRESS, not usable yet: $CK (no 'best mlm_loss' in its log)"
        inprogress=$((inprogress + 1))
      fi
      continue
    fi
    NAME="budget_pretrain_ednet_${TAG}"
    {
      echo '#!/bin/bash'
      echo '#SBATCH --partition=gpu'
      echo "#SBATCH --gres=gpu:${GPUTYPE}:1"
      echo '#SBATCH --cpus-per-task=8'
      echo "#SBATCH --time=${WALL}"
      echo '#SBATCH --mem=48G'
      echo "#SBATCH --output=${NAME}_%j.log"
      echo "cd $CODE"
      echo "PYTHONPATH=. $PY scripts/pretrain_edubert.py --processed_dir ../processed/ednet --n_students ${SIZE} --seed ${DRAW} --epochs ${EPOCHS} --batch_size 128 --lr 1e-3 --warmup_frac 0.05 --run_type pretrain_ednet_${TAG} --wandb"
    } > "$PQDIR/${NAME}.sbatch"
    pcount=$((pcount + 1))
  done
  echo "spec ${SIZE} learners x ${EPOCHS} epochs = ${UPDATES} updates (walltime ${WALL})"
done

fcount=0; waiting=0; fsub=0
for SPEC in $SPECS; do
  SIZE="${SPEC%%:*}"; EPOCHS="${SPEC##*:}"
  for DRAW in $DRAWS; do
    TAG=$(tag_of "$SIZE" "$EPOCHS" "$DRAW")
    CK="../checkpoints/edubert_ednet_pretrain_ednet_${TAG}_encoder.pt"
    if [ ! -f "$CODE/$CK" ] || ! grep -l -q "best mlm_loss" budget_pretrain_ednet_${TAG}_*.log 2>/dev/null; then
      waiting=$((waiting + 1))
      continue
    fi
    for TARGET in $TARGETS; do
      case "$TARGET" in
        assist2017) TOK=assist; WALL=00:30:00 ;;
        junyi)      TOK=junyi;  WALL=02:00:00 ;;
      esac
      for SEED in $SEEDS; do
        NAME="kt_${TOK}_fromednet_${TAG}_bud_n${N}_seed${SEED}"
        if [ -f "$FQDIR/done/${NAME}.sbatch" ]; then
          fsub=$((fsub + 1))
          continue
        fi
        {
          echo '#!/bin/bash'
          echo '#SBATCH --partition=gpu'
          echo "#SBATCH --gres=gpu:${GPUTYPE}:1"
          echo '#SBATCH --cpus-per-task=8'
          echo "#SBATCH --time=${WALL}"
          echo '#SBATCH --mem=24G'
          echo "#SBATCH --output=${NAME}_%j.log"
          echo "cd $CODE"
          echo "PYTHONPATH=. $PY scripts/finetune_edubert.py --processed_dir ../processed/$TARGET --init pretrained --encoder_ckpt $CK --n_students $N --seed $SEED --epochs 20 --run_type $NAME --wandb"
        } > "$FQDIR/${NAME}.sbatch"
        fcount=$((fcount + 1))
      done
    done
  done
done

echo
echo "encoders to run      : $pcount   (finished: $have, in progress: $inprogress, "\
"submitted earlier: $queued)"
echo "fine-tune jobs       : $fcount   (encoders not finished yet: $waiting, "\
"submitted earlier: $fsub)"
echo "a job whose file is in a queue's done/ folder counts as submitted; to resubmit one"
echo "that failed, delete its file from done/ and re-run this generator"
echo "specs / draws        : $SPECS / $DRAWS"
echo "targets / seeds / gpu: $TARGETS / $SEEDS / $GPUTYPE"
if [ "$pcount" -gt 0 ]; then
  echo "next: bash tools/drip_submit.sh queue_budget_pretrain, then re-run this generator"
fi
if [ "$fcount" -gt 0 ]; then
  echo "next: bash tools/drip_submit.sh queue_budget_ft"
fi
