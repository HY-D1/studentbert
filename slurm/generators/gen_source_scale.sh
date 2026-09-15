#!/usr/bin/env bash
# MATCHED-SCALE SOURCE ABLATION.
#
# WHAT THIS ANSWERS. Every transfer claim in the paper rests on three sources
# whose SIZE is perfectly confounded with their IDENTITY: platform, population,
# item pool and skill taxonomy all move together with student count. The paper
# concedes this and calls corpus size a predictor rather than a cause. This
# grid holds identity fixed at EdNet KT1 and varies only the number of
# pretraining students, which is the manipulation that separates the two.
#
# READ THE OUTCOMES BEFORE RUNNING, because none of them is a losing result.
#   - Transfer rises with source size, and EdNet-at-1,366 transfers about as
#     poorly as ASSISTments does  ->  scale is causal, "predicts" becomes
#     "causes", which is the single biggest upgrade available to the paper.
#   - EdNet-at-1,366 still transfers well  ->  size is NOT the driver and
#     corpus identity or diversity is. That reverses a headline but it is a
#     sharper finding than the one it replaces.
#   - Flat or noisy  ->  an honest null that bounds how much of the observed
#     ordering size can explain, reported as such.
#
# SECOND RESULT AT ZERO EXTRA COST. Section 5.2 declines to rank sources by
# pretraining loss because the three sources have vocabularies differing by an
# order of magnitude, so their masked-interaction losses are not on a common
# scale. Within EdNet at varying size the vocabulary is CONSTANT, so those
# losses ARE comparable, and this grid gives the first clean test of whether
# pretraining loss predicts transfer. Record the final mlm_loss of every
# encoder; it is printed on the "best mlm_loss" line of each pretrain log.
#
# DESIGN NOTES THAT MATTER.
#   * Targets are assist2017 and junyi ONLY, the two where EdNet is foreign.
#     EdNet as its own target is in-domain and answers a different question.
#   * Task is kt ONLY. The scale claim is scoped to knowledge tracing in the
#     paper, because next-skill reproduces the ordering on just one of three
#     targets. Adding ns would double the cost for a claim the paper does not
#     make. Set TASKS="kt ns" if that scope ever changes.
#   * Target budget is N=3000 and seeds are 42 1 2 3 4 5, IDENTICAL to the
#     existing n3000 grid, so every new point is directly comparable to
#     Table 3 and the paired gains reuse the scratch runs already in hand:
#     kt_assist_scratch_n3000_seed* and kt_junyi_scratch_n3000_seed*.
#     No scratch jobs are emitted here. That is deliberate, not an omission.
#   * GPU is pinned to v100-sxm2 by default because those existing scratch
#     controls ran on v100-sxm2. Pairing a new pretrained run against an old
#     scratch run across different silicon would reintroduce exactly the
#     hardware confound gen_n3000_grid.sh was rewritten to remove.
#   * ONE source draw per size by default. The draw seed is the pretrain
#     --seed, so extra draws at a fixed size are available later without
#     touching this file: DRAWS="1 2" SIZES="1366" bash <this script>.
#     Worth doing at the smallest size if the curve turns on it, because a
#     single small draw is the point a reviewer will press hardest.
#
# USAGE
#   bash slurm/generators/gen_source_scale.sh            # emit both queues
#   bash tools/drip_submit.sh queue_srcscale_pretrain    # encoders first
#   ... wait for all encoders to exist, then ...
#   bash slurm/generators/gen_source_scale.sh            # re-run to pick them up
#   bash tools/drip_submit.sh queue_srcscale_ft
#
# The second invocation is not a mistake: the fine-tune queue can only be
# written once the encoders it points at exist, and this script refuses to
# emit fine-tune jobs for a checkpoint that is not on disk.

set -uo pipefail

CODE=/projects/algl/dai.hany/studentbert/code
PY=/projects/algl/dai.hany/envs/sb/bin/python
PQDIR="$CODE/queue_srcscale_pretrain"
FQDIR="$CODE/queue_srcscale_ft"
GPUTYPE="${GPUTYPE:-v100-sxm2}"
SEEDS="${SEEDS:-42 1 2 3 4 5}"
DRAWS="${DRAWS:-42}"
TASKS="${TASKS:-kt}"
N=3000

# Log-spaced ladder. 1366 and 49153 are the ASSISTments and Junyi TRAINING
# split sizes, so those two points are exact size matches to the other two
# sources. 353597 is EdNet's full training split and already exists as
# edubert_ednet_pretrain_full_encoder.pt, so it is NOT re-run here; it is the
# top of the curve for free.
SIZES="${SIZES:-1366 5000 15000 49153 150000}"

cd "$CODE" || exit 1

FULL=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
if [ ! -f "$CODE/$FULL" ]; then
  echo "MISSING: $FULL   (this is the free top-of-curve point)"
  exit 1
fi

if ! sinfo -p gpu -h -o "%G" | grep -q "gpu:${GPUTYPE}:"; then
  echo "GPUTYPE '$GPUTYPE' does not appear in any gpu GRES on this cluster."
  echo "available:"
  sinfo -p gpu -h -o "%G" | sort -u
  exit 1
fi

mkdir -p "$PQDIR" "$FQDIR"
rm -f "$PQDIR"/*.sbatch "$FQDIR"/*.sbatch

# ---------------------------------------------------------------- encoders
pcount=0
have=0
for SIZE in $SIZES; do
  for DRAW in $DRAWS; do
    if [ "$DRAW" = "42" ]; then
      TAG="n${SIZE}"
    else
      TAG="n${SIZE}d${DRAW}"
    fi
    CK="../checkpoints/edubert_ednet_pretrain_ednet_${TAG}_encoder.pt"
    if [ -f "$CODE/$CK" ]; then
      echo "have encoder, skipping: $CK"
      have=$((have + 1))
      continue
    fi
    if [ "$SIZE" -ge 100000 ]; then
      WALL=03:00:00
    elif [ "$SIZE" -ge 20000 ]; then
      WALL=01:30:00
    else
      WALL=00:40:00
    fi
    NAME="srcscale_pretrain_ednet_${TAG}"
    {
      echo '#!/bin/bash'
      echo '#SBATCH --partition=gpu'
      echo "#SBATCH --gres=gpu:${GPUTYPE}:1"
      echo '#SBATCH --cpus-per-task=8'
      echo "#SBATCH --time=${WALL}"
      echo '#SBATCH --mem=48G'
      echo "#SBATCH --output=${NAME}_%j.log"
      echo "cd $CODE"
      echo "PYTHONPATH=. $PY scripts/pretrain_edubert.py --processed_dir ../processed/ednet --n_students ${SIZE} --seed ${DRAW} --epochs 10 --batch_size 128 --lr 1e-3 --warmup_frac 0.05 --run_type pretrain_ednet_${TAG} --wandb"
    } > "$PQDIR/${NAME}.sbatch"
    pcount=$((pcount + 1))
  done
done

# --------------------------------------------------------------- fine-tune
fcount=0
missing=0
for SIZE in $SIZES; do
  for DRAW in $DRAWS; do
    if [ "$DRAW" = "42" ]; then
      TAG="n${SIZE}"
    else
      TAG="n${SIZE}d${DRAW}"
    fi
    CK="../checkpoints/edubert_ednet_pretrain_ednet_${TAG}_encoder.pt"
    if [ ! -f "$CODE/$CK" ]; then
      missing=$((missing + 1))
      continue
    fi
    for TARGET in assist2017 junyi; do
      case "$TARGET" in
        assist2017) TOK=assist; WALL=00:30:00; MEM=24G ;;
        junyi)      TOK=junyi;  WALL=02:00:00; MEM=24G ;;
      esac
      for SEED in $SEEDS; do
        for TASK in $TASKS; do
          if [ "$TASK" = "kt" ]; then
            SCRIPT="scripts/finetune_edubert.py"
            EXTRA=""
          else
            SCRIPT="scripts/downstream_edubert.py"
            EXTRA="--task next_skill"
          fi
          NAME="${TASK}_${TOK}_fromednet_${TAG}_src_n${N}_seed${SEED}"
          {
            echo '#!/bin/bash'
            echo '#SBATCH --partition=gpu'
            echo "#SBATCH --gres=gpu:${GPUTYPE}:1"
            echo '#SBATCH --cpus-per-task=8'
            echo "#SBATCH --time=${WALL}"
            echo "#SBATCH --mem=${MEM}"
            echo "#SBATCH --output=${NAME}_%j.log"
            echo "cd $CODE"
            echo "PYTHONPATH=. $PY $SCRIPT $EXTRA --processed_dir ../processed/$TARGET --init pretrained --encoder_ckpt $CK --n_students $N --seed $SEED --epochs 20 --run_type $NAME --wandb"
          } > "$FQDIR/${NAME}.sbatch"
          fcount=$((fcount + 1))
        done
      done
    done
  done
done

echo
echo "encoders to run   : $pcount   (already on disk: $have)"
echo "fine-tune jobs    : $fcount   (sizes still missing an encoder: $missing)"
echo "gpu type          : $GPUTYPE"
echo "sizes             : $SIZES"
echo "draws per size    : $DRAWS"
echo "target seeds      : $SEEDS"
echo "tasks             : $TASKS"
echo
if [ "$pcount" -gt 0 ]; then
  echo "step 1: bash tools/drip_submit.sh queue_srcscale_pretrain"
  echo "step 2: once every encoder exists, RE-RUN this generator, then:"
fi
echo "step 3: bash tools/drip_submit.sh queue_srcscale_ft"
echo
echo "scratch controls are NOT emitted: the paired gains reuse the existing"
echo "kt_assist_scratch_n3000_seed* and kt_junyi_scratch_n3000_seed* runs,"
echo "same budget, same seeds, same GPU type."
