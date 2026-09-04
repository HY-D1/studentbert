#!/usr/bin/env bash
# Regenerate the ENTIRE N=3000 source-comparison grid: 3 targets x 2 tasks x
# 4 conditions x 6 seeds = 144 single-run jobs, all on one pinned GPU type.
#
# Supersedes gen_seed_extension.sh, which only added seeds 3,4,5 and left the
# GPU unpinned.
#
# Two reasons for the rewrite.
#
# 1. GPU pinning. AVAIL_FEATURES on this cluster does not carry the GPU model
#    (most gpu nodes report "(null)"), so --constraint cannot select it. The
#    model lives in GRES, so the selector is --gres=gpu:<type>:1. Without it
#    the scheduler mixes v100-pcie, v100-sxm2, t4, a100 and h200 across a
#    single grid. Because each condition is now its own job, an unpinned grid
#    can put "scratch seed 3" and "indomain seed 3" on different silicon,
#    which breaks the pairing that every CI in the paper depends on, not just
#    the seed-to-seed comparison.
#
# 2. All six seeds, not three. Seeds 42, 1 and 2 of the ASSISTments grid ran
#    on c2204/c2205 (v100-pcie). Adding seeds 3 to 5 elsewhere would confound
#    seed with hardware. Rerunning all six on one pool removes the confound,
#    and comparing the new seed 42/1/2 values against the old ones measures
#    the hardware effect directly instead of assuming it away.
#
# GPU type is overridable:
#   GPUTYPE=h200 bash slurm/generators/gen_n3000_grid.sh
# Valid types on this cluster, from `sinfo -p gpu -o "%20N %30f %20G"`:
#   v100-sxm2 (largest pool, d10xx), v100-pcie (c22xx), a100 (d1026,d1028-9),
#   h200 (d4052-4055), t4 (d1025)
#
# No `source ~/.bashrc` and no `conda activate`: that file does not exist on
# this cluster any more and its absence has already killed jobs at wandb.init.
#
#   bash slurm/generators/gen_n3000_grid.sh
#   bash tools/drip_submit.sh queue_n3000

set -uo pipefail

CODE=/projects/algl/dai.hany/studentbert/code
PY=/projects/algl/dai.hany/envs/sb/bin/python
QDIR="$CODE/queue_n3000"
GPUTYPE="${GPUTYPE:-v100-sxm2}"
SEEDS="${SEEDS:-42 1 2 3 4 5}"
N=3000

cd "$CODE" || exit 1
mkdir -p "$QDIR"
rm -f "$QDIR"/*.sbatch

ASSIST=../checkpoints/edubert_assist2017_pretrain_full_encoder.pt
EDNET=../checkpoints/edubert_ednet_pretrain_full_encoder.pt
JUNYI=../checkpoints/edubert_junyi_pretrain_full_encoder.pt

for CK in "$ASSIST" "$EDNET" "$JUNYI"; do
  if [ ! -f "$CODE/$CK" ]; then
    echo "MISSING ENCODER: $CK"
    exit 1
  fi
done

if ! sinfo -p gpu -h -o "%G" | grep -q "gpu:${GPUTYPE}:"; then
  echo "GPUTYPE '$GPUTYPE' does not appear in any gpu GRES on this cluster."
  echo "available:"
  sinfo -p gpu -h -o "%G" | sort -u
  exit 1
fi

count=0
for TARGET in assist2017 ednet junyi; do
  case "$TARGET" in
    assist2017) TOK=assist; INDOM="$ASSIST"; C1=fromednet;  K1="$EDNET";  C2=fromjunyi; K2="$JUNYI" ;;
    ednet)      TOK=ednet;  INDOM="$EDNET";  C1=fromassist; K1="$ASSIST"; C2=fromjunyi; K2="$JUNYI" ;;
    junyi)      TOK=junyi;  INDOM="$JUNYI";  C1=fromassist; K1="$ASSIST"; C2=fromednet; K2="$EDNET" ;;
  esac
  for SEED in $SEEDS; do
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
        {
          echo '#!/bin/bash'
          echo '#SBATCH --partition=gpu'
          echo "#SBATCH --gres=gpu:${GPUTYPE}:1"
          echo '#SBATCH --cpus-per-task=8'
          echo '#SBATCH --time=08:00:00'
          echo '#SBATCH --mem=48G'
          echo "#SBATCH --output=${NAME}_%j.log"
          echo "cd $CODE"
          echo "PYTHONPATH=. $PY $SCRIPT $EXTRA --processed_dir ../processed/$TARGET $INIT --n_students $N --seed $SEED --epochs 20 --run_type $NAME --wandb"
        } > "$QDIR/${NAME}.sbatch"
        count=$((count + 1))
      done
    done
  done
done

echo "wrote $count sbatch files to $QDIR"
echo "gpu type : $GPUTYPE"
echo "seeds    : $SEEDS"
echo "submit with: bash tools/drip_submit.sh queue_n3000"
