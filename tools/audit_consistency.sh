#!/bin/bash
# 10-WEEK CONSISTENCY AUDIT for StudentBERT. Read-only. Run on cluster:
#   cd /projects/algl/dai.hany/studentbert/code && bash audit_consistency.sh
# Paste the whole output back for review.
cd /projects/algl/dai.hany/studentbert/code
echo "############################################################"
echo "# StudentBERT CONSISTENCY AUDIT  $(date)"
echo "############################################################"

echo ""
echo "===== D. DATA: all datasets same npz schema + splits ====="
for d in ../processed/*/; do
  ds=$(basename "$d")
  [ -f "$d/sequences.npz" ] || { echo "  $ds: NO sequences.npz"; continue; }
  python -c "
import numpy as np, json, os
z=np.load('$d/sequences.npz')
keys=sorted(z.files)
off=z['offsets']; nstu=len(off)-1
try:
    v=json.load(open('$d/skill_vocab.json')); nsk=max(int(x) for x in v.values())
except: nsk='NO_VOCAB'
try:
    s=json.load(open('$d/splits.json')); sp=f\"{len(s['train'])}/{len(s['val'])}/{len(s['test'])}\"
except: sp='NO_SPLITS'
hasstats=os.path.exists('$d/vocab_stats.md')
lens=np.diff(off); import numpy as _n
print(f'  $ds: keys={keys} students={nstu} skills={nsk} split={sp} stats_md={hasstats} med_len={float(_n.median(lens)):.0f} base={float(z[\"correct\"].mean()):.4f}')
" 2>&1 | head -3
done

echo ""
echo "===== C. CODE: all scripts parse (from __future__ bug check) ====="
bad=0
for f in $(ls scripts/*.py analysis/*.py tools/patches/*.py 2>/dev/null); do
  [ -f "$f" ] || continue
  if python -c "import ast; ast.parse(open('$f').read())" 2>/dev/null; then :; else echo "  BROKEN: $f"; bad=1; fi
done
[ $bad -eq 0 ] && echo "  all scripts parse OK"

echo ""
echo "===== C. CODE: git status - uncommitted / untracked ====="
git status --short 2>/dev/null | head -40
echo "  --- last 5 commits ---"
git log --oneline -5 2>/dev/null

echo ""
echo "===== C. CODE: key scripts present in repo? ====="
for f in scripts/preprocess_algebra2005.py scripts/preprocess_assist2009.py \
         scripts/finetune_edubert.py scripts/pretrain_edubert.py \
         scripts/train_baseline.py scripts/downstream_edubert.py \
         scripts/probe_edubert_v2.py scripts/measure_dataset_properties.py \
         scripts/characterize_datasets.py; do
  if [ -f "$f" ]; then tracked=$(git ls-files "$f" | wc -l); echo "  $f: exists, tracked=$tracked"; else echo "  $f: MISSING"; fi
done

echo ""
echo "===== E. CHECKPOINTS: encoders preserved, count of _best ====="
echo "  pretrain encoders (MUST keep):"
ls -1 ../checkpoints/*pretrain*encoder*.pt 2>/dev/null | sed 's#.*/#    #'
echo "  fine-tuned _best.pt (safe to delete, count): $(ls ../checkpoints/*_best.pt 2>/dev/null | wc -l)"
echo "  total checkpoint disk: $(du -sh ../checkpoints 2>/dev/null | awk '{print $1}')"

echo ""
echo "===== A. RESULTS: objective ablation numbers from logs (all 7 datasets) ====="
# collect the objective ablation means per dataset from whatever logs exist
declare -A LOGPFX=( [assist2017]="w7_objabl" [junyi]="w7_objabl2" [ednet]="w8_regime" \
  [algebra2005]="w8_algabl" [bridge2006]="w8_bridgeabl" [assist2009]="w8_a09abl" [algebra2006]="w8_alg06abl" )
for ds in assist2017 junyi ednet algebra2005 bridge2006 assist2009 algebra2006; do
  pfx=${LOGPFX[$ds]}
  echo "  --- $ds (logs: ${pfx}_*) ---"
  for OBJ in full skill_only correct_only; do
    n=$(ls ${pfx}_${OBJ}_*.log 2>/dev/null | wc -l)
    vals=$(for f in ${pfx}_${OBJ}_*.log; do grep -h -A3 "=== EduBERT-KT" "$f" 2>/dev/null | grep "test AUC" | head -1 | awk '{print $NF}'; done 2>/dev/null | tr '\n' ' ')
    echo "    $OBJ: nlogs=$n vals=[$vals]"
  done
done

echo ""
echo "===== E. DUPLICATE LOGS (same condition, >1 jobid) ====="
for pfx in w7_objabl w7_objabl2 w8_regime w8_algabl w8_bridgeabl w8_a09abl w8_alg06abl w8_trunc w8_scratch w8_base; do
  dups=$(ls ${pfx}_*.log 2>/dev/null | sed -E 's/_[0-9]+\.log$//' | sort | uniq -d | wc -l)
  [ "$dups" -gt 0 ] && echo "  $pfx: $dups conditions with duplicate logs"
done
echo "  (duplicates are harmless - collect uses head -1 - but listed for cleanup)"

echo ""
echo "===== B. RUN AUDIT: any CPU-fallback / crash / NaN / leakage in logs ====="
echo "  CPU fallback (should be empty):"
grep -L "device=cuda" w8_*.log w7_*.log 2>/dev/null | head -5
echo "  crashes (Traceback/Killed/OOM, FutureWarning benign):"
grep -lE "Traceback|Killed|OutOfMemory" w8_*.log w7_*.log 2>/dev/null | head -5
echo "  NaN / AUC=1.0 leakage signature:"
grep -liE "nan|auc.*: 1\.0000" w8_*.log w7_*.log 2>/dev/null | head -5

echo ""
echo "############################################################"
echo "# END AUDIT"
echo "############################################################"
