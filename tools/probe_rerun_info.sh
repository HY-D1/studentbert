#!/usr/bin/env bash
# =============================================================================
# probe_rerun_info.sh
#
# READ ONLY. Dumps the exact invocations behind the two job sets that are still
# parked, so the sbatch generators can be written from evidence instead of
# guesswork:
#   (a) in-domain probe extension to all 7 datasets  (~21 short jobs)
#   (b) macro_top1 for the assist2017 next-skill sweep (~24 short jobs)
# Also dumps the two vocab_stats.md files needed to fill the empty README rows.
#
#   bash tools/probe_rerun_info.sh            # from the code dir
# Output: probe_rerun_info.txt   <- upload this back
# =============================================================================
set -uo pipefail
CODE="${1:-/projects/algl/dai.hany/studentbert/code}"
cd "$CODE" 2>/dev/null || { echo "ERROR: cannot cd to $CODE"; exit 1; }
OUT="$CODE/probe_rerun_info.txt"
exec > >(tee "$OUT") 2>&1

echo "host: $(hostname)   code: $CODE   date: $(date)"

echo
echo "=============================================================="
echo " A. WHICH ENCODER DID EACH probe7 RUN ACTUALLY LOAD?"
echo "    decides whether the 7-dataset probe column is EdNet-source"
echo "    or in-domain, and therefore what the extension must run"
echo "=============================================================="
for f in w8_probe7_*_full_s1_*.log; do
  [ -e "$f" ] || continue
  printf "%-44s %s\n" "$(basename "$f")" \
    "$(grep -m1 -oE "loaded [0-9]+/[0-9]+ encoder tensors from .*" "$f" 2>/dev/null)"
done

echo
echo "--- the sbatch that launched them (first match) ---"
for p in slurm/w8_probe7*.sbatch w8_probe7*.sbatch slurm/generators/gen_probe7*.sh; do
  [ -e "$p" ] || continue
  echo "### $p"
  cat "$p"
  break
done

echo
echo "=============================================================="
echo " B. THE W5 NEXT-SKILL INVOCATION (for the macro_top1 rerun)"
echo "=============================================================="
echo "--- a full command line as recorded in a log ---"
grep -h -m3 -oE "downstream_edubert.py .*--task next_skill.*" *.log 2>/dev/null | head -3
echo
echo "--- run= banner lines, assist2017 next-skill, N=25 and N=1000 ---"
grep -h -oE "run=edubert_assist2017_ns[A-Za-z0-9_.-]*" *.log 2>/dev/null \
  | grep -E "_n25_|_n1000_" | sort -u | head -12
echo
echo "--- the sbatch that launched the sweep (first match) ---"
for p in slurm/w5_ns*.sbatch slurm/w5_nextskill*.sbatch slurm/generators/gen_ns*.sh \
         slurm/generators/gen_nextskill*.sh; do
  [ -e "$p" ] || continue
  echo "### $p"
  cat "$p"
  break
done
echo
echo "--- does any assist2017 next-skill log already carry macro-top1? ---"
grep -l "macro-top1" *.log 2>/dev/null | head -5
echo "(empty above confirms the metric was never logged for this sweep)"

echo
echo "=============================================================="
echo " C. vocab_stats.md FOR THE TWO EMPTY README ROWS"
echo "=============================================================="
for d in algebra2005 bridge2006; do
  echo "### ../processed/$d/vocab_stats.md"
  if [ -f "../processed/$d/vocab_stats.md" ]; then
    head -25 "../processed/$d/vocab_stats.md"
  else
    echo "NOT FOUND; try: ls ../processed/"
  fi
  echo
done
ls ../processed/ 2>/dev/null

echo
echo "=============================================================="
echo " D. IS THE PER-SEED next-skill CSV PRESENT?"
echo "    needed by analysis/paired_bootstrap_lak.py"
echo "=============================================================="
ls -la nextskill_results_long.csv nextskill_results_agg.csv 2>/dev/null || \
  echo "not present; regenerate with: python analysis/parse_nextskill_full.py --dir ."
echo "--- first 5 rows (schema: dataset,cond,N,seed,metric,value) ---"
head -5 nextskill_results_long.csv 2>/dev/null
echo "--- row count and distinct metrics ---"
wc -l < nextskill_results_long.csv 2>/dev/null
cut -d, -f5 nextskill_results_long.csv 2>/dev/null | sort | uniq -c

echo
echo "DONE. Upload: $OUT"
