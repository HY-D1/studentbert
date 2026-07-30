#!/usr/bin/env bash
# =============================================================================
# collect_poster_gaps.sh
#
# Dumps the raw log evidence for the only two poster numbers that are NOT in
# RESULTS.md:
#   (1) KT AUC gain over scratch at N=3000 into the ASSIST2017 target
#       poster R2 bars: EdNet +0.0269, in-domain +0.0232, Junyi +0.0189
#   (2) probe-decodability vs transfer Spearman rho 0.83, and LogME rho 0.50
#
# READ ONLY. No GPU. No sbatch. No files modified. Runs in a few seconds.
#
#   bash collect_poster_gaps.sh                    # default code dir
#   bash collect_poster_gaps.sh /path/to/code      # explicit dir
#   WANDB=1 bash collect_poster_gaps.sh            # also query W&B (optional)
#
# Writes: poster_gaps_evidence.txt   <- upload this back
# =============================================================================
set -uo pipefail

CODE="${1:-/projects/algl/dai.hany/studentbert/code}"
cd "$CODE" 2>/dev/null || { echo "ERROR: cannot cd to $CODE"; exit 1; }
OUT="$CODE/poster_gaps_evidence.txt"
exec > >(tee "$OUT") 2>&1

echo "=============================================================="
echo " 0. ORIENTATION"
echo "=============================================================="
echo "host      : $(hostname)"
echo "date      : $(date)"
echo "code dir  : $CODE"
echo "python    : $(which python 2>/dev/null || echo 'none on PATH')"

# every log that might matter, including archived duplicates
mapfile -t LOGS < <(find "$CODE" "$CODE/logs_archive" "$CODE/../logs" \
                      -maxdepth 1 -name '*.log' 2>/dev/null | sort -u)
echo "log files : ${#LOGS[@]}"
if [ "${#LOGS[@]}" -eq 0 ]; then
  echo "ERROR: no .log files found. Pass the correct code dir as argument 1."
  exit 1
fi

echo
echo "=============================================================="
echo " 1. DISTINCT LOG BLOCK HEADERS   (reveals run-name conventions)"
echo "    digits normalised to N so the unique set stays small"
echo "=============================================================="
grep -h "^===" "${LOGS[@]}" 2>/dev/null \
  | sed 's/[0-9][0-9]*/N/g' | cut -c1-90 | sort -u | head -40

echo
echo "=============================================================="
echo " 2. WHAT A METRIC LINE LOOKS LIKE   (first few AUC lines seen)"
echo "=============================================================="
grep -h -iE "test .*auc|val .*auc" "${LOGS[@]}" 2>/dev/null | head -6

echo
echo "=============================================================="
echo " 3. RUN NAMES CONTAINING 3000       (count = how many log hits)"
echo "=============================================================="
grep -h -oE "edubert[A-Za-z0-9_.-]*" "${LOGS[@]}" 2>/dev/null \
  | grep -E "3000" | sort | uniq -c | sort -rn

echo
echo "=============================================================="
echo " 4. METRIC LINE FOR EACH N=3000 RUN"
echo "    first 'test ... auc' within 10 lines of the run name;"
echo "    head -1 guards against resubmit duplicates"
echo "=============================================================="
mapfile -t R3000 < <(grep -h -oE "edubert[A-Za-z0-9_.-]*" "${LOGS[@]}" 2>/dev/null \
                       | grep -E "3000" | sort -u)
if [ "${#R3000[@]}" -eq 0 ]; then
  echo "(none found by name; see section 5 for a filename-based fallback)"
else
  for r in "${R3000[@]}"; do
    m=$(grep -h -F -A10 "$r" "${LOGS[@]}" 2>/dev/null \
          | grep -m1 -iE "test .*auc")
    printf "%-62s %s\n" "$r" "${m:-NOT FOUND}"
  done
fi

echo
echo "=============================================================="
echo " 5. FALLBACK: FILES MENTIONING 3000, AND THEIR ASSIST2017 LINES"
echo "=============================================================="
mapfile -t F3000 < <(grep -l -E "3000" "${LOGS[@]}" 2>/dev/null | sort -u)
echo "files mentioning 3000: ${#F3000[@]}"
for f in "${F3000[@]:0:12}"; do
  echo "--- $(basename "$f")"
  grep -h -iE "assist2017|n_students|first_n|3000" "$f" 2>/dev/null \
    | grep -v "^$" | head -6
done

echo
echo "=============================================================="
echo " 6. ALL BUDGETS SEEN FOR THE ASSIST2017 KT TARGET"
echo "    (tells us whether 3000 is stored as n3000 / N=3000 / 3000)"
echo "=============================================================="
grep -h -oE "edubert_assist2017[A-Za-z0-9_.-]*" "${LOGS[@]}" 2>/dev/null \
  | sed 's/_seed[0-9]*$//' | sort -u | head -50

echo
echo "=============================================================="
echo " 7. PROBE v2 RAW   (w6_probe2* and w8_probe7* ONLY)"
echo "    w6_probe_*.log holds the DEPRECATED v1 circular probe and is"
echo "    deliberately excluded here"
echo "=============================================================="
mapfile -t PLOGS < <(find "$CODE" "$CODE/logs_archive" -maxdepth 1 \
                       \( -name 'w6_probe2*.log' -o -name 'w8_probe7*.log' \) \
                       2>/dev/null | sort)
echo "probe v2 logs: ${#PLOGS[@]}"
for f in "${PLOGS[@]}"; do
  hdr=$(grep -m1 -h "^=== probe" "$f" 2>/dev/null)
  met=$(grep -h -iE "acc|top-?1" "$f" 2>/dev/null | head -3 | tr '\n' ' | ')
  printf "%-46s %-28s %s\n" "$(basename "$f")" "${hdr:-no header}" "${met:-no metric}"
done

echo
echo "=============================================================="
echo " 8. LOGME EVIDENCE"
echo "=============================================================="
grep -h -iE "logme" "${LOGS[@]}" 2>/dev/null | sort -u | head -40
echo "--- files with logme in the name:"
find "$CODE" -maxdepth 2 -iname '*logme*' 2>/dev/null | head -20
echo "--- compute_logme.py interface (so it can be re-run if nothing was saved):"
if [ -f scripts/compute_logme.py ]; then
  python scripts/compute_logme.py --help 2>&1 | head -25
else
  echo "scripts/compute_logme.py NOT PRESENT"
fi

echo
echo "=============================================================="
echo " 9. ANY SPEARMAN / RHO ALREADY COMPUTED AND WRITTEN DOWN"
echo "=============================================================="
grep -h -iE "spearman|rho ?[=:]|correlat" "${LOGS[@]}" 2>/dev/null | sort -u | head -30
grep -rn -iE "spearman|rho ?[=:]" --include='*.md' --include='*.csv' \
     --include='*.json' --include='*.txt' "$CODE" 2>/dev/null | head -20

echo
echo "=============================================================="
echo "10. ANALYSIS ARTIFACTS PRESENT ON DISK"
echo "=============================================================="
ls -la *.csv *.json 2>/dev/null | head -30
echo "--- checkpoints kept (encoders only, _best.pt were deleted):"
ls -la ../checkpoints/*encoder.pt 2>/dev/null | head -20

# ---------------------------------------------------------------------------
# Optional: pull the same numbers straight from W&B, which is the system of
# record and is what Prof. Hazra sees. Needs `wandb login` already done.
# ---------------------------------------------------------------------------
if [ "${WANDB:-0}" = "1" ]; then
echo
echo "=============================================================="
echo "11. W&B EXPORT"
echo "=============================================================="
cat > /tmp/_wb_poster.py <<'PY'
import re, statistics as st
try:
    import wandb
except ImportError:
    raise SystemExit("wandb not importable in this env")
api = wandb.Api(timeout=40)

paths = ["dhy666666o-n/StudentBERT", "dhy666666o/StudentBERT",
         "dhy666666o-n/studentbert", "dhy666666o/studentbert"]
runs = None
for p in paths:
    try:
        r = list(api.runs(p, per_page=500))
        if r:
            runs, used = r, p
            break
    except Exception as e:
        print(f"  {p}: {type(e).__name__}")
if runs is None:
    raise SystemExit("no W&B project matched; run `wandb login` or fix the entity")
print(f"project: {used}   runs: {len(runs)}")

def auc(s):
    for k in ("test/auc", "test_auc", "test/AUC", "final_test_auc", "best_test_auc"):
        if k in s:
            return s[k]
    for k in s.keys():
        if "auc" in k.lower() and "val" not in k.lower():
            return s[k]
    return None

print("\n--- runs whose name mentions 3000 ---")
rows = []
for r in runs:
    if "3000" in r.name:
        a = auc(r.summary)
        rows.append((r.name, r.state, a))
for n, stt, a in sorted(rows):
    print(f"  {n:58s} {stt:9s} {a}")

print("\n--- assist2017 KT runs, any budget (name, state, test auc) ---")
for r in sorted(runs, key=lambda x: x.name):
    if "assist2017" in r.name and "probe" not in r.name and "dropout" not in r.name:
        print(f"  {r.name:58s} {r.state:9s} {auc(r.summary)}")

print("\n--- probe runs ---")
for r in sorted(runs, key=lambda x: x.name):
    if "probe" in r.name:
        s = {k: v for k, v in r.summary.items() if "acc" in k.lower()}
        print(f"  {r.name:58s} {s}")

print("\n--- any summary key mentioning logme / rho / spearman ---")
for r in runs:
    hits = {k: v for k, v in r.summary.items()
            if re.search(r"logme|rho|spearman", k, re.I)}
    if hits:
        print(f"  {r.name:58s} {hits}")
PY
python /tmp/_wb_poster.py
fi

echo
echo "=============================================================="
echo " DONE. Upload: $OUT"
echo "=============================================================="
