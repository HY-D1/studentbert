#!/usr/bin/env bash
# Final TG1 benchmark build: fresh job and GPU snapshots, builder, W&B export of any new ids,
# builder again, estimator evaluation over every score file, and one zip to upload. Run it by
# hand, or let tools/drip_multi.sh run it as DONE_CMD once the queues are drained. Stops at the
# first failing step (set -e) and refuses to evaluate if a Task 2 score file is missing or empty.
set -euo pipefail
cd /projects/algl/dai.hany/studentbert/code
PY=/projects/algl/dai.hany/envs/sb/bin/python
OUT=benchmark_final
echo "$(date +%T) snapshots"
sacct -u "$USER" -S 2026-05-01 -X -P --format=JobID,JobName,State,Start,End,Elapsed,NodeList,ExitCode > sacct_final.txt
sinfo -N -h -o "%N %G" | sort -u > nodes_gres_final.txt
build() {
  srun --mem=16G --time=00:30:00 --cpus-per-task=2 bash -c "PYTHONPATH=. $PY analysis/build_transfer_benchmark.py --logdir . --sacct sacct_final.txt --gres nodes_gres_final.txt --results-md RESULTS.md --wandb wandb_by_id.jsonl --encoders encoder_configs.tsv --outdir $OUT"
}
echo "$(date +%T) build, pass 1"
build
echo "$(date +%T) W&B export of new ids"
srun --mem=4G --time=01:00:00 "$PY" analysis/wandb_export_by_id.py --ids "$OUT/wandb_ids_needed.txt" --out wandb_by_id.jsonl
echo "$(date +%T) build, pass 2"
build
TASK2=$(grep -h -o -- "--out [^ ]*" queue_tg1_task2*/done/*.sbatch 2>/dev/null | cut -d' ' -f2 | sort -u | tr '\n' ' ')
if [ -z "$TASK2" ]; then
  echo "no Task 2 score files named in queue_tg1_task2*/done; stopping before evaluation"
  exit 1
fi
# Completeness, not just presence: a non-empty file passed on 2026-09-24 although its job had been
# cancelled after 4 of 8 candidates. Every score file must cover seeds 42, 1 and 2 with the same
# candidates for every estimator and seed; the grid scorers must also cover all 8 candidates.
check_scores() {
  "$PY" - "$@" << 'PYEOF'
import json
import sys
from collections import defaultdict

want, files, bad = int(sys.argv[1]), sys.argv[2:], []
for f in files:
    by = defaultdict(set)
    try:
        for line in open(f):
            r = json.loads(line)
            by[(r["estimator"], r["seed"])].add(r["candidate"])
    except FileNotFoundError:
        bad.append(f"{f}: missing")
        continue
    if not by:
        bad.append(f"{f}: empty")
        continue
    ests, seeds = {e for e, _ in by}, {s for _, s in by}
    before = len(bad)
    if seeds != {42, 1, 2}:
        bad.append(f"{f}: seeds {sorted(seeds)}, expected [1, 2, 42]")
    # Candidate counts are compared per estimator across seeds: diagnostic variants such as
    # *_skillrand legitimately score only the in-domain candidate.
    for e in sorted(ests):
        ncand = max(len(by.get((e, s), ())) for s in seeds)
        if want and ncand != want:
            bad.append(f"{f}: {e} has {ncand} candidates, expected {want}")
        for s in sorted(seeds):
            if len(by.get((e, s), ())) != ncand:
                bad.append(f"{f}: {e} seed {s} has {len(by.get((e, s), ()))} of {ncand} candidates")
    state = "complete" if len(bad) == before else "INCOMPLETE"
    print(f"  {state}: {f} ({len(ests)} estimators x {len(seeds)} seeds)")
if bad:
    print("INCOMPLETE score files:")
    for b in bad:
        print("  " + b)
    sys.exit(1)
PYEOF
}
GRID="$(ls tg1_logme_kt_7x7_*.jsonl | tr '\n' ' ')"
DIAG="$(ls tg1_logmediag_kt_*.jsonl | tr '\n' ' ')"
echo "$(date +%T) checking score files"
if ! check_scores 8 $GRID $TASK2 || ! check_scores 0 $DIAG; then
  echo "stopping before evaluation: rerun the jobs behind the incomplete files first"
  exit 1
fi
SCORES="$GRID$DIAG$TASK2"
echo "$(date +%T) evaluation over: $SCORES"
srun --mem=8G --time=00:30:00 --cpus-per-task=2 bash -c "PYTHONPATH=. $PY analysis/evaluate_estimators.py --executions $OUT/executions.tsv --scores $SCORES --out-prefix $OUT/estimators"
cp tg1_logme_kt_*.jsonl tg1_logmediag_kt_*.jsonl $TASK2 "$OUT/"
awk -F'\t' 'NR>1 && $6!="primary" && $1!="S11"' "$OUT/missing_cells.tsv" > "$OUT/missing_not_primary.tsv"
echo "$(date +%T) design cells not primary (ladder excluded): $(wc -l < "$OUT/missing_not_primary.tsv")"
"$PY" -m zipfile -c "$OUT.zip" "$OUT"
ls -la "$OUT.zip"
echo "$(date +%T) final build done; upload $OUT.zip"
