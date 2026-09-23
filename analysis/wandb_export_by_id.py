#!/usr/bin/env python3
"""Fetch W&B runs by id for the transfer benchmark: full-precision summaries plus config.

By id, never by name: 1,576 runs share 1,332 names, so a name lookup can return the wrong
execution. The ids come from the logs themselves (build_transfer_benchmark.py writes
wandb_ids_needed.txt). This never lists the project, which is what OOM-kills the login node, and
it still belongs under srun.

Streams one JSON line per id. On a rerun, ids with a successful record are skipped and failed ids
are retried, so an interrupted export resumes. Metadata (git commit, GPU, host, argv) is
best-effort: if the installed wandb has no Run.metadata the record says so, and sacct still
supplies the GPU.

  srun --mem=4G --time=01:00:00 /projects/algl/dai.hany/envs/sb/bin/python \
      analysis/wandb_export_by_id.py --ids benchmark_20260922/wandb_ids_needed.txt \
      --out wandb_by_id.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

META_KEYS = ("git", "gpu", "gpu_count", "host", "cuda", "python", "startedAt", "program", "args",
             "executable")


def jsonable(d) -> dict:
    out = {}
    for k, v in dict(d).items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="one W&B run id per line (first column)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--entity", default="dhy666666o-n")
    ap.add_argument("--project", default="StudentBERT")
    ap.add_argument("--sleep", type=float, default=0.2)
    a = ap.parse_args()

    import wandb  # imported late so --help works without wandb

    ids = [ln.split()[0] for ln in Path(a.ids).read_text().splitlines()
           if ln.strip() and not ln.startswith("#")]
    done = set()
    out = Path(a.out)
    if out.exists():
        for ln in out.read_text().splitlines():
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            if "error" not in rec:
                done.add(rec["id"])
    todo = [i for i in dict.fromkeys(ids) if i not in done]
    print(f"{len(ids)} ids, {len(done)} already exported, {len(todo)} to fetch", flush=True)

    try:
        api = wandb.Api(timeout=60)
    except TypeError:
        api = wandb.Api()
    n_ok = n_err = 0
    with out.open("a") as fh:
        for k, rid in enumerate(todo, 1):
            rec: dict = {"id": rid}
            try:
                run = api.run(f"{a.entity}/{a.project}/{rid}")
                rec.update(name=run.name, state=run.state, created_at=str(run.created_at),
                           summary=jsonable(run.summary._json_dict), config=jsonable(run.config))
                try:
                    meta = run.metadata or {}
                    rec["metadata"] = jsonable({k2: meta[k2] for k2 in META_KEYS if k2 in meta})
                except Exception as ex:  # optional field; its absence must not lose the run
                    rec["metadata_error"] = repr(ex)[:200]
                n_ok += 1
            except Exception as ex:
                rec["error"] = repr(ex)[:300]
                n_err += 1
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if k % 50 == 0:
                print(f"  {k}/{len(todo)}  ok={n_ok}  errors={n_err}", flush=True)
            time.sleep(a.sleep)
    print(f"done: fetched {n_ok}, errors {n_err}, output {out}")
    if n_err:
        sys.exit(f"{n_err} ids failed; rerun the same command to retry only those")


if __name__ == "__main__":
    main()
