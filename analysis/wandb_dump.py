"""Dump W&B runs whose names match the next-skill sweep to a CSV for
comparison against nextskill_results_long.csv and RESULTS.md section 3.

Run on a login node (needs internet; W&B key is in ~/.netrc):
  python analysis/wandb_dump.py --entity dhy666666o --project StudentBERT
If it reports 0 runs, retry with --entity dhy666666o-n.
"""
import argparse
import csv
import json

import wandb


def jsonable(d):
    out = {}
    for k, val in dict(d).items():
        try:
            json.dumps(val)
            out[k] = val
        except (TypeError, ValueError):
            out[k] = str(val)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="dhy666666o")
    ap.add_argument("--project", default="StudentBERT")
    ap.add_argument("--contains", default="nextskill,nsauc,mt1")
    ap.add_argument("--out", default="wandb_runs_dump.csv")
    a = ap.parse_args()

    tokens = [t.strip() for t in a.contains.split(",") if t.strip()]
    api = wandb.Api()
    try:
        runs = list(api.runs(f"{a.entity}/{a.project}"))
    except Exception as e:
        print(f"could not list runs for {a.entity}/{a.project}: {e}")
        print("check the entity/project spelling in the W&B UI URL and retry")
        return
    print(f"{len(runs)} total runs in {a.entity}/{a.project}", flush=True)

    kept = [r for r in runs if any(t in r.name for t in tokens)]
    print(f"{len(kept)} runs match tokens {tokens}", flush=True)

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "id", "state", "created", "summary_json",
                    "config_json"])
        for r in kept:
            w.writerow([r.name, r.id, r.state, str(r.created_at),
                        json.dumps(jsonable(r.summary._json_dict)),
                        json.dumps(jsonable(r.config))])
    print(f"wrote {a.out} with {len(kept)} rows")


if __name__ == "__main__":
    main()
