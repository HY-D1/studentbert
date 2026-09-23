#!/usr/bin/env python3
"""Score candidate encoders on one target with the leakage-safe KT LogME (plain and per-skill).

Supersedes scripts/compute_logme.py for any new number. That script is left untouched because
RESULTS.md 6.1 cites it; it fails the Section G audit (bidirectional features paired with
correct[t+1], a double-counted prior term, in-domain encoders scored without their skill table).

Writes one JSON line per (estimator, candidate, seed) and appends, so an interrupted run keeps
what it finished. Candidates are checkpoint paths or the word scratch. Every candidate at a seed
must share one sample fingerprint; the script stops if they do not.

  PYTHONPATH=. python scripts/score_transferability.py --target_dir ../processed/assist2017 \
      --candidates scratch ../checkpoints/edubert_ednet_pretrain_full_encoder.pt \
      --n_students 3000 --seeds 42 1 2 --out logme_kt_assist2017.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.estimators.kt_logme import ranks_by_seed, score_kt_logme


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_dir", required=True)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--n_students", type=int, default=None,
                    help="fine-tune budget to mirror (same learner draw); omit for the full split")
    ap.add_argument("--max_positions", type=int, default=50000)
    ap.add_argument("--min_group", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    for c in a.candidates:
        if c != "scratch" and not Path(c).is_file():
            raise SystemExit(f"missing candidate checkpoint: {c}")

    results = []
    with open(a.out, "a") as fh:
        for seed in a.seeds:
            prints = set()
            for c in a.candidates:
                rs = score_kt_logme(None if c == "scratch" else c, a.target_dir, seed=seed,
                                    n_students=a.n_students, max_positions=a.max_positions,
                                    min_group=a.min_group, device=a.device)
                for r in rs:
                    fh.write(json.dumps(r.to_json()) + "\n")
                    fh.flush()
                    prints.add(r.metadata["sample_fingerprint"])
                    print(f"LOGME_KT_RESULT est={r.estimator} target={r.target} seed={seed} "
                          f"cand={r.candidate} score={r.score:.8f} "
                          f"positions={r.n_target_examples} fp={r.metadata['sample_fingerprint']} "
                          f"loaded={r.metadata['load']['loaded']}/{r.metadata['load']['total']} "
                          f"conv={r.uncertainty_signals['converged']}", flush=True)
                results.extend(rs)
            if len(prints) != 1:
                raise SystemExit(f"seed {seed}: candidates saw different samples {sorted(prints)}")

    for key, order in ranks_by_seed(results).items():
        print("RANK", *key, "->", " > ".join(order))


if __name__ == "__main__":
    main()
