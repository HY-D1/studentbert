from __future__ import annotations

# MRAP failure review, diagnostic D10: is the Algebra 2006 disagreement between the validation and
# the test learners (failure review D8) larger than learner sampling explains? The saved best
# checkpoints of two candidates' N=3000 fine-tunes (6 seeds each, paired by seed) are re-scored on
# both splits exactly as scripts/finetune_edubert.py scores its test set (causal encoder, the next
# skill's logit, real next steps only). Each checkpoint must reproduce the benchmark's recorded test
# AUC (full-precision W&B value) within --tol, which also catches a checkpoint overwritten by a
# later run of the same name. Then learners are resampled with replacement, the same resample for
# every checkpoint (seeded), and the statistic is the mean over seeds of AUC(b) minus AUC(a). The
# two splits hold disjoint learners and are resampled independently, so their bootstrap statistics
# also give an interval for the test-minus-validation gap. Gold-side: this reads the test split,
# like the fine-tune itself, and nothing here feeds a transferability estimate.
#
#   PYTHONPATH=. python scripts/split_bootstrap_kt.py --processed_dir ../processed/algebra2006 \
#       --executions benchmark_final/executions.tsv --ckpt_dir ../checkpoints \
#       --a_run edubert_algebra2006_tgb_algebra2006_indomain_n3000_seed{s} \
#       --b_run edubert_algebra2006_tgb_algebra2006_fromjunyi_n3000_seed{s} \
#       --seeds 1 2 3 4 5 42 --out benchmark_final/split_bootstrap_algebra2006

import argparse
import csv
import importlib.util
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def q(x, places: str = "0.0001") -> str:
    d = Decimal(repr(float(x))).quantize(Decimal(places), rounding=ROUND_HALF_UP)
    return f"{d:+}"


def auc(y: np.ndarray, p: np.ndarray) -> float:
    from src.eval.metrics import auc as project_auc

    return project_auc(y, p)


def learner_index(owner: np.ndarray, n_learners: int) -> list[np.ndarray]:
    """Positions of each learner, so a learner resample maps to a position index."""
    order = np.argsort(owner, kind="stable")
    bounds = np.searchsorted(owner[order], np.arange(n_learners + 1))
    return [order[bounds[i]:bounds[i + 1]] for i in range(n_learners)]


def paired_stat(y, preds_a: list, preds_b: list, idx: np.ndarray) -> float:
    return float(np.mean([auc(y[idx], pb[idx]) - auc(y[idx], pa[idx])
                          for pa, pb in zip(preds_a, preds_b)]))


def bootstrap(y, owner, preds_a, preds_b, n_learners: int, boots: int, rng) -> np.ndarray:
    pos = learner_index(owner, n_learners)
    out = np.empty(boots)
    for b in range(boots):
        pick = rng.integers(0, n_learners, size=n_learners)
        idx = np.concatenate([pos[i] for i in pick])
        out[b] = paired_stat(y, preds_a, preds_b, idx)
    return out


def recorded_test(executions: str, runs: list[str]) -> dict:
    want, got = set(runs), {}
    with open(executions, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if (r["run_id"] in want and r["metric"] == "test_auc"
                    and r["valid_for_primary_analysis"] == "yes"):
                if r["run_id"] in got:
                    sys.exit(f"ABORT: two primary test_auc rows for {r['run_id']}")
                got[r["run_id"]] = float(r["value"])
    missing = want - set(got)
    if missing:
        sys.exit(f"ABORT: no primary test_auc row for {sorted(missing)}")
    return got


def predict(model, ds, device: str, batch_size: int = 64):
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import collate_fn

    ys, ps, owners = [], [], []
    base = 0
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn):
            skill = batch["skill"].to(device)
            correct = batch["correct"].to(device)
            time_bin = batch["time_bin"].to(device)
            mask = batch["mask"].to(device)
            nv = mask[:, 1:] & mask[:, :-1]
            logits = model(skill, correct, time_bin, key_padding_mask=~mask)
            step = model.gather_next_step(logits[:, :-1], skill[:, 1:])
            sel = nv.cpu().numpy().astype(bool)
            ys.append(correct[:, 1:].float().cpu().numpy()[sel])
            ps.append(torch.sigmoid(step).cpu().numpy()[sel])
            owners.append((np.arange(sel.shape[0])[:, None] + base).repeat(sel.shape[1], 1)[sel])
            base += sel.shape[0]
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(owners), base


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--executions", required=True)
    ap.add_argument("--ckpt_dir", default="../checkpoints")
    ap.add_argument("--a_run", required=True, help="run name with {s} for the seed (reference)")
    ap.add_argument("--b_run", required=True, help="run name with {s} for the seed (compared)")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--boots", type=int, default=2000)
    ap.add_argument("--boot_seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=5e-5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True, help="prefix for _summary.tsv and _report.md")
    a = ap.parse_args(argv)

    import torch

    from src.data.dataset import InteractionDataset

    spec = importlib.util.spec_from_file_location("finetune_edubert",
                                                  REPO / "scripts" / "finetune_edubert.py")
    ft = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ft)
    runs_a = [a.a_run.format(s=s) for s in a.seeds]
    runs_b = [a.b_run.format(s=s) for s in a.seeds]
    rec = recorded_test(a.executions, runs_a + runs_b)
    splits = {sp: InteractionDataset(a.processed_dir, sp, 512) for sp in ("val", "test")}
    preds = {sp: {} for sp in splits}
    ys, owners, n_learn = {}, {}, {}
    checks = []
    for run in runs_a + runs_b:
        ck = torch.load(Path(a.ckpt_dir) / f"{run}_best.pt", map_location=a.device,
                        weights_only=False)
        cfg = ck.get("config") or {}
        model = ft.EduBERTForKT(num_skills=ck["num_skills"], d_model=cfg.get("d_model", 256),
                                n_layers=cfg.get("n_layers", 6), dropout=cfg.get("dropout", 0.1),
                                max_len=cfg.get("max_seq_len", 512)).to(a.device)
        model.load_state_dict(ck["model_state"])
        for sp, ds in splits.items():
            y, p, own, n = predict(model, ds, a.device)
            if sp in ys and not (np.array_equal(ys[sp], y) and np.array_equal(owners[sp], own)):
                sys.exit(f"ABORT: {run} sees different {sp} positions from the first checkpoint")
            ys[sp], owners[sp], n_learn[sp] = y, own, n
            preds[sp][run] = p
        test_auc = auc(ys["test"], preds["test"][run])
        diff = abs(test_auc - rec[run])
        checks.append((run, test_auc, rec[run], diff))
        print(f"CHECK {run} test AUC recomputed {test_auc:.8f} recorded {rec[run]:.8f} "
              f"diff {diff:.2e} best epoch {ck.get('epoch')}", flush=True)
        if diff > a.tol:
            sys.exit(f"ABORT: {run} checkpoint does not reproduce the recorded test AUC "
                     f"(diff {diff:.2e} > {a.tol:g})")
    rng = np.random.default_rng(a.boot_seed)
    rows, dist, obs = [], {}, {}
    for sp in ("val", "test"):
        pa = [preds[sp][r] for r in runs_a]
        pb = [preds[sp][r] for r in runs_b]
        per_seed = [auc(ys[sp], b) - auc(ys[sp], a_) for a_, b in zip(pa, pb)]
        obs[sp] = float(np.mean(per_seed))
        dist[sp] = bootstrap(ys[sp], owners[sp], pa, pb, n_learn[sp], a.boots, rng)
        lo, hi = np.percentile(dist[sp], [2.5, 97.5])
        rows.append({"split": sp, "learners": n_learn[sp], "positions": int(ys[sp].size),
                     "b_minus_a": q(obs[sp]), "ci_low": q(lo), "ci_high": q(hi),
                     "p_boot_gt0": f"{float(np.mean(dist[sp] > 0)):.4f}",
                     "seeds_positive": f"{sum(d > 0 for d in per_seed)}/{len(per_seed)}",
                     "per_seed": " ".join(q(d) for d in per_seed)})
    gap = dist["test"] - dist["val"]
    obs_gap = obs["test"] - obs["val"]
    glo, ghi = np.percentile(gap, [2.5, 97.5])
    rows.append({"split": "test minus val", "learners": "", "positions": "",
                 "b_minus_a": q(obs_gap), "ci_low": q(glo), "ci_high": q(ghi),
                 "p_boot_gt0": f"{float(np.mean(gap > 0)):.4f}", "seeds_positive": "",
                 "per_seed": ""})
    with open(f"{a.out}_summary.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    L = ["# Learner-level bootstrap of the validation and test splits", "",
         f"b = {a.b_run}, a = {a.a_run}, seeds {' '.join(map(str, a.seeds))}; {a.boots} learner "
         f"resamples (seed {a.boot_seed}); every checkpoint reproduces its recorded test AUC "
         f"within "
         f"{max(c[3] for c in checks):.2e}.", "",
         "| split | learners | positions | b minus a | 95% interval | P(>0) | seeds positive |",
         "|---|---|---|---|---|---|---|"]
    L += [f"| {r['split']} | {r['learners']} | {r['positions']} | {r['b_minus_a']} | "
          f"[{r['ci_low']}, {r['ci_high']}] | {r['p_boot_gt0']} | {r['seeds_positive']} |"
          for r in rows]
    Path(f"{a.out}_report.md").write_text("\n".join(L) + "\n")
    Path(f"{a.out}_checks.json").write_text(json.dumps(
        [{"run": c[0], "recomputed": c[1], "recorded": c[2], "abs_diff": c[3]} for c in checks],
        indent=1))
    for r in rows:
        print(" ".join(f"{k}={v}" for k, v in r.items() if k != "per_seed"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
