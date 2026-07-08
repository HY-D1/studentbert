from __future__ import annotations
import re
import os
"""Train a KT baseline (DKT or SAINT+) on a processed dataset; report test AUC.

Usage:
    python scripts/train_baseline.py \
        --model saint --processed_dir ../processed/assist2017 \
        --epochs 100 --batch_size 64 --lr 1e-3 --warmup_frac 0.1 \
        --dropout 0.1 --run_type baseline --wandb

Reports val AUC each epoch and final test AUC + ECE.

Next-step framing: at position t we predict correctness of position t+1 using
the queried skill_{t+1}. Loss/metrics computed only on real (non-PAD) next steps.

W&B logging (when --wandb):
  - run name: {model}_{dataset}_{run_type}   (e.g. saint_assist2017_baseline)
  - tags: [model]                            (filter by model in dashboard)
  - config: model, dataset, epochs, batch_size, lr, warmup_frac, dropout,
            max_seq_len, num_skills, seed
  - per-step: lr (train/lr) from the scheduler
  - per-epoch: train/loss, val/auc
  - final: test/auc, test/ece
Best checkpoint (by val AUC) saved to ../checkpoints/{run_name}_best.pt.

LR schedule: linear warmup over warmup_frac of total steps, then linear decay
to 0. Logged per optimizer step so decay is verifiable in W&B.
"""


import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import InteractionDataset, collate_fn
from src.eval.metrics import auc, ece
from src.models.dkt import DKT
from src.models.saint_plus import SAINTPlus
from src.models.akt import AKT
from src.utils import get_device, set_seed


def infer_num_skills(processed_dir: str) -> int:
    vocab = json.loads((Path(processed_dir) / "skill_vocab.json").read_text())
    return max(vocab.values())  # indices are 1..K, 0 = PAD


def make_lr_lambda(total_steps: int, warmup_frac: float):
    """Linear warmup then linear decay to 0. Returns a fn(step)->multiplier in [0,1].
    Used with torch.optim.lr_scheduler.LambdaLR (multiplies the base lr)."""
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps                      # 0 -> 1 over warmup
        # linear decay from 1 -> 0 over the remaining steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    return lr_lambda


def run_epoch(model, loader, device, model_name, optimizer=None,
              scheduler=None, wandb_mod=None):
    """One pass. If optimizer given -> train, else eval. Returns (loss, y_true, y_prob).
    When training with a scheduler, steps it per batch and logs train/lr to W&B."""
    train = optimizer is not None
    model.train() if train else model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    total_loss, n = 0.0, 0
    all_true, all_prob = [], []

    for batch in loader:
        skill = batch["skill"].to(device)
        correct = batch["correct"].to(device)
        time_bin = batch["time_bin"].to(device)
        mask = batch["mask"].to(device)  # (B, L) True = real

        # next-step targets: predict step t+1 from history up to t
        next_valid = mask[:, 1:] & mask[:, :-1]  # (B, L-1)
        target = correct[:, 1:].float()          # (B, L-1)

        with torch.set_grad_enabled(train):
            if model_name == "dkt":
                logits_all = model(skill, correct)          # (B, L, K+1)
                next_skill = skill[:, 1:]                    # (B, L-1)
                step_logits = model.gather_next_step(logits_all[:, :-1], next_skill)
            else:  # saint
                key_pad = ~mask                              # (B, L) True=PAD
                logits = model(skill, correct, time_bin, key_padding_mask=key_pad)
                step_logits = logits[:, 1:]                  # predict t+1

            loss_mat = bce(step_logits, target)
            loss = (loss_mat * next_valid).sum() / next_valid.sum().clamp(min=1)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                    if wandb_mod is not None:
                        wandb_mod.log({"train/lr": scheduler.get_last_lr()[0]})

        total_loss += loss.item() * int(next_valid.sum())
        n += int(next_valid.sum())
        sel = next_valid.detach().cpu().numpy().astype(bool)
        all_true.append(target.detach().cpu().numpy()[sel])
        all_prob.append(torch.sigmoid(step_logits).detach().cpu().numpy()[sel])

    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_prob)
    return total_loss / max(n, 1), y_true, y_prob


def _wandb_fields(args, dataset):
    # derive source/target/condition for clean W&B grouping
    src = 'none'
    ck = getattr(args, 'encoder_ckpt', None)
    if ck:
        b = os.path.basename(ck)
        m = re.match(r'edubert_([a-zA-Z0-9]+)_pretrain', b)
        if m: src = m.group(1)
    init = getattr(args, 'init', 'scratch')
    cond = 'scratch' if init == 'scratch' else ('indomain' if src == dataset else src)
    return {'source': src, 'target': dataset, 'condition': cond}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["dkt", "saint", "akt"], required=True)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup_frac", type=float, default=0.1,
                    help="fraction of total steps for linear warmup (0 disables schedule)")
    ap.add_argument("--dropout", type=float, default=None,
                    help="model dropout; defaults to model's own default if unset")
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_students", type=int, default=None,
                    help="subsample TRAIN to first N students (seeded); None=all")
    ap.add_argument("--run_type", default="baseline",
                    help="run-name suffix, e.g. baseline / pretrain / scratch")
    ap.add_argument("--ckpt_dir", default="../checkpoints")
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    num_skills = infer_num_skills(args.processed_dir)
    dataset = Path(args.processed_dir).name
    run_name = f"{args.model}_{dataset}_{args.run_type}"
    print(f"device={device}  num_skills={num_skills}  run={run_name}  seed={args.seed}")

    # resolve dropout (use model default if not supplied, but always log the value)
    default_dropout = 0.2 if args.model == "dkt" else 0.1
    dropout = args.dropout if args.dropout is not None else default_dropout

    wandb_mod = None
    if args.wandb:
        import wandb
        wandb_mod = wandb
        wandb.init(
            project="StudentBERT", entity="dhy666666o-n",
            name=run_name, tags=[args.model],
            config={
                "model": args.model, "dataset": dataset, "run_type": args.run_type,
                "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                "warmup_frac": args.warmup_frac, "dropout": dropout,
                "max_seq_len": args.max_seq_len, "num_skills": num_skills,
                "seed": args.seed,
                **_wandb_fields(args, dataset),
            },
        )

    def make_loader(split, shuffle):
        ds = InteractionDataset(args.processed_dir, split, args.max_seq_len)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_fn)

    train_ds = InteractionDataset(args.processed_dir, "train", args.max_seq_len)
    if args.n_students is not None and args.n_students < len(train_ds.rows):
        import numpy as _np
        _rng = _np.random.default_rng(args.seed)
        _order = _rng.permutation(len(train_ds.rows))[:args.n_students]
        train_ds.rows = [train_ds.rows[i] for i in _order]
        print(f"subsampled train to {len(train_ds.rows)} students (seed {args.seed})")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = make_loader("val", False)
    test_loader = make_loader("test", False)

    if args.model == "dkt":
        model = DKT(num_skills=num_skills, hidden_size=128, dropout=dropout).to(device)
    elif args.model == "akt":
        model = AKT(num_skills=num_skills, d_model=256, n_heads=8,
                    n_blocks=2, d_ff=1024, dropout=dropout, max_len=args.max_seq_len).to(device)
    else:
        model = SAINTPlus(num_skills=num_skills, d_model=256, n_heads=8,
                          n_layers=2, dropout=dropout, max_len=args.max_seq_len).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # linear warmup -> linear decay scheduler (per-step). warmup_frac=0 -> no schedule.
    scheduler = None
    if args.warmup_frac and args.warmup_frac > 0:
        total_steps = args.epochs * max(1, len(train_loader))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=make_lr_lambda(total_steps, args.warmup_frac)
        )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_auc = -1.0
    best_path = ckpt_dir / f"{run_name}_best.pt"

    for ep in range(1, args.epochs + 1):
        tr_loss, *_ = run_epoch(model, train_loader, device, args.model,
                                optimizer=opt, scheduler=scheduler, wandb_mod=wandb_mod)
        _, yt, yp = run_epoch(model, val_loader, device, args.model)
        val_auc = auc(yt, yp)
        print(f"epoch {ep:3d}  train_loss={tr_loss:.4f}  val_AUC={val_auc:.4f}")
        if wandb_mod is not None:
            wandb_mod.log({"epoch": ep, "train/loss": tr_loss, "val/auc": val_auc})

        # save best checkpoint by val AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {"model_state": model.state_dict(), "epoch": ep,
                 "val_auc": val_auc, "config": vars(args),
                 "num_skills": num_skills},
                best_path,
            )

    # final test using the BEST checkpoint (not the last, which may be overfit)
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"\nloaded best checkpoint from epoch {ckpt['epoch']} "
              f"(val AUC {ckpt['val_auc']:.4f}) for test eval")

    _, yt, yp = run_epoch(model, test_loader, device, args.model)
    test_auc, test_ece = auc(yt, yp), ece(yt, yp)
    print(f"\n=== {args.model.upper()} on {dataset} ===")
    print(f"best val AUC : {best_val_auc:.4f}")
    print(f"test AUC     : {test_auc:.4f}")
    print(f"test ECE     : {test_ece:.4f}")
    print(f"best ckpt    : {best_path}")

    if wandb_mod is not None:
        wandb_mod.log({"test/auc": test_auc, "test/ece": test_ece,
                       "best/val_auc": best_val_auc})
        wandb_mod.finish()


if __name__ == "__main__":
    main()
