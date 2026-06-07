"""Train a KT baseline (DKT or SAINT+) on a processed dataset; report test AUC.

Usage:
    python scripts/train_baseline.py \
        --model dkt \
        --processed_dir data/full/assist2017 \
        --epochs 10 --batch_size 64

    python scripts/train_baseline.py \
        --model saint --processed_dir data/full/assist2017 --epochs 10

Reports val AUC each epoch and final test AUC + ECE. This is a WORKING baseline
(standard config, lightly trained), not a tuned result.

Next-step framing: at position t we predict correctness of position t+1 using
the queried skill_{t+1}. Loss/metrics computed only on real (non-PAD) next steps.
"""

from __future__ import annotations

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
from src.utils import get_device, set_seed


def infer_num_skills(processed_dir: str) -> int:
    vocab = json.loads((Path(processed_dir) / "skill_vocab.json").read_text())
    return max(vocab.values())  # indices are 1..K, 0 = PAD


def run_epoch(model, loader, device, model_name, optimizer=None):
    """One pass. If optimizer given -> train, else eval. Returns (loss, y_true, y_prob)."""
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
        # valid positions are t where both t and t+1 are real
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

        total_loss += loss.item() * int(next_valid.sum())
        n += int(next_valid.sum())
        sel = next_valid.detach().cpu().numpy().astype(bool)
        all_true.append(target.detach().cpu().numpy()[sel])
        all_prob.append(torch.sigmoid(step_logits).detach().cpu().numpy()[sel])

    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_prob)
    return total_loss / max(n, 1), y_true, y_prob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["dkt", "saint"], required=True)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    num_skills = infer_num_skills(args.processed_dir)
    print(f"device={device}  num_skills={num_skills}")
    if args.wandb:
        import wandb
        wandb.init(project="StudentBERT", entity="dhy666666o-n",
                   name=f"{args.model}_{Path(args.processed_dir).name}",
                   config={"model": args.model, "dataset": Path(args.processed_dir).name,
                           "epochs": args.epochs, "batch_size": args.batch_size,
                           "lr": args.lr, "max_seq_len": args.max_seq_len,
                           "num_skills": num_skills})

    def make_loader(split, shuffle):
        ds = InteractionDataset(args.processed_dir, split, args.max_seq_len)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_fn)

    train_loader = make_loader("train", True)
    val_loader = make_loader("val", False)
    test_loader = make_loader("test", False)

    if args.model == "dkt":
        model = DKT(num_skills=num_skills, hidden_size=128).to(device)
    else:
        model = SAINTPlus(num_skills=num_skills, d_model=256, n_heads=8,
                          n_layers=2, max_len=args.max_seq_len).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        tr_loss, *_ = run_epoch(model, train_loader, device, args.model, opt)
        _, yt, yp = run_epoch(model, val_loader, device, args.model)
        val_auc = auc(yt, yp)
        print(f"epoch {ep:2d}  train_loss={tr_loss:.4f}  val_AUC={val_auc:.4f}")
        if args.wandb:
            import wandb; wandb.log({"epoch": ep, "train/loss": tr_loss, "val/auc": val_auc})

    _, yt, yp = run_epoch(model, test_loader, device, args.model)
    print(f"\n=== {args.model.upper()} on {Path(args.processed_dir).name} ===")
    print(f"test AUC : {auc(yt, yp):.4f}")
    test_auc, test_ece = auc(yt, yp), ece(yt, yp)
    print(f"test ECE : {test_ece:.4f}")
    if args.wandb:
        import wandb; wandb.log({"test/auc": test_auc, "test/ece": test_ece}); wandb.finish()


if __name__ == "__main__":
    main()
