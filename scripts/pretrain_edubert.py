"""Pretrain EduBERT with the masked-interaction (MLM) objective.

Loads a processed dataset, masks ~15% of interactions per sequence (BERT 80/10/10),
and trains the encoder to reconstruct masked skill + correctness. Saves the
pretrained encoder weights for downstream fine-tuning.

Usage:
    PYTHONPATH=. python scripts/pretrain_edubert.py \
        --processed_dir ../processed/assist2017 \
        --epochs 50 --batch_size 64 --lr 1e-3 --warmup_frac 0.1 \
        --run_type pretrain_full --wandb

    # tiny variant (subset by interaction budget):
    PYTHONPATH=. python scripts/pretrain_edubert.py \
        --processed_dir ../processed/assist2017 \
        --max_interactions 5000 --epochs 50 --run_type pretrain_tiny --wandb

Logs masked-prediction loss per epoch to W&B (train/mlm_loss, plus skill/correct
components). Saves checkpoint to ../checkpoints/{run_name}_encoder.pt.

W&B: run name {model}_{dataset}_{run_type}, tags [edubert, pretrain], seed +
dropout + config logged; train/lr per step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data.dataset import InteractionDataset, collate_fn
from src.models.edubert import EduBERT
from src.training.masking import mask_interactions, IGNORE
from src.utils import get_device, set_seed


def infer_num_skills(processed_dir: str) -> int:
    vocab = json.loads((Path(processed_dir) / "skill_vocab.json").read_text())
    return max(vocab.values())


def make_lr_lambda(total_steps: int, warmup_frac: float):
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    return lr_lambda


def subset_by_interactions(ds: InteractionDataset, max_interactions: int, seed: int):
    """Return a Subset whose cumulative interaction count is ~max_interactions.
    Selects students in a fixed shuffled order until the budget is hit. Keeps the
    tiny experiment honest (a fixed, reproducible small slice)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds))
    chosen, total = [], 0
    for i in order:
        row = ds.rows[i]
        n = int(ds.offsets[row + 1] - ds.offsets[row])
        n = min(n, ds.max_seq_len)  # count what the model actually sees
        chosen.append(i)
        total += n
        if total >= max_interactions:
            break
    return Subset(ds, chosen), total, len(chosen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup_frac", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--mask_ratio", type=float, default=0.15)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--max_interactions", type=int, default=None,
                    help="if set, pretrain on a subset of ~this many interactions (tiny exp)")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_type", default="pretrain_full")
    ap.add_argument("--ckpt_dir", default="../checkpoints")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    num_skills = infer_num_skills(args.processed_dir)
    dataset = Path(args.processed_dir).name
    run_name = f"edubert_{dataset}_{args.run_type}"

    train_ds = InteractionDataset(args.processed_dir, "train", args.max_seq_len)
    note = ""
    if args.max_interactions is not None:
        train_ds, total_int, n_stu = subset_by_interactions(
            train_ds, args.max_interactions, args.seed
        )
        note = f"  subset={n_stu} students / ~{total_int} interactions"
    print(f"device={device}  num_skills={num_skills}  run={run_name}  seed={args.seed}{note}")

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn)

    wandb_mod = None
    if args.wandb:
        import wandb
        wandb_mod = wandb
        wandb.init(project="StudentBERT", entity="dhy666666o-n",
                   name=run_name, tags=["edubert", "pretrain"],
                   config={"model": "edubert", "dataset": dataset,
                           "run_type": args.run_type, "epochs": args.epochs,
                           "batch_size": args.batch_size, "lr": args.lr,
                           "warmup_frac": args.warmup_frac, "dropout": args.dropout,
                           "mask_ratio": args.mask_ratio, "max_seq_len": args.max_seq_len,
                           "d_model": args.d_model, "n_layers": args.n_layers,
                           "num_skills": num_skills, "seed": args.seed,
                           "max_interactions": args.max_interactions})

    model = EduBERT(num_skills=num_skills, d_model=args.d_model,
                    n_layers=args.n_layers, dropout=args.dropout,
                    max_len=args.max_seq_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = args.epochs * max(1, len(loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=make_lr_lambda(total_steps, args.warmup_frac))

    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)
    K = num_skills + 1
    gen = torch.Generator(device=device).manual_seed(args.seed)

    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    enc_path = ckpt_dir / f"{run_name}_encoder.pt"
    best_loss = float("inf")

    for ep in range(1, args.epochs + 1):
        model.train()
        tot, tot_s, tot_c, nb = 0.0, 0.0, 0.0, 0
        for batch in loader:
            skill = batch["skill"].to(device)
            correct = batch["correct"].to(device)
            time_bin = batch["time_bin"].to(device)
            pad_mask = batch["mask"].to(device)  # True=real

            s_in, c_in, t_in, s_lab, c_lab = mask_interactions(
                skill, correct, time_bin, pad_mask, num_skills,
                mask_ratio=args.mask_ratio, generator=gen)

            out = model(s_in, c_in, t_in, key_padding_mask=~pad_mask)
            loss_s = ce(out["skill_logits"].reshape(-1, K), s_lab.reshape(-1))
            loss_c = ce(out["correct_logits"].reshape(-1, 2), c_lab.reshape(-1))
            loss = loss_s + loss_c

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            if wandb_mod is not None:
                wandb_mod.log({"train/lr": scheduler.get_last_lr()[0]})

            tot += loss.item(); tot_s += loss_s.item(); tot_c += loss_c.item(); nb += 1

        mlm = tot / max(nb, 1)
        print(f"epoch {ep:3d}  mlm_loss={mlm:.4f}  (skill={tot_s/max(nb,1):.4f} "
              f"correct={tot_c/max(nb,1):.4f})")
        if wandb_mod is not None:
            wandb_mod.log({"epoch": ep, "train/mlm_loss": mlm,
                           "train/mlm_skill": tot_s/max(nb,1),
                           "train/mlm_correct": tot_c/max(nb,1)})

        # save encoder at best (lowest) mlm loss
        if mlm < best_loss:
            best_loss = mlm
            torch.save({"model_state": model.state_dict(), "epoch": ep,
                        "mlm_loss": mlm, "num_skills": num_skills,
                        "config": vars(args)}, enc_path)

    print(f"\n=== EduBERT pretrain ({run_name}) ===")
    print(f"best mlm_loss : {best_loss:.4f}")
    print(f"encoder ckpt  : {enc_path}")
    if wandb_mod is not None:
        wandb_mod.log({"best/mlm_loss": best_loss}); wandb_mod.finish()


if __name__ == "__main__":
    main()
