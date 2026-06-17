"""Fine-tune EduBERT for knowledge tracing (next-step correctness).

Two modes:
  --init pretrained --encoder_ckpt <path>  : load pretrained encoder, then fine-tune
  --init scratch                           : random init (the control)

CRITICAL — causal masking during KT fine-tuning:
EduBERT's encoder is bidirectional (BERT-style, for the MLM pretraining objective).
But knowledge tracing predicts step t+1 from history up to t, and the model must
NOT see the token it is predicting (that would leak the label). So at fine-tune
time we pass a CAUSAL attention mask to the encoder, turning it into a left-to-right
model for the KT task. We also feed the TRUE outcomes (no MASK tokens) as input.

KT readout: a fresh linear head maps each position's hidden state to a per-skill
correctness logit; at step t we read the logit for skill_{t+1} as the prediction
for t+1. Loss/metrics on real (non-PAD) next steps only. Mirrors train_baseline.py
so AUC/ECE are directly comparable to DKT/SAINT+.

Usage:
    PYTHONPATH=. python scripts/finetune_edubert.py \
        --processed_dir ../processed/assist2017 --init pretrained \
        --encoder_ckpt ../checkpoints/edubert_assist2017_pretrain_tiny_encoder.pt \
        --epochs 20 --run_type pretrain_tiny --wandb

    PYTHONPATH=. python scripts/finetune_edubert.py \
        --processed_dir ../processed/assist2017 --init scratch \
        --epochs 20 --run_type scratch_tiny --wandb

W&B run name: edubert_{dataset}_{run_type}; tags [edubert, finetune, {init}].
Best checkpoint by val AUC; seed + dropout + config + train/lr logged.
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
from src.eval.metrics import auc, ece
from src.models.edubert import EduBERT
from src.utils import get_device, set_seed


def infer_num_skills(processed_dir: str) -> int:
    vocab = json.loads((Path(processed_dir) / "skill_vocab.json").read_text())
    return max(vocab.values())


def make_lr_lambda(total_steps, warmup_frac):
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    return lr_lambda


def first_n_students(ds: InteractionDataset, n: int, seed: int):
    """Deterministic subset of n students (for the cold-start experiment)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds))[:n]
    return Subset(ds, order.tolist())


class EduBERTForKT(nn.Module):
    """EduBERT encoder + a KT head. Encoder runs CAUSALLY here (no future peeking)."""

    def __init__(self, num_skills, d_model=256, n_layers=6, dropout=0.1, max_len=512):
        super().__init__()
        self.backbone = EduBERT(num_skills=num_skills, d_model=d_model,
                                n_layers=n_layers, dropout=dropout, max_len=max_len)
        self.num_skills = num_skills
        self.kt_head = nn.Linear(d_model, num_skills + 1)  # per-skill correctness logit

    def _causal_mask(self, L, device):
        # True = blocked. Upper triangle (excl. diagonal) so position t sees <= t only.
        return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

    def encode_causal(self, skill, correct, time_bin, key_padding_mask):
        # replicate EduBERT.encode but with a causal attention mask
        bb = self.backbone
        B, L = skill.shape
        pos = torch.arange(L, device=skill.device).unsqueeze(0).expand(B, L)
        x = (bb.skill_emb(skill) + bb.outcome_emb(correct)
             + bb.time_emb(time_bin) + bb.pos_emb(pos))
        x = bb.emb_drop(bb.emb_norm(x))
        causal = self._causal_mask(L, skill.device)
        return bb.encoder(x, mask=causal, src_key_padding_mask=key_padding_mask)

    def forward(self, skill, correct, time_bin, key_padding_mask):
        h = self.encode_causal(skill, correct, time_bin, key_padding_mask)  # (B,L,d)
        return self.kt_head(h)  # (B, L, K+1)

    @staticmethod
    def gather_next_step(logits, next_skill):
        return torch.gather(logits, 2, next_skill.unsqueeze(-1)).squeeze(-1)


def run_epoch(model, loader, device, optimizer=None, scheduler=None, wandb_mod=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    total_loss, n = 0.0, 0
    all_true, all_prob = [], []

    for batch in loader:
        skill = batch["skill"].to(device)
        correct = batch["correct"].to(device)
        time_bin = batch["time_bin"].to(device)
        mask = batch["mask"].to(device)

        next_valid = mask[:, 1:] & mask[:, :-1]
        target = correct[:, 1:].float()

        with torch.set_grad_enabled(train):
            logits = model(skill, correct, time_bin, key_padding_mask=~mask)  # (B,L,K+1)
            next_skill = skill[:, 1:]
            step_logits = model.gather_next_step(logits[:, :-1], next_skill)  # (B,L-1)

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

    return total_loss / max(n, 1), np.concatenate(all_true), np.concatenate(all_prob)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--init", choices=["pretrained", "scratch"], required=True)
    ap.add_argument("--encoder_ckpt", default=None,
                    help="required when --init pretrained")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup_frac", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_students", type=int, default=None,
                    help="if set, fine-tune on only this many students (cold-start exp)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_type", required=True)
    ap.add_argument("--ckpt_dir", default="../checkpoints")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    if args.init == "pretrained" and not args.encoder_ckpt:
        raise SystemExit("--init pretrained requires --encoder_ckpt")

    set_seed(args.seed)
    device = get_device()
    num_skills = infer_num_skills(args.processed_dir)
    dataset = Path(args.processed_dir).name
    run_name = f"edubert_{dataset}_{args.run_type}"

    train_ds = InteractionDataset(args.processed_dir, "train", args.max_seq_len)
    if args.n_students is not None:
        train_ds = first_n_students(train_ds, args.n_students, args.seed)
    val_ds = InteractionDataset(args.processed_dir, "val", args.max_seq_len)
    test_ds = InteractionDataset(args.processed_dir, "test", args.max_seq_len)

    def loader(ds, sh): return DataLoader(ds, batch_size=args.batch_size,
                                          shuffle=sh, collate_fn=collate_fn)
    train_loader, val_loader, test_loader = loader(train_ds, True), loader(val_ds, False), loader(test_ds, False)

    n_train_students = len(train_ds)
    print(f"device={device}  num_skills={num_skills}  run={run_name}  "
          f"init={args.init}  train_students={n_train_students}  seed={args.seed}")

    model = EduBERTForKT(num_skills=num_skills, d_model=args.d_model,
                         n_layers=args.n_layers, dropout=args.dropout,
                         max_len=args.max_seq_len).to(device)

    # load pretrained encoder weights (everything except the KT head).
    # CROSS-DATASET SAFE: keep only tensors whose shape matches the target model.
    # For in-domain transfer (same skill vocab) this loads everything incl. skill_emb.
    # For cross-dataset transfer (disjoint vocab, different K) this automatically
    # SKIPS skill_emb + skill_head (size mismatch) and transfers the shared parts:
    # encoder layers, pos_emb, time_emb, outcome_emb, emb_norm. The skill table is
    # then (re)learned fresh on the target dataset — the correct behavior, since a
    # skill id means different things across datasets.
    if args.init == "pretrained":
        ck = torch.load(args.encoder_ckpt, map_location=device)
        state = ck["model_state"]
        tgt = model.backbone.state_dict()
        transferable = {k: v for k, v in state.items()
                        if k in tgt and v.shape == tgt[k].shape}
        skipped = sorted(k for k in state if k not in transferable)
        missing, unexpected = model.backbone.load_state_dict(transferable, strict=False)
        mlm = ck.get("mlm_loss")
        mlm_s = f"{mlm:.4f}" if isinstance(mlm, (int, float)) else str(mlm)
        print(f"loaded {len(transferable)}/{len(state)} tensors from {args.encoder_ckpt} "
              f"(epoch {ck.get('epoch')}, mlm {mlm_s})")
        if skipped:
            print(f"  skipped {len(skipped)} vocab-specific tensors (cross-dataset): {skipped}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = args.epochs * max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=make_lr_lambda(total_steps, args.warmup_frac))

    wandb_mod = None
    if args.wandb:
        import wandb
        wandb_mod = wandb
        wandb.init(project="StudentBERT", entity="dhy666666o-n",
                   name=run_name, tags=["edubert", "finetune", args.init],
                   config={"model": "edubert", "dataset": dataset,
                           "run_type": args.run_type, "init": args.init,
                           "epochs": args.epochs, "batch_size": args.batch_size,
                           "lr": args.lr, "warmup_frac": args.warmup_frac,
                           "dropout": args.dropout, "max_seq_len": args.max_seq_len,
                           "d_model": args.d_model, "n_layers": args.n_layers,
                           "n_students": n_train_students, "num_skills": num_skills,
                           "seed": args.seed,
                           "encoder_ckpt": args.encoder_ckpt or "none"})

    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{run_name}_best.pt"
    best_val = -1.0

    for ep in range(1, args.epochs + 1):
        tr_loss, *_ = run_epoch(model, train_loader, device, opt, scheduler, wandb_mod)
        _, yt, yp = run_epoch(model, val_loader, device)
        val_auc = auc(yt, yp)
        print(f"epoch {ep:3d}  train_loss={tr_loss:.4f}  val_AUC={val_auc:.4f}")
        if wandb_mod is not None:
            wandb_mod.log({"epoch": ep, "train/loss": tr_loss, "val/auc": val_auc})
        if val_auc > best_val:
            best_val = val_auc
            torch.save({"model_state": model.state_dict(), "epoch": ep,
                        "val_auc": val_auc, "config": vars(args),
                        "num_skills": num_skills}, best_path)

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])

    _, yt, yp = run_epoch(model, test_loader, device)
    test_auc, test_ece = auc(yt, yp), ece(yt, yp)
    print(f"\n=== EduBERT-KT ({run_name}, init={args.init}) ===")
    print(f"train students : {n_train_students}")
    print(f"best val AUC   : {best_val:.4f}")
    print(f"test AUC       : {test_auc:.4f}")
    print(f"test ECE       : {test_ece:.4f}")
    if wandb_mod is not None:
        wandb_mod.log({"test/auc": test_auc, "test/ece": test_ece,
                       "best/val_auc": best_val}); wandb_mod.finish()


if __name__ == "__main__":
    main()
