# Linear probing of a (frozen) EduBERT encoder for skill-relational structure.
# FIX vs v1: v1 was circular. The encoder input includes skill_emb(skill) at each
# position, so a linear probe trivially inverted it (scratch reached ~99.9%),
# measuring nothing about pretraining. v2 removes the circularity by MASKING the
# skill at the probed position (skill set to 0=PAD before encoding), so the probe
# must infer the position's skill from CONTEXT (neighboring interactions, timing,
# correctness). Now a random encoder cannot cheat, and the gap between sources
# reflects learned skill-relational structure. This is the mechanistic evidence
# for what each pretraining source captures.
#
# Method: for each position t, zero out skill[t] (keep correct[t], time[t]),
# encode the full (context-masked) sequence, take hidden[t], train a linear probe
# to predict the TRUE skill[t]. Frozen encoder; only the probe learns.
#
# Usage:
#   python scripts/probe_edubert_v2.py --processed_dir ../processed/assist2017 \
#       --init scratch --seed 42 --run_type probe2_scratch --wandb
#   python scripts/probe_edubert_v2.py --processed_dir ../processed/assist2017 \
#       --init pretrained --encoder_ckpt <enc.pt> --seed 42 --run_type probe2_ednet --wandb
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.edubert import EduBERT


def infer_num_skills(processed_dir):
    vocab = json.loads((Path(processed_dir) / "skill_vocab.json").read_text())
    return max(vocab.values())


class SeqDataset(Dataset):
    def __init__(self, processed_dir, split, max_seq_len=512):
        d = Path(processed_dir)
        data = np.load(d / "sequences.npz")
        self.skill = data["skill"]; self.correct = data["correct"]
        self.time_bin = data["time_bin"]; self.offsets = data["offsets"]
        ids = data["student_ids"]
        splits = json.loads((d / "splits.json").read_text())
        wanted = set(int(s) for s in splits[split])
        id_to_row = {int(s): i for i, s in enumerate(ids)}
        self.rows = [id_to_row[s] for s in wanted if s in id_to_row]
        self.max_seq_len = max_seq_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        s, e = self.offsets[row], self.offsets[row + 1]
        sk = self.skill[s:e].astype(np.int64)
        co = self.correct[s:e].astype(np.int64)
        tb = self.time_bin[s:e].astype(np.int64)
        if len(sk) > self.max_seq_len:
            sk, co, tb = sk[-self.max_seq_len:], co[-self.max_seq_len:], tb[-self.max_seq_len:]
        return {"skill": torch.from_numpy(sk), "correct": torch.from_numpy(co),
                "time_bin": torch.from_numpy(tb), "length": len(sk)}


def collate(batch):
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    L = int(lengths.max()); B = len(batch)
    skill = torch.zeros(B, L, dtype=torch.long)
    correct = torch.zeros(B, L, dtype=torch.long)
    time_bin = torch.zeros(B, L, dtype=torch.long)
    mask = torch.zeros(B, L, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["length"]
        skill[i, :n] = b["skill"]; correct[i, :n] = b["correct"]
        time_bin[i, :n] = b["time_bin"]; mask[i, :n] = True
    return {"skill": skill, "correct": correct, "time_bin": time_bin, "mask": mask}


def load_encoder(backbone, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    own = backbone.state_dict()
    keep = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
    skipped = sorted(k for k in state if k not in keep)
    backbone.load_state_dict(keep, strict=False)
    print(f"loaded {len(keep)}/{len(state)} encoder tensors from {ckpt_path}")
    if skipped:
        print(f"  skipped (vocab-specific): {skipped}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--init", choices=["scratch", "pretrained"], required=True)
    ap.add_argument("--encoder_ckpt", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--run_type", default="probe2")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    if args.init == "pretrained" and not args.encoder_ckpt:
        raise SystemExit("--init pretrained requires --encoder_ckpt")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_skills = infer_num_skills(args.processed_dir)
    dataset = Path(args.processed_dir).name
    run_name = f"edubert_{dataset}_{args.run_type}"
    print(f"device={device}  num_skills={num_skills}  run={run_name}  "
          f"init={args.init}  seed={args.seed}  [MASKED-SKILL probe]")

    backbone = EduBERT(num_skills=num_skills, d_model=args.d_model,
                       max_len=args.max_seq_len).to(device)
    if args.init == "pretrained":
        load_encoder(backbone, args.encoder_ckpt, device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    probe = nn.Linear(args.d_model, num_skills + 1).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss(ignore_index=0)

    wb = None
    if args.wandb:
        try:
            import wandb as wb_mod
            wb_mod.init(project="StudentBERT", name=run_name,
                        config={"task": "probe_masked_skill", "init": args.init,
                                "dataset": dataset, "seed": args.seed,
                                "encoder_ckpt": args.encoder_ckpt or "none"})
            wb = wb_mod
        except Exception as ex:
            print(f"wandb disabled: {ex}")

    def loader(split, sh):
        return DataLoader(SeqDataset(args.processed_dir, split, args.max_seq_len),
                          batch_size=args.batch_size, shuffle=sh, collate_fn=collate)
    tr, va, te = loader("train", True), loader("val", False), loader("test", False)

    def rep_masked(batch):
        # KEY FIX: zero the skill at every position before encoding, so the
        # encoder cannot read the skill it will be asked to predict. correctness
        # and timing are kept, so the encoder must infer skill from context.
        sk = batch["skill"].to(device); co = batch["correct"].to(device)
        tb = batch["time_bin"].to(device); mask = batch["mask"].to(device)
        sk_masked = torch.zeros_like(sk)              # all positions -> PAD skill 0
        with torch.no_grad():
            h = backbone.encode(sk_masked, co, tb, key_padding_mask=~mask)  # (B,L,d)
        return h, sk, mask

    def run(split_loader, train=False):
        probe.train() if train else probe.eval()
        cn, tn, ls, nb = 0, 0, 0.0, 0
        for batch in split_loader:
            h, sk, mask = rep_masked(batch)
            logits = probe(h)
            tgt = sk.clone(); tgt[~mask] = 0
            loss = ce(logits.view(-1, logits.size(-1)), tgt.view(-1))
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            pred = logits.argmax(-1)
            cn += int(((pred == sk) & mask).sum()); tn += int(mask.sum())
            ls += float(loss); nb += 1
        return ls / max(nb, 1), cn / max(tn, 1)

    best_va = -1.0; best_state = None
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = run(tr, train=True)
        _, va_acc = run(va, train=False)
        print(f"epoch {ep:3d}  train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  val_acc={va_acc:.4f}")
        if wb is not None:
            wb.log({"epoch": ep, "train/loss": tr_loss, "train/acc": tr_acc, "val/acc": va_acc})
        if va_acc > best_va:
            best_va = va_acc
            best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}

    if best_state is not None:
        probe.load_state_dict(best_state)
    _, te_acc = run(te, train=False)
    print(f"\n=== probe (masked-skill) ({run_name}, init={args.init}) ===")
    print(f"best val acc  : {best_va:.4f}")
    print(f"test probe acc: {te_acc:.4f}")
    if wb is not None:
        wb.log({"test/probe_acc": te_acc, "best/val_acc": best_va})


if __name__ == "__main__":
    main()
