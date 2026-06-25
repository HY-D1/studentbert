"""Two downstream tasks on ASSISTments using a (pretrained or scratch) EduBERT encoder.

  --task dropout      : student-level early-dropout prediction (binary).
                        Label per student: dropped = (gap_to_last_activity > 30 days)
                        AND (total interactions < dataset mean). Sequence -> mean-pooled
                        encoder state -> binary head. Reports AUC + F1 (minority class)
                        + positive-class rate (advisor asked to track this).

  --task next_skill   : predict skill_{t+1} from history up to t (multiclass over K).
                        Encoder run CAUSALLY (no future peeking). Reports top-1 and
                        top-5 accuracy (primary) + macro one-vs-rest AUC. Also prints
                        the skill-frequency distribution head/tail (advisor: macro AUC
                        can behave oddly under heavy skew).

Cross-dataset-safe encoder load (same as finetune_edubert.py): keeps only shape-
matching tensors, so a pretrained EdNet/Junyi encoder transfers (skill_emb/head are
re-init for the target vocab).

Usage:
  PYTHONPATH=. python scripts/downstream_edubert.py --task dropout \
    --processed_dir ../processed/assist2017 --init pretrained \
    --encoder_ckpt ../checkpoints/edubert_ednet_pretrain_full_encoder.pt \
    --epochs 30 --run_type ednet_dropout --wandb

  PYTHONPATH=. python scripts/downstream_edubert.py --task next_skill \
    --processed_dir ../processed/assist2017 --init scratch \
    --epochs 30 --run_type scratch_nextskill --wandb

NOTE on dropout labels: ASSISTments processed npz does NOT carry raw timestamps
(only time_bin 1-5). The "30 days inactivity" rule needs wall-clock gaps, which are
not in sequences.npz. We approximate dropout with a proxy that IS reconstructable
from the npz (short total activity + trailing large time-gaps). If you want the exact
30-day rule, we must re-derive per-student last-activity dates from the raw CSV during
preprocessing and store a dropout label in the npz. This is flagged clearly at runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.eval.metrics import auc
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


# ---------------------------------------------------------------------------
# Dataset wrappers
# ---------------------------------------------------------------------------
class StudentSeqDataset(Dataset):
    """Loads one split; exposes per-student (skill, correct, time_bin) + a
    precomputed dropout label (for --task dropout)."""

    def __init__(self, processed_dir, split, max_seq_len, dropout_labels=None, k_prefix=None):
        d = Path(processed_dir)
        data = np.load(d / "sequences.npz")
        self.skill = data["skill"]; self.correct = data["correct"]
        self.time_bin = data["time_bin"]; self.offsets = data["offsets"]
        self.student_ids = data["student_ids"]
        self.max_seq_len = max_seq_len
        splits = json.loads((d / "splits.json").read_text())
        wanted = set(int(s) for s in splits[split])
        id_to_row = {int(sid): i for i, sid in enumerate(self.student_ids)}
        self.rows = [id_to_row[s] for s in wanted if s in id_to_row]
        self.dropout_labels = dropout_labels  # dict row->0/1 or None
        self.k_prefix = k_prefix  # if set: use FIRST k interactions (early-pred)
        if dropout_labels is not None:
            # keep only eligible students (those present in the label dict)
            self.rows = [r for r in self.rows if r in dropout_labels]

    def seq(self, row):
        s, e = self.offsets[row], self.offsets[row + 1]
        sk = self.skill[s:e].astype(np.int64)
        co = self.correct[s:e].astype(np.int64)
        tb = self.time_bin[s:e].astype(np.int64)
        if self.k_prefix is not None:
            # early-prediction: return FIRST k tokens, prefix taken in __getitem__
            return sk, co, tb
        if len(sk) > self.max_seq_len:
            sk, co, tb = sk[-self.max_seq_len:], co[-self.max_seq_len:], tb[-self.max_seq_len:]
        return sk, co, tb

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        sk, co, tb = self.seq(row)
        if self.k_prefix is not None:
            sk, co, tb = sk[:self.k_prefix], co[:self.k_prefix], tb[:self.k_prefix]
        item = {"skill": torch.from_numpy(sk), "correct": torch.from_numpy(co),
                "time_bin": torch.from_numpy(tb), "length": len(sk)}
        if self.dropout_labels is not None:
            item["label"] = float(self.dropout_labels[row])
        return item


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
    out = {"skill": skill, "correct": correct, "time_bin": time_bin,
           "mask": mask, "length": lengths}
    if "label" in batch[0]:
        out["label"] = torch.tensor([b["label"] for b in batch], dtype=torch.float)
    return out


# ---------------------------------------------------------------------------
# Dropout label construction (proxy from npz; see module docstring caveat)
# ---------------------------------------------------------------------------
def build_dropout_labels(processed_dir, max_seq_len):
    """Load the real dropout label file (built by build_dropout_labels.py)."""
    import json
    d = Path(processed_dir)
    lf = d / "dropout_labels.json"
    if not lf.exists():
        raise SystemExit(f"missing {lf}; run scripts/build_dropout_labels.py first")
    raw = json.loads(lf.read_text())
    meta = raw.pop("_meta", {})
    npz = np.load(d / "sequences.npz")
    sid_to_row = {int(s): i for i, s in enumerate(npz["student_ids"])}
    labels = {}
    for sid_str, lab in raw.items():
        sid = int(sid_str)
        if sid in sid_to_row:
            labels[sid_to_row[sid]] = int(lab)
    for sid, row in sid_to_row.items():
        labels.setdefault(row, 0)
    pos_rate = float(np.mean([labels[r] for r in range(len(sid_to_row))]))
    return labels, pos_rate, meta.get("threshold_interactions", -1)


class EduBERTForDropout(nn.Module):
    """Mean-pool encoder states (over real tokens) -> binary logit."""
    def __init__(self, num_skills, d_model=256, n_layers=6, dropout=0.1, max_len=512):
        super().__init__()
        self.backbone = EduBERT(num_skills=num_skills, d_model=d_model,
                                n_layers=n_layers, dropout=dropout, max_len=max_len)
        self.head = nn.Linear(d_model, 1)

    def forward(self, skill, correct, time_bin, key_padding_mask):
        h = self.backbone.encode(skill, correct, time_bin,
                                 key_padding_mask=key_padding_mask)  # (B,L,d)
        real = (~key_padding_mask).unsqueeze(-1).float()             # (B,L,1)
        pooled = (h * real).sum(1) / real.sum(1).clamp(min=1)        # (B,d) mean-pool
        return self.head(pooled).squeeze(-1)                         # (B,)


class EduBERTForNextSkill(nn.Module):
    """Causal encoder -> predict skill_{t+1} (multiclass over K+1)."""
    def __init__(self, num_skills, d_model=256, n_layers=6, dropout=0.1, max_len=512):
        super().__init__()
        self.backbone = EduBERT(num_skills=num_skills, d_model=d_model,
                                n_layers=n_layers, dropout=dropout, max_len=max_len)
        self.num_skills = num_skills
        self.head = nn.Linear(d_model, num_skills + 1)

    def _causal(self, L, device):
        return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, skill, correct, time_bin, key_padding_mask):
        bb = self.backbone
        B, L = skill.shape
        pos = torch.arange(L, device=skill.device).unsqueeze(0).expand(B, L)
        x = (bb.skill_emb(skill) + bb.outcome_emb(correct)
             + bb.time_emb(time_bin) + bb.pos_emb(pos))
        x = bb.emb_drop(bb.emb_norm(x))
        h = bb.encoder(x, mask=self._causal(L, skill.device),
                       src_key_padding_mask=key_padding_mask)
        return self.head(h)  # (B, L, K+1)


def load_encoder(model, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    state = ck["model_state"]; tgt = model.backbone.state_dict()
    keep = {k: v for k, v in state.items() if k in tgt and v.shape == tgt[k].shape}
    skipped = sorted(k for k in state if k not in keep)
    model.backbone.load_state_dict(keep, strict=False)
    print(f"loaded {len(keep)}/{len(state)} encoder tensors from {ckpt_path}")
    if skipped:
        print(f"  skipped (vocab-specific): {skipped}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def f1_minority(y_true, y_prob, thresh=0.5):
    yp = (np.asarray(y_prob) >= thresh).astype(int)
    yt = np.asarray(y_true).astype(int)
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def f1_at_rate(y_true, y_prob):
    """F1 for the positive class using a threshold that matches the true positive
    rate (predict the top-r fraction as positive, r = mean(y_true)). Avoids the
    default-0.5 artifact when probabilities are uncalibrated."""
    import numpy as np
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_prob)
    r = yt.mean()
    if r == 0 or r == 1:
        return 0.0
    k = max(1, int(round(r * len(yp))))
    thresh = np.sort(yp)[::-1][k - 1]      # k-th largest prob
    pred = (yp >= thresh).astype(int)
    tp = int(((pred == 1) & (yt == 1)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def topk_acc(logits, target, valid, k):
    """logits (N,K+1), target (N,), valid (N,) bool. Returns top-k accuracy."""
    lt = logits[valid]; tt = target[valid]
    if len(tt) == 0:
        return 0.0
    topk = lt.topk(k, dim=-1).indices            # (n,k)
    hit = (topk == tt.unsqueeze(-1)).any(-1).float()
    return hit.mean().item()


def macro_ovr_auc(probs, target, valid, present_classes):
    """One-vs-rest AUC averaged over classes that appear in y_true."""
    from src.eval.metrics import auc as _auc
    p = probs[valid].cpu().numpy()               # (n, K+1)
    t = target[valid].cpu().numpy()              # (n,)
    aucs = []
    for c in present_classes:
        yt = (t == c).astype(int)
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue                              # skip degenerate classes
        try:
            aucs.append(_auc(yt, p[:, c]))
        except Exception:
            continue
    return float(np.mean(aucs)) if aucs else float("nan"), len(aucs)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def run_dropout(model, loader, device, opt=None, sched=None, wb=None):
    train = opt is not None
    model.train() if train else model.eval()
    bce = nn.BCEWithLogitsLoss()
    ys, ps, tot, nb = [], [], 0.0, 0
    for b in loader:
        skill = b["skill"].to(device); correct = b["correct"].to(device)
        tb = b["time_bin"].to(device); mask = b["mask"].to(device)
        label = b["label"].to(device)
        with torch.set_grad_enabled(train):
            logit = model(skill, correct, tb, key_padding_mask=~mask)
            loss = bce(logit, label)
            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if sched: sched.step()
                if wb: wb.log({"train/lr": sched.get_last_lr()[0]})
        tot += loss.item(); nb += 1
        ys.append(label.detach().cpu().numpy())
        ps.append(torch.sigmoid(logit).detach().cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    return tot / max(nb, 1), y, p


def run_next_skill(model, loader, device, opt=None, sched=None, wb=None,
                   collect=False):
    train = opt is not None
    model.train() if train else model.eval()
    ce = nn.CrossEntropyLoss(ignore_index=0)  # 0 = PAD skill, ignore as target
    tot, nb = 0.0, 0
    all_logits, all_tgt, all_valid = [], [], []
    for b in loader:
        skill = b["skill"].to(device); correct = b["correct"].to(device)
        tb = b["time_bin"].to(device); mask = b["mask"].to(device)
        with torch.set_grad_enabled(train):
            logits = model(skill, correct, tb, key_padding_mask=~mask)  # (B,L,K+1)
            tgt = skill[:, 1:]                       # next-skill target
            pred = logits[:, :-1]                    # predict from step t
            valid = mask[:, 1:] & mask[:, :-1]
            loss = ce(pred.reshape(-1, pred.size(-1)), (tgt * valid).reshape(-1))
            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if sched: sched.step()
                if wb: wb.log({"train/lr": sched.get_last_lr()[0]})
        tot += loss.item(); nb += 1
        if collect:
            all_logits.append(pred.reshape(-1, pred.size(-1)).detach().cpu())
            all_tgt.append(tgt.reshape(-1).detach().cpu())
            all_valid.append(valid.reshape(-1).detach().cpu())
    if collect:
        return (tot / max(nb, 1), torch.cat(all_logits),
                torch.cat(all_tgt), torch.cat(all_valid))
    return tot / max(nb, 1), None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["dropout", "next_skill"], required=True)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--init", choices=["pretrained", "scratch"], required=True)
    ap.add_argument("--encoder_ckpt", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup_frac", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--n_students", type=int, default=0,
                    help="next_skill: subset train to N students (0=all)")
    ap.add_argument("--k_prefix", type=int, default=20,
                    help="dropout: use first K interactions (early prediction)")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=6)
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
    print(f"device={device}  task={args.task}  num_skills={num_skills}  "
          f"run={run_name}  init={args.init}  seed={args.seed}")

    # ---- skill distribution check (advisor asked, for next_skill) ----
    if args.task == "next_skill":
        data = np.load(Path(args.processed_dir) / "sequences.npz")
        counts = np.bincount(data["skill"][data["skill"] > 0], minlength=num_skills + 1)
        nz = counts[counts > 0]
        print(f"skill freq: {len(nz)} skills present; "
              f"top5={np.sort(counts)[::-1][:5].tolist()}; "
              f"min nonzero={int(nz.min())}; median={int(np.median(nz))}; "
              f"max/median skew={nz.max()/max(1,np.median(nz)):.1f}x")

    wb = None
    if args.wandb:
        import wandb
        wb = wandb
        wandb.init(project="StudentBERT", entity="dhy666666o-n",
                   name=run_name, tags=["edubert", "downstream", args.task, args.init],
                   config={"task": args.task, "dataset": dataset, "init": args.init,
                           "run_type": args.run_type, "epochs": args.epochs,
                           "batch_size": args.batch_size, "lr": args.lr,
                           "dropout": args.dropout, "seed": args.seed,
                           "num_skills": num_skills,
                           "encoder_ckpt": args.encoder_ckpt or "none"})

    # ---- build datasets ----
    if args.task == "dropout":
        labels, pos_rate, mean_total = build_dropout_labels(args.processed_dir, args.max_seq_len)
        print(f"[dropout] label=bottom-quartile disengagement; positive rate={pos_rate:.4f}")
        if wb: wb.log({"data/pos_rate": pos_rate})
        def mk(split): return StudentSeqDataset(args.processed_dir, split,
                                                args.max_seq_len, labels,
                                                k_prefix=args.k_prefix)
        Model = EduBERTForDropout
    else:
        def mk(split):
            ds = StudentSeqDataset(args.processed_dir, split, args.max_seq_len)
            if split == "train" and args.n_students and args.n_students < len(ds.rows):
                import random as _r
                rng = _r.Random(args.seed)
                ds.rows = rng.sample(ds.rows, args.n_students)
            return ds
        Model = EduBERTForNextSkill

    def loader(split, sh):
        return DataLoader(mk(split), batch_size=args.batch_size, shuffle=sh,
                          collate_fn=collate)
    train_loader = loader("train", True)
    val_loader = loader("val", False)
    test_loader = loader("test", False)

    model = Model(num_skills=num_skills, d_model=args.d_model, n_layers=args.n_layers,
                  dropout=args.dropout, max_len=args.max_seq_len).to(device)
    if args.init == "pretrained":
        load_encoder(model, args.encoder_ckpt, device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = args.epochs * max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, make_lr_lambda(total_steps, args.warmup_frac))

    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{run_name}_best.pt"
    best_metric = -1.0

    for ep in range(1, args.epochs + 1):
        if args.task == "dropout":
            tr_loss, *_ = run_dropout(model, train_loader, device, opt, sched, wb)
            _, yv, pv = run_dropout(model, val_loader, device)
            val_auc = auc(yv, pv)
            print(f"epoch {ep:3d}  loss={tr_loss:.4f}  val_AUC={val_auc:.4f}")
            if wb: wb.log({"epoch": ep, "train/loss": tr_loss, "val/auc": val_auc})
            metric = val_auc
        else:
            tr_loss, *_ = run_next_skill(model, train_loader, device, opt, sched, wb)
            _, lg, tg, vd = run_next_skill(model, val_loader, device, collect=True)
            t1 = topk_acc(lg, tg, vd, 1)
            print(f"epoch {ep:3d}  loss={tr_loss:.4f}  val_top1={t1:.4f}")
            if wb: wb.log({"epoch": ep, "train/loss": tr_loss, "val/top1": t1})
            metric = t1
        if metric > best_metric:
            best_metric = metric
            torch.save({"model_state": model.state_dict(), "epoch": ep,
                        "metric": metric, "config": vars(args)}, best_path)

    # ---- test with best checkpoint ----
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])

    print(f"\n=== {args.task} ({run_name}, init={args.init}) ===")
    if args.task == "dropout":
        _, yt, pt = run_dropout(model, test_loader, device)
        test_auc = auc(yt, pt); test_f1 = f1_minority(yt, pt)
        test_f1_rate = f1_at_rate(yt, pt)
        test_pos = float(np.mean(yt))
        print(f"test AUC          : {test_auc:.4f}")
        print(f"test F1 (minority): {test_f1:.4f}")
        print(f"test F1 (rate-matched): {test_f1_rate:.4f}")
        print(f"test pos-rate     : {test_pos:.4f}")
        if wb: wb.log({"test/auc": test_auc, "test/f1": test_f1, "test/f1_rate": test_f1_rate, "test/pos_rate": test_pos})
    else:
        _, lg, tg, vd = run_next_skill(model, test_loader, device, collect=True)
        t1 = topk_acc(lg, tg, vd, 1); t5 = topk_acc(lg, tg, vd, 5)
        probs = torch.softmax(lg, dim=-1)
        present = np.unique(tg[vd].numpy())
        present = present[present > 0]
        m_auc, n_cls = macro_ovr_auc(probs, tg, vd, present.tolist())
        print(f"test top-1 acc    : {t1:.4f}")
        print(f"test top-5 acc    : {t5:.4f}")
        print(f"test macro-OVR AUC: {m_auc:.4f}  (over {n_cls} classes)")
        if wb: wb.log({"test/top1": t1, "test/top5": t5, "test/macro_auc": m_auc})

    if wb: wb.finish()


if __name__ == "__main__":
    main()
