# Task 4: transferability estimation via LogME (You et al. ICML 2020).
# Loads a (frozen) pretrained EduBERT encoder, extracts features on a TARGET
# dataset, computes LogME for predicting next-step correctness (KT). No fine-tuning.
# Higher LogME = predicted better transfer. Correlate scores vs measured gains.
#
# Usage (run in sb env, on a GPU node or login is fine since it is light):
#   PYTHONPATH=. python compute_logme.py --encoder_ckpt <enc.pt> \
#       --target_dir ../processed/<target> --max_students 3000 --seed 42
# init=scratch (no ckpt) gives the scratch baseline LogME.
import argparse, json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, ".")
from src.data.dataset import InteractionDataset, collate_fn
from src.models.edubert import EduBERT

def infer_num_skills(processed_dir):
    v = json.loads((Path(processed_dir) / "skill_vocab.json").read_text())
    return max(int(i) for i in v.values())

def load_encoder(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    own = model.state_dict()
    # transfer encoder/pos/time/outcome/norm; skip vocab-specific skill_emb/skill_head
    keep = {k: v for k, v in state.items()
            if k in own and v.shape == own[k].shape
            and not k.startswith("skill_emb") and not k.startswith("skill_head")}
    skipped = sorted(k for k in state if k not in keep)
    model.load_state_dict(keep, strict=False)
    print(f"loaded {len(keep)}/{len(state)} encoder tensors from {ckpt_path}")
    if skipped:
        print(f"  skipped (vocab-specific or mismatched): {skipped[:6]}")

def logme(F, y, max_iter=200, tol=1e-3):
    N, D = F.shape
    F = F - F.mean(0, keepdims=True)
    y = y.astype(np.float64); y = y - y.mean()
    U, s, _ = np.linalg.svd(F, full_matrices=False)
    sigma = s**2
    z = U.T @ y; z2 = z**2; y2 = float((y**2).sum())
    alpha, beta = 1.0, 1.0
    for _ in range(max_iter):
        gamma = float((beta*sigma/(alpha+beta*sigma)).sum())
        m2 = float(((beta**2)*sigma*z2/(alpha+beta*sigma)**2).sum())
        alpha_new = gamma/(m2+1e-12)
        res = y2 - float(((beta*sigma*z2*(2*(alpha+beta*sigma)-beta*sigma))/(alpha+beta*sigma)**2).sum())
        res = max(res, 1e-8)
        beta_new = (N-gamma)/res
        if abs(alpha_new-alpha)<tol and abs(beta_new-beta)<tol:
            alpha, beta = alpha_new, beta_new; break
        alpha, beta = alpha_new, beta_new
    m = beta*np.sqrt(sigma)*z/(alpha+beta*sigma)
    term = (0.5*D*np.log(alpha) + 0.5*N*np.log(beta)
            - 0.5*np.sum(np.log(alpha+beta*sigma))
            - 0.5*beta*(y2 - float((beta*sigma*z2/(alpha+beta*sigma)).sum()))
            - 0.5*alpha*float((m**2).sum())
            - 0.5*N*np.log(2*np.pi))
    return term/N

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_ckpt", default=None, help="omit for scratch")
    ap.add_argument("--target_dir", required=True)
    ap.add_argument("--max_students", type=int, default=3000)
    ap.add_argument("--max_positions", type=int, default=50000,
                    help="cap total (position) samples for LogME speed")
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=6)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  target={Path(args.target_dir).name}  ckpt={args.encoder_ckpt or 'SCRATCH'}")

    K = infer_num_skills(args.target_dir)
    ds = InteractionDataset(args.target_dir, "train", args.max_seq_len)
    # subsample students
    rng = np.random.default_rng(args.seed)
    if args.max_students < len(ds.rows):
        idx = rng.permutation(len(ds.rows))[:args.max_students]
        ds.rows = [ds.rows[i] for i in idx]
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    model = EduBERT(num_skills=K, d_model=args.d_model, n_layers=args.n_layers,
                    max_len=args.max_seq_len).to(device)
    if args.encoder_ckpt:
        load_encoder(model, args.encoder_ckpt, device)
    else:
        print("scratch: random init encoder")
    model.eval()

    feats, labels = [], []
    with torch.no_grad():
        for batch in loader:
            skill = batch["skill"].to(device)
            correct = batch["correct"].to(device)
            time_bin = batch["time_bin"].to(device)
            pad = batch["mask"].to(device)  # True=real
            key_pad = ~pad
            h = model.encode(skill, correct, time_bin, key_padding_mask=key_pad)  # (B,L,D)
            # KT target: predict correct[t+1] from hidden[t]. Use positions 0..L-2, label=correct[t+1].
            B, L, D = h.shape
            for b in range(B):
                valid = pad[b].sum().item()
                if valid < 2: continue
                hb = h[b, :valid-1, :].cpu().numpy()       # hidden[t], t=0..valid-2
                yb = correct[b, 1:valid].cpu().numpy()      # correct[t+1]
                feats.append(hb); labels.append(yb)
    F = np.concatenate(feats, 0); y = np.concatenate(labels, 0).astype(np.int64)
    # cap positions for speed
    if len(y) > args.max_positions:
        sel = rng.permutation(len(y))[:args.max_positions]
        F = F[sel]; y = y[sel]
    print(f"features: {F.shape}  positive rate: {y.mean():.3f}")
    score = logme(F, y)
    print(f"LOGME_RESULT target={Path(args.target_dir).name} ckpt={Path(args.encoder_ckpt).name if args.encoder_ckpt else 'scratch'} logme={score:.6f}")

if __name__ == "__main__":
    main()
