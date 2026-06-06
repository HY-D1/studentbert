"""Smoke test: confirm EduBERT runs a forward pass and masking produces correct
shapes on a real batch from the PyTorch Dataset.

Usage:
    python scripts/test_edubert_forward.py --processed_dir data/full/assist2017
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import InteractionDataset, collate_fn
from src.models.edubert import EduBERT
from src.training.masking import mask_interactions
from src.utils import get_device, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_seq_len", type=int, default=512)
    args = ap.parse_args()

    set_seed(42)
    device = get_device()
    vocab = json.loads((Path(args.processed_dir) / "skill_vocab.json").read_text())
    num_skills = max(vocab.values())

    ds = InteractionDataset(args.processed_dir, "train", args.max_seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))
    skill = batch["skill"].to(device)
    correct = batch["correct"].to(device)
    time_bin = batch["time_bin"].to(device)
    pad_mask = batch["mask"].to(device)  # True=real

    print(f"device={device}  num_skills={num_skills}")
    print(f"batch shapes: skill={tuple(skill.shape)} correct={tuple(correct.shape)} "
          f"time_bin={tuple(time_bin.shape)}")

    model = EduBERT(num_skills=num_skills, d_model=256, n_heads=8, n_layers=6,
                    max_len=args.max_seq_len).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"EduBERT params: {n_params:,}")

    # 1) plain forward pass
    out = model(skill, correct, time_bin, key_padding_mask=~pad_mask)
    print("forward OK:")
    print(f"  hidden         {tuple(out['hidden'].shape)}")
    print(f"  skill_logits   {tuple(out['skill_logits'].shape)}  (expect B,L,{num_skills+1})")
    print(f"  correct_logits {tuple(out['correct_logits'].shape)}  (expect B,L,2)")

    # 2) masking + forward + loss
    s_in, c_in, t_in, s_lab, c_lab = mask_interactions(
        skill, correct, time_bin, pad_mask, num_skills, mask_ratio=0.15
    )
    n_masked = int((c_lab != -100).sum())
    print(f"masked positions: {n_masked} (~15% of real tokens)")

    out2 = model(s_in, c_in, t_in, key_padding_mask=~pad_mask)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    K = num_skills + 1
    loss_skill = loss_fn(out2["skill_logits"].reshape(-1, K), s_lab.reshape(-1))
    loss_correct = loss_fn(out2["correct_logits"].reshape(-1, 2), c_lab.reshape(-1))
    print(f"MLM loss  skill={loss_skill.item():.4f}  correct={loss_correct.item():.4f}")
    print("\nAll EduBERT checks passed.")


if __name__ == "__main__":
    main()
