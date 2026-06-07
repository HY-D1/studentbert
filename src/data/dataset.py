"""PyTorch Dataset + collate for StudentBERT interaction sequences.

Loads the packed output of the preprocess_*.py scripts:
    sequences.npz  -> student_ids, skill, correct, time_bin, offsets (CSR-style)
    splits.json    -> {"train": [...ids], "val": [...], "test": [...]}

Each item is one student's sequence, truncated to max_seq_len (keep MOST RECENT
interactions, since next-step prediction cares about recent history). Variable
lengths are padded in the collate fn; PAD index is 0 for every field.

Tensor shapes (per batch of B sequences, padded to length L):
    skill     : (B, L)  int64
    correct   : (B, L)  int64   values {0,1}, PAD=0  (use mask to ignore PAD)
    time_bin  : (B, L)  int64   values {1..5}, PAD=0
    mask      : (B, L)  bool    True where real token, False where PAD
    length    : (B,)    int64   true (pre-pad) length of each sequence
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PAD_IDX = 0


class InteractionDataset(Dataset):
    def __init__(self, processed_dir: str, split: str, max_seq_len: int = 512):
        """processed_dir: folder containing sequences.npz + splits.json.
        split: 'train' | 'val' | 'test'.
        """
        d = Path(processed_dir)
        data = np.load(d / "sequences.npz")
        self.student_ids = data["student_ids"]
        self.skill = data["skill"]
        self.correct = data["correct"]
        self.time_bin = data["time_bin"]
        self.offsets = data["offsets"]
        self.max_seq_len = max_seq_len

        splits = json.loads((d / "splits.json").read_text())
        wanted = set(splits[split])
        # map student_id -> row index so we can select this split's sequences
        id_to_row = {int(sid): i for i, sid in enumerate(self.student_ids)}
        self.rows = [id_to_row[s] for s in wanted if s in id_to_row]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        s, e = self.offsets[row], self.offsets[row + 1]
        skill = self.skill[s:e].astype(np.int64)
        correct = self.correct[s:e].astype(np.int64)
        time_bin = self.time_bin[s:e].astype(np.int64)

        # keep most recent max_seq_len interactions
        if len(skill) > self.max_seq_len:
            skill = skill[-self.max_seq_len:]
            correct = correct[-self.max_seq_len:]
            time_bin = time_bin[-self.max_seq_len:]

        return {
            "skill": torch.from_numpy(skill),
            "correct": torch.from_numpy(correct),
            "time_bin": torch.from_numpy(time_bin),
            "length": len(skill),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad a list of variable-length items to the batch max length. PAD=0."""
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    max_len = int(lengths.max())
    B = len(batch)

    skill = torch.zeros(B, max_len, dtype=torch.long)
    correct = torch.zeros(B, max_len, dtype=torch.long)
    time_bin = torch.zeros(B, max_len, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, b in enumerate(batch):
        n = b["length"]
        skill[i, :n] = b["skill"]
        correct[i, :n] = b["correct"]
        time_bin[i, :n] = b["time_bin"]
        mask[i, :n] = True

    return {
        "skill": skill,
        "correct": correct,
        "time_bin": time_bin,
        "mask": mask,
        "length": lengths,
    }
