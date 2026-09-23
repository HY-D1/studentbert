"""Frozen-feature extraction for knowledge-tracing transferability, matching the fine-tune protocol.

Three properties are deliberate, each tested in tests/test_estimators.py:

1. Causal information regime. hidden[t] is paired with correct[t+1], exactly the fine-tune's loss
   positions, and comes from the same causal mask as EduBERTForKT.encode_causal
   (scripts/finetune_edubert.py lines 86 to 99). scripts/compute_logme.py used the bidirectional
   EduBERT.encode, so hidden[t] attended to position t+1, whose input contains the label.
2. Loading by intent. Vocabulary tensors (skill_emb, skill_head) are loaded only when the
   checkpoint's own config.processed_dir names the target dataset. The fine-tune decides by shape,
   which is correct today only because all seven vocabularies differ in size.
3. Same random start as the fine-tune. The backbone is built immediately after set_seed(seed),
   the RNG position the fine-tune uses, so every tensor the candidate does not supply (the
   cross-domain skill table) equals the fine-tune's initialisation at that seed, and is identical
   across candidates.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import InteractionDataset, collate_fn
from src.models.edubert import EduBERT
from src.utils import set_seed

VOCAB_PREFIXES = ("skill_emb.", "skill_head.")


def target_num_skills(processed_dir: str | Path) -> int:
    vocab = json.loads((Path(processed_dir) / "skill_vocab.json").read_text())
    return max(int(v) for v in vocab.values())


def build_backbone(num_skills: int, *, seed: int, d_model: int = 256, n_layers: int = 6,
                   dropout: float = 0.1, max_len: int = 512) -> EduBERT:
    set_seed(seed)
    return EduBERT(num_skills=num_skills, d_model=d_model, n_layers=n_layers, dropout=dropout,
                   max_len=max_len)


def source_dataset_of(ckpt: dict, ckpt_path: str | Path) -> str:
    cfg = ckpt.get("config") or {}
    pd = cfg.get("processed_dir")
    if not pd:
        raise ValueError(f"{ckpt_path}: no config.processed_dir, so in-domain versus "
                         "cross-domain cannot be decided by intent")
    return Path(pd).name


def load_candidate(backbone: EduBERT, ckpt_path: str | Path, target_name: str,
                   device: str = "cpu") -> dict:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not (isinstance(ck, dict) and "model_state" in ck):
        raise ValueError(f"{ckpt_path}: expected a pretrain checkpoint dict with model_state")
    state = ck["model_state"]
    source = source_dataset_of(ck, ckpt_path)
    in_domain = source == target_name
    own = backbone.state_dict()
    keep, skipped = {}, []
    for key, value in state.items():
        if key.startswith(VOCAB_PREFIXES) and not in_domain:
            skipped.append(key)
            continue
        if key not in own:
            raise ValueError(f"{ckpt_path}: {key} does not exist in the target model")
        if value.shape != own[key].shape:
            raise ValueError(f"{ckpt_path}: {key} has shape {tuple(value.shape)}, target model "
                             f"expects {tuple(own[key].shape)}")
        keep[key] = value
    missing, _ = backbone.load_state_dict(keep, strict=False)
    lost = [k for k in missing if not k.startswith(VOCAB_PREFIXES)]
    if lost:
        raise ValueError(f"{ckpt_path}: shared tensors missing from the checkpoint: {lost}")
    mlm = ck.get("mlm_loss")
    return {"source": source, "in_domain": in_domain, "loaded": len(keep), "total": len(state),
            "skipped": sorted(skipped), "ckpt_epoch": ck.get("epoch"),
            "ckpt_mlm_loss": float(mlm) if isinstance(mlm, (int, float)) else None}


def encode_causal(backbone: EduBERT, skill, correct, time_bin, key_padding_mask):
    """Replicates EduBERTForKT.encode_causal in scripts/finetune_edubert.py."""
    B, L = skill.shape
    pos = torch.arange(L, device=skill.device).unsqueeze(0).expand(B, L)
    x = (backbone.skill_emb(skill) + backbone.outcome_emb(correct)
         + backbone.time_emb(time_bin) + backbone.pos_emb(pos))
    x = backbone.emb_drop(backbone.emb_norm(x))
    causal = torch.triu(torch.ones(L, L, device=skill.device, dtype=torch.bool), diagonal=1)
    return backbone.encoder(x, mask=causal, src_key_padding_mask=key_padding_mask)


def sample_target(processed_dir: str | Path, n_students: int | None, seed: int,
                  max_seq_len: int = 512):
    """The fine-tune's learner draw (first_n_students in finetune_edubert.py) for this seed."""
    ds = InteractionDataset(str(processed_dir), "train", max_seq_len)
    if n_students is None:
        rows = list(ds.rows)
        subset = ds
    else:
        order = np.random.default_rng(seed).permutation(len(ds))[:n_students].tolist()
        rows = [ds.rows[i] for i in order]
        subset = Subset(ds, order)
    fingerprint = hashlib.sha1(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()[:16]
    return subset, rows, fingerprint


@torch.no_grad()
def kt_features(backbone: EduBERT, subset, device: str = "cpu", *, batch_size: int = 64,
                causal: bool = True):
    """Features at the fine-tune's loss positions: hidden[t] for label correct[t+1], plus the
    next skill for the per-skill readout. causal=False reproduces the old leaky extractor and
    exists only for the negative-control test."""
    backbone.eval()
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    feats, labels, next_skills = [], [], []
    for batch in loader:
        skill = batch["skill"].to(device)
        correct = batch["correct"].to(device)
        time_bin = batch["time_bin"].to(device)
        mask = batch["mask"].to(device)
        if causal:
            h = encode_causal(backbone, skill, correct, time_bin, ~mask)
        else:
            h = backbone.encode(skill, correct, time_bin, key_padding_mask=~mask)
        next_valid = mask[:, 1:] & mask[:, :-1]
        feats.append(h[:, :-1][next_valid].float().cpu().numpy())
        labels.append(correct[:, 1:][next_valid].cpu().numpy())
        next_skills.append(skill[:, 1:][next_valid].cpu().numpy())
    return (np.concatenate(feats), np.concatenate(labels).astype(np.int64),
            np.concatenate(next_skills).astype(np.int64))


def cap_positions(n: int, max_positions: int | None, seed: int) -> np.ndarray:
    """Candidate-independent subsample of positions: depends only on seed and the target count."""
    if max_positions is None or n <= max_positions:
        return np.arange(n)
    return np.sort(np.random.default_rng([seed, 1]).permutation(n)[:max_positions])
