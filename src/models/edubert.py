"""EduBERT — BERT-style encoder skeleton for student interaction sequences.

This is the backbone for StudentBERT pretraining. Week-2 scope: embedding +
positional encoding + standard transformer encoder + a forward pass that runs
on a real batch. The masked-interaction objective lives in masking.py and a
prediction head is included so week-3 pretraining can plug in directly.

Embedding (factored): each interaction token = sum of three learned embeddings
  skill_emb(skill) + outcome_emb(correct) + time_emb(time_bin)
plus a learned positional embedding. Factored embeddings (vs one compound
vocabulary of size num_skills*2*num_time_bins) keep the table small and let the
model share structure across skills/outcomes — and generalize across datasets
with different skill counts.

Shapes (B, L, ...):
  skill, correct, time_bin : (B, L)   ints; PAD=0 on every field
  hidden                   : (B, L, d_model)
  mlm logits (if head used):
      skill_logits   : (B, L, num_skills+1)
      correct_logits : (B, L, 2)
"""

from __future__ import annotations

import torch
import torch.nn as nn

# special outcome ids: 0=wrong, 1=right, 2=MASK  (PAD handled by key_padding_mask)
OUTCOME_MASK_ID = 2


class EduBERT(nn.Module):
    def __init__(
        self,
        num_skills: int,
        num_time_bins: int = 5,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.num_skills = num_skills
        self.d_model = d_model

        # +1 on skill/time for PAD idx 0; outcome has 3 ids (wrong/right/MASK)
        self.skill_emb = nn.Embedding(num_skills + 1, d_model, padding_idx=0)
        self.outcome_emb = nn.Embedding(3, d_model)
        self.time_emb = nn.Embedding(num_time_bins + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_drop = nn.Dropout(dropout)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, n_heads, d_ff, dropout, batch_first=True
            ),
            num_layers=n_layers,
        )

        # MLM heads: reconstruct masked skill and masked correctness
        self.skill_head = nn.Linear(d_model, num_skills + 1)
        self.correct_head = nn.Linear(d_model, 2)

    def encode(self, skill, correct, time_bin, key_padding_mask=None):
        """Bidirectional encode -> (B, L, d_model). No causal mask (BERT-style)."""
        B, L = skill.shape
        pos = torch.arange(L, device=skill.device).unsqueeze(0).expand(B, L)
        x = (
            self.skill_emb(skill)
            + self.outcome_emb(correct)
            + self.time_emb(time_bin)
            + self.pos_emb(pos)
        )
        x = self.emb_drop(self.emb_norm(x))
        # src_key_padding_mask: True where PAD (ignored by attention)
        return self.encoder(x, src_key_padding_mask=key_padding_mask)

    def forward(self, skill, correct, time_bin, key_padding_mask=None):
        """Returns dict with hidden states and MLM logits."""
        h = self.encode(skill, correct, time_bin, key_padding_mask)
        return {
            "hidden": h,                       # (B, L, d_model)
            "skill_logits": self.skill_head(h),    # (B, L, num_skills+1)
            "correct_logits": self.correct_head(h),  # (B, L, 2)
        }
