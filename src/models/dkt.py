"""DKT — Deep Knowledge Tracing (Piech et al., NeurIPS 2015).

Single-layer LSTM over interaction embeddings. At each step t the model has seen
interactions 1..t and predicts P(correct) for the NEXT interaction's skill.

Standard DKT formulation:
  input at step t  = embedding of interaction (skill_t, correct_t)
  output at step t = per-skill correctness logits; we read out the logit for
                     skill_{t+1} as the prediction for step t+1.

Shapes (B=batch, L=seq len, K=num_skills, H=hidden):
  skill, correct : (B, L)
  interaction id : (B, L)   = skill * 2 + correct, range [0, 2K)
  lstm out       : (B, L, H)
  logits         : (B, L, K+1)  (index 0 = PAD skill, ignored)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DKT(nn.Module):
    def __init__(self, num_skills: int, hidden_size: int = 128, dropout: float = 0.2):
        super().__init__()
        self.num_skills = num_skills
        # interaction vocab = 2 * (num_skills + 1) to include PAD skill 0
        self.n_interactions = 2 * (num_skills + 1)
        self.interaction_emb = nn.Embedding(self.n_interactions, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, num_skills + 1)  # +1 for PAD slot

    def forward(self, skill: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
        """Returns per-step, per-skill logits: (B, L, num_skills+1)."""
        interaction = skill * 2 + correct  # (B, L) in [0, 2K)
        x = self.interaction_emb(interaction)  # (B, L, H)
        h, _ = self.lstm(x)  # (B, L, H)
        h = self.dropout(h)
        return self.out(h)  # (B, L, K+1)

    @staticmethod
    def gather_next_step(logits: torch.Tensor, next_skill: torch.Tensor) -> torch.Tensor:
        """Pick the logit for the queried next skill at each step.
        logits: (B, L, K+1), next_skill: (B, L) -> (B, L)."""
        return torch.gather(logits, dim=2, index=next_skill.unsqueeze(-1)).squeeze(-1)
