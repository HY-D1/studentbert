"""SAINT+ — Choi et al. (SAINT, L@S 2020) + Shin et al. (SAINT+, LAK 2021).

Encoder-decoder Transformer for knowledge tracing.
  Encoder input  : the EXERCISE stream (skill ids) — "what was asked"
  Decoder input  : the RESPONSE stream (correctness + elapsed/time features) —
                   "how the student did", shifted so step t predicts step t.
  Output         : P(correct) for each position.

SAINT+ adds temporal features (elapsed time, lag time). Here we use the
available time_bin as the temporal feature (a reasonable stand-in for the
elapsed-time feature in the original paper).

This is a working, standard-config implementation (not tuned), per the task:
"working baselines with reasonable numbers, not tuned performance."

Shapes (B, L, ...):
  skill, correct, time_bin : (B, L)
  causal mask              : (L, L)  upper-triangular, prevents seeing future
  output logits            : (B, L)  one correctness logit per position
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SAINTPlus(nn.Module):
    def __init__(
        self,
        num_skills: int,
        num_time_bins: int = 5,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        # +1 on vocabs for PAD index 0
        self.skill_emb = nn.Embedding(num_skills + 1, d_model, padding_idx=0)
        # response stream: start token (2) + correct(0/1) -> 3 ids; PAD handled via mask
        self.resp_emb = nn.Embedding(3, d_model)         # 0=wrong,1=right,2=START
        self.time_emb = nn.Embedding(num_time_bins + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, n_heads, d_ff, dropout, batch_first=True
            ),
            num_layers=n_layers,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, n_heads, d_ff, dropout, batch_first=True
            ),
            num_layers=n_layers,
        )
        self.out = nn.Linear(d_model, 1)

    def _causal_mask(self, L: int, device) -> torch.Tensor:
        # True = masked (not allowed to attend). Upper triangle excluding diagonal.
        return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, skill, correct, time_bin, key_padding_mask=None):
        """key_padding_mask: (B, L) True where PAD (to ignore). Returns (B, L) logits."""
        B, L = skill.shape
        device = skill.device
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

        # Encoder: exercise (skill) + position + time
        enc_in = self.skill_emb(skill) + self.pos_emb(pos) + self.time_emb(time_bin)

        # Decoder: shifted response stream. Position 0 gets START(2); positions
        # 1..L-1 get the PREVIOUS step's correctness, so step t never sees its
        # own answer (prevents label leakage).
        resp_ids = torch.full((B, L), 2, dtype=torch.long, device=device)  # START
        resp_ids[:, 1:] = correct[:, :-1]
        dec_in = self.resp_emb(resp_ids) + self.pos_emb(pos)

        causal = self._causal_mask(L, device)

        memory = self.encoder(
            enc_in, mask=causal, src_key_padding_mask=key_padding_mask
        )
        dec = self.decoder(
            dec_in,
            memory,
            tgt_mask=causal,
            memory_mask=causal,
            tgt_key_padding_mask=key_padding_mask,
            memory_key_padding_mask=key_padding_mask,
        )
        return self.out(dec).squeeze(-1)  # (B, L)
