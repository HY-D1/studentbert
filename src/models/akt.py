# AKT - Context-Aware Attentive Knowledge Tracing (Ghosh, Heffernan, Lan, KDD 2020).
# Compact reimplementation matching this repo's baseline interface (same forward
# signature as SAINT+: returns per-step (B, L) correctness logits, takes a
# key_padding_mask, uses a causal mask internally so step t attends to <= t).
# Ingredients: Rasch/IRT skill embeddings, monotonic exponential-decay attention,
# question + knowledge encoders, knowledge retriever. Response stream is SHIFTED
# (pos t carries correct[t-1], pos 0 = START) so step t never sees its own answer.

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotonicAttention(nn.Module):
    # Multi-head attention with AKT's monotonic exponential-decay reweighting.
    # Scores are scaled by exp(-theta * dist) where dist is a context distance
    # from cumulative attention mass; theta >= 0 is a learned per-head decay.
    # Causal mask restricts attention to tau <= t.

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.theta = nn.Parameter(torch.zeros(n_heads))

    def forward(self, q, k, v, key_padding_mask=None):
        B, L, _ = q.shape
        H, dh = self.n_heads, self.d_head

        def shape(x):
            return x.view(B, L, H, dh).transpose(1, 2)

        Q, K, V = shape(self.q_proj(q)), shape(self.k_proj(k)), shape(self.v_proj(v))
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(dh)

        causal = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal.view(1, 1, L, L), float("-inf"))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.view(B, 1, 1, L), float("-inf"))

        with torch.no_grad():
            base = torch.softmax(scores, dim=-1)
            base = torch.nan_to_num(base)
            pos = torch.arange(L, device=q.device)
            gap = (pos.view(L, 1) - pos.view(1, L)).clamp(min=0).float()
            cum = torch.cumsum(base.flip(-1), dim=-1).flip(-1)
            dist = gap.view(1, 1, L, L) * cum

        theta = F.softplus(self.theta).view(1, H, 1, 1)
        decay = torch.exp(-theta * dist)
        scores = scores + torch.log(decay + 1e-9)

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)
        attn = self.dropout(attn)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)


class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class AKTBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MonotonicAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        h = self.ln1(x)
        x = x + self.drop(self.attn(h, h, h, key_padding_mask=key_padding_mask))
        h = self.ln2(x)
        x = x + self.drop(self.ffn(h))
        return x


class AKT(nn.Module):
    # Context-Aware Attentive Knowledge Tracing, skill-level. Forward signature
    # matches SAINT+ so it drops into the baseline trainer: returns (B, L) logits.

    def __init__(self, num_skills: int, d_model: int = 256, n_heads: int = 8,
                 n_blocks: int = 2, d_ff: int = 1024, dropout: float = 0.1,
                 max_len: int = 512):
        super().__init__()
        self.num_skills = num_skills
        K = num_skills + 1
        self.skill_emb = nn.Embedding(K, d_model, padding_idx=0)
        self.variation = nn.Embedding(K, d_model, padding_idx=0)
        self.mu = nn.Embedding(K, 1, padding_idx=0)
        # response embeddings size 3: 0/1 = correctness, 2 = START (for the shift)
        self.resp_emb = nn.Embedding(3, d_model)
        self.resp_variation = nn.Embedding(3, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.q_blocks = nn.ModuleList(
            [AKTBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_blocks)]
        )
        self.k_blocks = nn.ModuleList(
            [AKTBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_blocks)]
        )
        self.retriever = MonotonicAttention(d_model, n_heads, dropout)
        self.ln_out = nn.LayerNorm(d_model)
        self.out = nn.Linear(2 * d_model, 1)

    def _rasch_skill(self, skill):
        c = self.skill_emb(skill)
        d = self.variation(skill)
        mu = self.mu(skill)
        return c + mu * d

    def _rasch_interaction(self, skill, correct):
        # SHIFTED response: pos 0 = START(2), pos t = correct[t-1], so step t
        # never encodes its own answer (prevents label leakage).
        B, L = skill.shape
        resp_ids = torch.full((B, L), 2, dtype=torch.long, device=skill.device)
        resp_ids[:, 1:] = correct[:, :-1]
        x_s = self._rasch_skill(skill)
        r = self.resp_emb(resp_ids)
        rv = self.resp_variation(resp_ids)
        return x_s + r + rv

    def forward(self, skill, correct, time_bin=None, key_padding_mask=None):
        # skill, correct: (B, L). key_padding_mask: (B, L) True=PAD. Returns (B, L).
        B, L = skill.shape
        pos = torch.arange(L, device=skill.device).unsqueeze(0).expand(B, L)
        pe = self.pos_emb(pos)
        q = self.drop(self._rasch_skill(skill) + pe)
        k = self.drop(self._rasch_interaction(skill, correct) + pe)
        for blk in self.q_blocks:
            q = blk(q, key_padding_mask=key_padding_mask)
        for blk in self.k_blocks:
            k = blk(k, key_padding_mask=key_padding_mask)
        retrieved = self.retriever(q, k, k, key_padding_mask=key_padding_mask)
        retrieved = self.ln_out(retrieved)
        feat = torch.cat([retrieved, q], dim=-1)
        return self.out(feat).squeeze(-1)
