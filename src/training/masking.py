"""Masked-interaction objective for EduBERT (BERT-style MLM on interactions).

Randomly select ~15% of REAL (non-PAD) positions per sequence. For each selected
position, following BERT's 80/10/10 recipe:
  80%: replace outcome with MASK id, and skill with a [MASK] skill (id 0 used as
       a learnable "unknown"; here we keep skill but mask the outcome, which is
       the KT-meaningful target — predicting whether the answer was correct).
  10%: replace with a random outcome.
  10%: leave unchanged.
Only the selected positions contribute to the loss (others are ignored via -100).

The advisor's framing: "mask entire student interactions ... predict what they
were, including whether the student got it right." So we predict BOTH the skill
and the correctness at masked positions. This module returns corrupted inputs
plus label tensors (with -100 at non-masked / PAD positions for CrossEntropy).

Shapes (B, L):
  inputs unchanged shape; labels are (B, L) with -100 where not predicted.
"""

from __future__ import annotations

import torch

from src.models.edubert import OUTCOME_MASK_ID

IGNORE = -100  # CrossEntropyLoss ignore_index


def mask_interactions(
    skill: torch.Tensor,
    correct: torch.Tensor,
    time_bin: torch.Tensor,
    pad_mask: torch.Tensor,   # (B, L) True where REAL token
    num_skills: int,
    mask_ratio: float = 0.15,
    generator: torch.Generator | None = None,
):
    """Return (skill_in, correct_in, time_in, skill_labels, correct_labels).

    Corrupted inputs feed the model; labels hold targets at masked positions and
    IGNORE elsewhere.
    """
    device = skill.device
    B, L = skill.shape

    # candidate positions = real tokens
    rand = torch.rand(B, L, device=device, generator=generator)
    selected = (rand < mask_ratio) & pad_mask  # (B, L) bool

    skill_in = skill.clone()
    correct_in = correct.clone()
    time_in = time_bin.clone()

    skill_labels = torch.full_like(skill, IGNORE)
    correct_labels = torch.full_like(correct, IGNORE)
    # set labels at selected positions to the ORIGINAL values
    skill_labels[selected] = skill[selected]
    correct_labels[selected] = correct[selected]

    # 80/10/10 split among selected
    r2 = torch.rand(B, L, device=device, generator=generator)
    do_mask = selected & (r2 < 0.8)
    do_random = selected & (r2 >= 0.8) & (r2 < 0.9)
    # remaining 10% (r2 >= 0.9): leave unchanged

    # 80% -> MASK the outcome; skill set to PAD-id 0 as "unknown skill" sentinel
    correct_in[do_mask] = OUTCOME_MASK_ID
    skill_in[do_mask] = 0

    # 10% -> random outcome (0/1) and random skill in [1, num_skills]
    if do_random.any():
        n = int(do_random.sum())
        correct_in[do_random] = torch.randint(0, 2, (n,), device=device, generator=generator)
        skill_in[do_random] = torch.randint(1, num_skills + 1, (n,), device=device, generator=generator)

    return skill_in, correct_in, time_in, skill_labels, correct_labels
