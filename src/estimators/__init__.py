"""Transferability estimators for pretrained learner models (MRAP Task 2).

base.py      EstimatorResult and the ranking helper (numpy-free)
logme.py     corrected LogME and the per-next-skill variant (numpy only)
features.py  causal KT features and intent-based checkpoint loading (torch)
kt_logme.py  the two KT LogME scores for one candidate on one target (torch)
hscore.py    H-score, plain and Ledoit-Wolf shrunk (numpy only)
nleep.py     NLEEP (numpy only)
parc.py      PARC (numpy and scipy)
protocol.py  target-sample protocol shared by the scoring CLIs (no torch)

Nothing is imported here, so the numpy-only parts load on machines without torch.
"""
