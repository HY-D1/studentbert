"""Transferability estimators for pretrained learner models (MRAP Task 2).

base.py      EstimatorResult and the ranking helper (numpy-free)
logme.py     corrected LogME and the per-next-skill variant (numpy only)
features.py  causal KT features and intent-based checkpoint loading (torch)
kt_logme.py  the two KT LogME scores for one candidate on one target (torch)

Nothing is imported here, so the numpy-only parts load on machines without torch.
"""
