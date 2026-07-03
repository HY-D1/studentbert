# Week 6 Experiments Map

## Baselines (Task 3)
- w6_akt_baseline / w6_akt_ednet / w6_akt_junyi / w6_akt_ednet_s2: AKT (Ghosh 2020) on assist2017/ednet/junyi, 3 seeds.

## Task 1 - cross-dataset transfer (KT + next-skill), N=3000 subsampled
- w6_kt_ednet_sub / w6_kt_junyi_sub: KT into EdNet/Junyi targets, 4 conditions x 3 seeds.
- w6_ns_ednet_sub / w6_ns_junyi_sub: next-skill into EdNet/Junyi targets.
- w6_kt_assist_n3000 / w6_ns_assist_n3000: budget-matched ASSISTments target (granularity control).

## Task 2 - dropout K-sweep (ASSISTments)
- w6_dropoutK_s42/s1/s2: K=5,10,20,50,100,200 x 4 cond x 3 seeds. Clean K<=50; K>=100 leak panel.

## Task 1c - dropout on EdNet/Junyi targets (K=5,10 only; larger K leaks 100%)
- w6_drop_junyi_k5/k10: Junyi target.
- w6_de_{scratch,indomain,fromassist,fromjunyi}_k{5,10}: EdNet target, split by condition.
- w6_de_id_k10_s{42,1,2}: EdNet indomain K=10 split to one-seed (slow condition).

## Task 4 - probing (masked-skill decodability; probe_edubert_v2.py)
- w6_probe2: ASSISTments, 4 cond x 3 seeds.
- w6_probe2_targets: EdNet + Junyi targets, 8 cond x 3 seeds.

## Notes
- EdNet/Junyi downstream subsampled to n_students=3000 (full-data times out).
- Dropout on EdNet/Junyi limited to K=5,10 (K>=20 reveals 100% via short sequences).
- EdNet dropout is high-variance/inconclusive across seeds.
- probe_edubert_v1 DEPRECATED (circular: skill visible in input -> scratch ~99.9%).
