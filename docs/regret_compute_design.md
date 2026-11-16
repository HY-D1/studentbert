# Regret against compute: static selection, early leader, override rule, successive halving

Prof. Hazra (Nov 2026): for the NeurIPS methods question, compare H-score (static selection), the
early leader, our warning and override rule, and a simple multi-fidelity baseline such as
successive halving, under the same pre-specified decision criteria, reporting selection regret and
compute used, and showing where each method sits on the regret-against-compute frontier. This file
fixes the comparison before `analysis/regret_compute.py` is written or run. Written on 2026-11-16.

Status, stated plainly: H-score's, the early leader's and the rule's results on these data are
already known (RESULTS.md 12.13 to 12.26). New here are successive halving, the common compute
accounting and the frontier. The comparison is pre-specified for those parts and descriptive for
the rest.

## Decisions, candidates and outcomes (unchanged from the earlier sections)

- StudentBERT: 21 decisions (7 KT targets x estimator seeds 42, 1, 2), 8 candidates (7 encoders
  and scratch). Trajectories: validation AUC at epochs 1 to 20 of the fine-tune whose seed equals
  the estimator seed (`trajectories.tsv`). Outcome: the six-seed test AUC of the gold
  (`gold_a`, `gold_b` in `reversal_pairs.tsv`). H-score's choice: `cand_a` of the head rows.
- NLP: 90 decisions (10 settings, 5 tasks x 2 poolings, at estimator seeds 42, 1, 2; 3, 4, 5; and
  6, 7, 8), 7 encoders. Trajectories: validation accuracy at epochs 1 to 3 of the early fine-tunes
  (`early_ft`, `early_ft_fresh`, `early_ft_p3`). Outcome: the published fine-tuned mean
  (`tuned_pick`, `tuned_alt` of the head-pair files `nlp_head_pairs.tsv`, `p2_head_pairs.tsv`,
  `p3_head_pairs.tsv`). H-score's choice: their `pick`. Full fine-tuning curves do not exist here,
  so every NLP method stops at epoch 3.
- Quick-Tune micro: 90 decisions (30 tasks x estimator seeds 42, 1, 2), the models with at least 3
  complete runs on a task (12.26, 614 pairs). Trajectory of a model: at each epoch, the median
  top-1 over its complete runs; its cost per epoch: the median of those runs' training plus
  evaluation seconds. Outcome: the epoch-50 median, which is the model-level outcome of 12.26.
  H-score's choice: shrunk H-score (amendment 2), with plain H-score as a secondary row. This
  replaces rule A for this comparison, so its limitation does not apply; the outcome instead
  averages over Quick-Tune's sampled hyperparameters.
- Regret of a choice: the best outcome among the decision's candidates minus the choice's outcome
  (AUC on StudentBERT, points on NLP and Quick-Tune). Ties in validation scores within a method
  go to H-score's order.

## Methods

1. Static (H-score): its choice. Compute: frozen scoring of every candidate (feature extraction
   plus scoring), from the recorded times: `tg1_task2_kt_<target>.jsonl` (`hscore_kt_causal`) on
   StudentBERT, the extraction and scoring records of the NLP features, and the extraction log
   plus `qt_scores.jsonl` on Quick-Tune.
2. Early leader at epoch k: the candidate with the highest validation score after k epochs.
   Compute: every candidate fine-tuned for k epochs. Budgets: 1, 2, 3, 5, 10, 20 (StudentBERT);
   1, 2, 3 (NLP); 1, 2, 3, 5, 10, 20, 30, 40, 50 (Quick-Tune).
3. Override rule at epoch k (the pre-declared framework of 12.18): keep H-score's choice unless
   the epoch-k leader beats it by more than a margin, then switch. Margin: 1.96 x sqrt(2 p (1 - p)
   / n) on NLP (p the two encoders' mean validation accuracy, n the validation rows, as in 12.18);
   0.001 AUC on StudentBERT (as in 12.18 part B); 0.5 points on Quick-Tune, its tie margin, since
   Quick-Tune's validation size is not recorded. Compute: static plus every candidate for k epochs.
   Same budgets as the early leader.
4. Warning rule with abstention at epoch k: follow H-score's choice when it leads at epoch k,
   otherwise abstain. It does not choose on every decision, so it is reported as coverage and the
   regret of the followed decisions, beside the frontier and not on it. Compute as the override.
5. Successive halving: all candidates start; at each rung the survivors are trained to the rung's
   epoch count, ranked by validation score, and the better half (rounded up) continues; the last
   survivor is the choice. Primary: halving factor 2 with rung epochs r, 2r, 4r, ... capped at the
   largest recorded epoch, for starting budgets r = 1, 2, 5 on StudentBERT (8 candidates, last
   rung at 4, 8 and 20 epochs) and r = 1, 2, 3 on Quick-Tune (20 to 24 candidates, last rung at 16,
   32 and 48 epochs). On NLP, where curves stop at epoch 3, rungs at epochs 1, 2 and 3 (7, 4 and 2
   encoders), named a truncated variant. Compute: for each candidate, its cost up to the last epoch
   it was trained. Sensitivity, reported separately: halving factor 3.
6. Reference points: on StudentBERT, every candidate fine-tuned for 20 epochs and chosen by its
   best validation AUC (the practitioner's full fine-tuning); on Quick-Tune, every candidate for 50
   epochs (regret zero by construction, since the outcome is the same curve). NLP has none.

## Reported, per benchmark

- For each method and budget: mean selection regret with a 95% cluster bootstrap interval over
  targets or tasks (10,000 resamples, `random.Random(0)`), the share of decisions with zero
  regret, and the mean compute per decision in GPU-hours.
- The frontier: the points that no other point beats on both mean compute and mean regret (one of
  the two strictly), listed and plotted by a committed script.
- Compute is comparable within a benchmark only where it was measured on the same hardware.
  StudentBERT and NLP fine-tuning and scoring ran on the cluster (V100-SXM2 for the fine-tunes; the
  GPU of the StudentBERT scoring jobs is not in their records and is reported as such). On
  Quick-Tune, fine-tuning time is Quick-Tune's own on unrecorded hardware and frozen scoring ran on
  an H200, so the static point's compute is shown separately, not on the same axis.
- StudentBERT's per-epoch costs are each run's measured time spread evenly over its epochs; the
  StudentBERT results are retrospective, and the Quick-Tune outcome shares its curves with the
  methods' trajectories, as stated in 12.24 to 12.26.
