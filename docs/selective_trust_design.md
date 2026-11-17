# Paper C: selective trust in the static recommendation (fixed before any computation)

Prof. Hazra (Nov 2026), for Paper C: the override rule is not the methods contribution; the
promising part is the warning, or selective-trust, signal. The questions are when the static
(H-score) recommendation can be trusted, when to abstain or keep evaluating, and how much compute
that decision needs; whether one common warning rule or threshold transfers across domains or each
domain needs its own budget; with successive halving and the early leader as the multi-fidelity
comparisons (12.27). This file fixes the analysis before `analysis/selective_trust.py` is written
or run. Written on 2026-11-17. Already known: the warning rule at threshold zero (12.17, 12.20,
12.27); new are the threshold family, the normalized units, the cross-domain transfer test and the
budget question.

## Decisions

The same as 12.27, loaded with the same code (`analysis/regret_compute.py`): 21 on StudentBERT,
90 on NLP (curves to epoch 3) and 90 on Quick-Tune micro (models with at least 3 complete runs,
median curves, model-level median outcome). The static recommendation is H-score's choice
(shrunk H-score on Quick-Tune, as pre-specified there).

## The signal, in units shared by every domain

- After k epochs, the normalized lead of the static choice: L = (its validation score minus the
  best other candidate's) divided by the spread of all candidates' validation scores at epoch k
  (best minus worst); L = 0 when the spread is zero. L lies between -1 and +1 whatever the metric.
- Trust the static choice when L is at least a threshold t; otherwise do not (abstain, or keep
  evaluating). t = 0 is the warning rule of 12.20 and 12.27 (trust when the static choice leads).
  Thresholds: -1.00 to +1.00 in steps of 0.05.
- Budgets: k = 1, 2, 3, 5, 10, 20 (StudentBERT); 1, 2, 3 (NLP); 1, 2, 3, 5, 10, 20, 30, 40, 50
  (Quick-Tune); k = 0 means no fine-tuning and trusting every decision.
- Normalized regret, to compare domains: a decision's regret divided by the domain's mean regret of
  a random candidate (the mean over its decisions of the mean regret over candidates).

## Reported

1. Per domain, budget and threshold: coverage (share of decisions trusted), the mean regret of the
   trusted static choices (selective regret, raw and normalized) and the compute per decision
   (as in 12.27; Quick-Tune's frozen scoring off the fine-tuning axis). Per domain and budget: the
   area under the normalized selective-regret against coverage curve over the thresholds.
2. A common fixed rule: t = 0 at each budget, in every domain.
3. Cross-domain transfer of a threshold, leave one domain out, at the budgets all three domains
   share (k = 1, 2, 3): on the two other domains, pooled, choose the threshold with the highest
   coverage whose normalized selective regret is at most 0.10; apply it to the held-out domain and
   compare its coverage and normalized selective regret with the held-out domain's own best
   threshold under the same target.
4. The domain-specific budget: per domain, the smallest budget at which some threshold reaches a
   normalized selective regret of at most 0.10 with coverage of at least 0.50, in epochs and, on
   StudentBERT and Quick-Tune, as a share of the compute of fine-tuning every candidate fully.
5. For reference beside each operating point: the early leader and successive halving at the same
   or nearest compute (12.27), as the multi-fidelity comparisons.
6. Intervals: 95% cluster bootstrap over targets or tasks (10,000 resamples, `random.Random(0)`)
   for coverage and selective regret at the reported operating points.

The targets 0.10 and 0.50 are fixed here, before any computation. Limitations, stated with the
result: StudentBERT is retrospective, NLP stops at epoch 3, Quick-Tune's outcome shares its curves
with the signal (its selective regret at 50 epochs is zero by construction), and the threshold-zero
results were known before.
