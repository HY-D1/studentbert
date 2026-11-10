# Trajectory analysis, amendment 1 (fixed before the analysis runs)

Prof. Hazra (Nov 2026) approved rule A for Quick-Tune and asked for the pre-specified trajectory
and compute analysis on StudentBERT and Quick-Tune. This amendment fixes the details that
`docs/trajectory_design.md` left open. It is committed with `analysis/trajectory_analysis.py`
before that script is run on the real table. While the code was written, it was test-run on the
StudentBERT scores with stand-in costs, to check that it reproduces RESULTS.md 12.17, 12.18 and
12.20; no definition below was changed after that run.

## Limitation of rule A, to be stated wherever Quick-Tune results appear

Each Quick-Tune model is represented by its first complete run, and every run has its own random
hyperparameters. The models on a task are therefore not compared under identical
hyperparameters: a model can rank low because its run drew poor settings, not because it
transfers worse. The comparison is between one sampled fine-tune per model, which is how a
practitioner without a tuning budget would meet them.

## StudentBERT

- Recommendations: the 21 of RESULTS.md 12.13 (7 targets x estimator seeds 42, 1 and 2). H-score's
  choice is `cand_a` of the head rows of `reversal_pairs.tsv`; the candidates are it and the 7
  alternatives of those rows (all 8, scratch included, as in the warning rule of 12.17).
- Trajectories: the fine-tunes of `trajectories.tsv` whose seed equals the estimator seed.
- Overtaken: a head pair labelled `reversal` (main label), as in 12.20.
- Early decision at epoch k: flag when an alternative's validation AUC exceeds that of H-score's
  choice; the early leader is the candidate with the highest validation AUC, ties going to
  H-score's choice.
- Regret of a candidate: the best mean test AUC among the 8 minus its own (`gold_a`, `gold_b`).
- Detection epoch and compute to a reliable decision as defined in the design, for epochs 1 to 20;
  compute is the sum over the 8 candidates of their cumulative seconds.

## Quick-Tune (micro, mini, extended reported separately; extended is thin)

- Candidates: the rule A runs of each task. Final score: validation top-1 at epoch 50; tied with
  the best when within 0.5 points (fixed in the design).
- Early leader at epoch k: the candidate with the highest top-1 at epoch k, ties broken by model
  name. Its regret: the best final score minus the early leader's final score.
- Winner-detection epoch (added here, because H-score's choice does not exist yet on Quick-Tune):
  the first epoch from which the early leader is tied with the best final score at every later
  epoch. Compute to that decision: the sum over the task's candidates of their cumulative seconds
  up to that epoch, also as a share of running all of them for 50 epochs.

## Reported

For both benchmarks: the mean regret of the early leader and, on StudentBERT, the number and mean
regret of the recommendations the warning rule follows, at budgets of 1, 2, 3, 5, 10 and 20
epochs (Quick-Tune also 30, 40 and 50), each with the mean compute per decision; the distribution
of detection epochs; and the compute at detection. Nothing is selected or tuned on these results.
