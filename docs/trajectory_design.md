# Per-epoch trajectories and compute (fixed before the analysis)

Prof. Hazra asked (Nov 2026) to keep, for every check, each candidate's per-epoch ranking
trajectory and its compute cost, for a later analysis of when an H-score reversal becomes
detectable and how much compute a reliable decision needs. This file fixes the table and the
definitions before any of them is computed. The analysis is descriptive: every result it uses was
seen before this file was written, except the Quick-Tune curves.

## One table for every benchmark

`analysis/trajectory_table.py` writes `benchmark_final/trajectories.tsv`, one row per candidate
per epoch: benchmark, target, setting, seed, candidate, epoch, metric, score, rank among the
candidates of the same (benchmark, target, setting, seed) at that epoch (1 is best, ties share
the lower rank), epoch seconds, cumulative seconds, cost source, GPU, run id and source file.

- StudentBERT (`studentbert`): `early_ft_epochs.tsv`, validation AUC at epochs 1 to 20 for the 7
  targets x 8 candidates x 6 seeds of the gold. Cost: each run's measured elapsed time from
  `executions.tsv`, matched by log file, divided evenly over its epochs (cost source
  `elapsed_even`, an approximation; start-up and test evaluation are included).
- NLP (`nlp_early`): the early fine-tune records of the three seed sets (42, 1, 2; 3, 4, 5;
  6, 7, 8), validation accuracy at epochs 1 to 3 for 7 encoders, 5 tasks and 2 poolings. Cost: the
  measured `epoch_s` of each epoch (cost source `measured`).
- Quick-Tune (`quicktune_micro`, `quicktune_mini`, `quicktune_extended`): rule A below,
  validation top-1 accuracy (percent) at epochs 1 to 50. Cost: the recorded training plus
  evaluation time of each epoch (cost source `measured`; the GPU is not recorded).

## Quick-Tune rule A (chosen by Harry, Nov 2026)

Each run in Quick-Tune has its own random hyperparameters, and no configuration is shared across
models on any task (`analysis/qt_inspect.py`). Each model on each task is therefore represented by
its first complete run: the first row of `unnormalized_args_table.csv`, in file order, whose curve
has 50 epochs. The rule uses no result of the run. Runs with fewer epochs are not used; a model
without a complete run on a task is absent from that task, and the counts are reported.

## Definitions for the later analysis

- Final ranking: the ranking at the last epoch of the trajectory (StudentBERT: the gold of
  RESULTS.md 12; Quick-Tune: epoch 50; NLP: the published fine-tuned gold, since the early
  fine-tunes stop at epoch 3).
- Detection epoch of a reversal: the first epoch k from which the early decision (whether H-score's
  choice is the validation leader) agrees with the final outcome at every later epoch of the
  trajectory. Defined only where the trajectory reaches the final epoch (StudentBERT, Quick-Tune).
- Compute to a reliable decision: the cumulative seconds of all candidates of the ranking up to the
  detection epoch.
- Regret against compute: for each budget k, the regret of following the early leader, and of the
  warning rule's followed recommendations, against the cumulative seconds of all candidates up to
  k. On Quick-Tune, ties within 0.5 accuracy points count as tied with the best; this margin is
  fixed here because single runs give no seed noise to estimate one from.
- H-score on Quick-Tune requires frozen features of the 24 models on the Meta-Album images and is
  a separate step; until it exists, the Quick-Tune analysis can only use the trajectories.
