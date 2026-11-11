# Trajectory analysis, amendment 2: H-score on Quick-Tune micro (fixed before any score)

Follows `docs/qt_hscore_design.md`, whose features now exist for all 30 micro tasks and 24 models
(job 11805425; 720 files). This amendment fixes how they are scored and how H-score's choice
enters the trajectory analysis. It is committed with `analysis/qt_score.py` and
`analysis/qt_reversal.py` before either runs on the real features. Written on 2026-11-11.

## Scoring (`analysis/qt_score.py`)

- Candidates on a task: the models with a rule A run on it in `trajectories.tsv`
  (`quicktune_micro`). Other models' features are not scored.
- Images: for estimator seeds 42, 1 and 2, a stratified half of each class, 20 of 40 images,
  drawn with `random.Random(seed).sample` from each class's indices in file-name order; the same
  indices for every candidate. The script stops unless every candidate's file has the same file
  names and labels in the same order.
- Estimators: H-score, shrunk H-score and LogME, the functions of `scripts/task4_score.py` that
  scored the NLP benchmark, on float64 copies of the features. Each record keeps the score, the
  number of images and dimensions, and the seconds taken.
- LogME is undefined when the features interpolate the labels exactly, which happens when a
  model's feature dimension is at least the number of images (400, or 380 on DOG and PNU); the
  implementation then raises an error. Such a record keeps a null score and the reason, and
  LogME's ranking is reported over the candidates where it is defined, with the counts. Shrunk
  H-score, the recommendation, is defined for every candidate, and the analysis stops otherwise.

## Reversal and compute (`analysis/qt_reversal.py`)

- Recommendation: per task and seed, the candidate with the highest shrunk H-score (fixed in the
  design), ties broken by model name. Plain H-score and LogME recommendations are reported in the
  same way, as secondary rows.
- Final outcome: validation top-1 at epoch 50 of each candidate's rule A run. A recommendation is
  overtaken when its final score is more than 0.5 points below the task's best (the tie margin of
  `docs/trajectory_design.md`). Regret: the best final score minus the recommendation's.
- Warning flag at epoch k: another candidate's top-1 at epoch k exceeds the recommendation's.
  Followed: not flagged. Detection epoch: the first epoch from which the flag equals the final
  outcome at every later epoch, as in amendment 1.
- Reported at budgets 1, 2, 3, 5, 10, 20, 30, 40 and 50 epochs: overtaken recommendations caught,
  false alarms among the others, how many are followed and their mean regret, and the mean
  compute per decision (the sum over the task's candidates of their recorded seconds up to k).
  Also: each estimator's mean Spearman correlation with the final ranking per task, top-1
  agreement (tied with the best) and mean regret; the distribution of detection epochs; and the
  frozen-scoring compute (extraction seconds on the H200 from `extract_log.jsonl`), reported next
  to the fine-tuning compute and not added to it, since the two were measured on different GPUs.
- Limitations, stated with every result: rule A (models not compared under identical
  hyperparameters) and the image split (the estimators may see images on which Quick-Tune's
  curves were validated).
