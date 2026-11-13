# Quick-Tune amendment 3: model-level outcome and dimension control (fixed before it runs)

Prof. Hazra's reply to RESULTS.md 12.25 (Nov 2026): do not use the best single run as the main
target, since it rewards lucky hyperparameters and drifts toward selecting a model together with
its tuning; define a model-level outcome by aggregating each model's complete runs on a task,
with best-run accuracy only as a sensitivity analysis; and check whether the conclusions hold
after controlling or normalizing for representation dimensionality. This amendment fixes how,
before `analysis/qt_model_level.py` runs on the real files. Written on 2026-11-11, after 12.25 was
seen; this is a follow-up analysis, not a replacement of 12.25.

## Outcome

- For each model and micro task, the complete runs are those whose curve has 50 epochs (all
  hyperparameter configurations, not only the rule A run). Final accuracy: top-1 at epoch 50.
- Primary outcome: the median final accuracy over the complete runs. Reason: 38% of runs end near
  chance (12.25), and the median is less moved by failed runs than the mean. Secondary: the mean.
  Sensitivity: the best run.
- A model enters a task only with at least 3 complete runs there; the counts of included and
  excluded model-task pairs are reported. A ranking (one task and seed) needs at least 3 models
  with a defined score; raw LogME, undefined for large models (12.25), ranks the rest.

## Scores

- Raw: the scores already recorded in `qt_scores.jsonl` (12.25), for the included models.
- Equal dimension: each model's features on the same scored images (the stratified halves of
  seeds 42, 1 and 2, as in amendment 2) are centred and projected on their top 128 principal
  components, computed from those images without labels (128 is the smallest feature dimension in
  the set; the 128-dimensional model is only centred and rotated). H-score, shrunk H-score and
  LogME are then computed with the same functions. LogME is defined for every model here, since
  128 is below the 380 or 400 images.
- The recommendation is again the shrunk H-score's choice (amendment 2); plain H-score and LogME
  are secondary rows.

## Reported, for each outcome (median, mean, best run)

- For each estimator, with raw and with equal-dimension features: the mean per-task Spearman
  correlation of its scores with the outcome; how often its choice is tied (within 0.5 points)
  with the best model; the mean regret of its choice; and the mean Spearman correlation of its
  scores with feature dimension.
- For raw features: the partial Spearman correlation of score and outcome controlling for the
  logarithm of feature dimension, per task and seed, then averaged.
- The mean regret of a random model, and the mean Spearman correlation of the outcome with
  feature dimension.
- Limitation, as before: the estimators may see images on which Quick-Tune's curves were
  validated. Rule A is not used here, so its limitation does not apply; the model-level outcome
  instead averages over Quick-Tune's sampled hyperparameters.
