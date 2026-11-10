# H-score on Quick-Tune micro (fixed before any feature is extracted or scored)

Purpose: give Quick-Tune an H-score recommendation, so the pre-specified trajectory analysis
(`docs/trajectory_design.md`, amendment 1) can ask on Quick-Tune what it asked on StudentBERT:
whether H-score's choice is overtaken after fine-tuning, when that becomes detectable, and at what
compute. Written on 2026-11-10, before any feature is extracted.

## Tasks, candidates and images

- The 30 Quick-Tune micro tasks. Images and labels: the Meta-Album micro releases on OpenML,
  fetched and checked by `analysis/qt_fetch_metaalbum.py` (report md5 1700e621): 40 images per
  class, 20 classes (19 on DOG and PNU).
- Candidates on a task: the models with a rule A run on it (`trajectories.tsv`, 20 to 24 per
  task), loaded with the timm 1.0.11 pretrained weights that the Quick-Tune names resolve to (for
  example `tf_efficientnet_b4_ns` to `tf_efficientnet_b4.ns_jft_in1k`).
- Images given to the estimators: for estimator seeds 42, 1 and 2, a stratified random half of
  each task, 20 images per class, drawn with `random.Random(seed)` from the images sorted by file
  name. The same images go to every candidate. Quick-Tune's own train and validation split is
  not recorded in the meta-dataset, so these images may include some on which its curves were
  validated; this is a limitation, stated wherever the scores are used.

## Features

- Each model's own timm evaluation preprocessing (`resolve_data_config` and `create_transform`):
  the 128-pixel Meta-Album images are resized to the model's input size (224 to 600 pixels) and
  normalized as in pretraining.
- The pooled features before the classifier: `forward_head(forward_features(x), pre_logits=True)`,
  in evaluation mode, without gradients, in float32. Features of all images of a task are
  extracted once per model and saved, so the three seeds and every estimator use the same
  features; time, GPU and peak memory are recorded per model and task.

## Scores and the recommendation

- Estimators, as implemented in `scripts/task4_score.py` for the NLP benchmark: H-score, shrunk
  H-score (Ledoit-Wolf) and LogME (one-hot, averaged over classes).
- Primary recommendation: the candidate with the highest shrunk H-score. Reason, fixed here: with
  400 images, many candidates have more feature dimensions than images (up to 2,560 for
  EfficientNet-B7), where the plain H-score's covariance is singular and its values are not
  comparable across candidates; shrinkage is the standard remedy and was already one of the
  estimators on NLP. Plain H-score and LogME are reported alongside.
- The trajectory analysis then follows amendment 1 with this recommendation: overtaken when its
  final (epoch 50) top-1 is more than 0.5 points below the best of the task; the early decision
  flags when another candidate leads it at epoch k; detection epoch, compute and regret as defined
  there. A separate amendment will fix the details before that analysis runs.
