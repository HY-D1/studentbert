# Option 1 design: the recommendation-level reversal test on the NLP benchmark

Approved by Prof. Hazra: run the NLP early fine-tunes, starting with a 5-run timing and protocol
check, keep the epoch-3 rule fixed for a test on data that do not exist yet, and compare H-score,
the fixed rule and the learned head-pair model with held-out-task evaluation. This file is committed
before the first early fine-tune runs; nothing below may change after any NLP early fine-tuning
result has been seen. Anything added later is labelled as added after the data.

## Data

- Candidates: the 7 encoders of Task 4. Settings: AGNews, MNLI, QNLI, RTE and Airline, each with
  first-token and mean pooling (10 settings); the 3 estimator seeds (42, 1, 2): 30 recommendations.
- Recommendation: H-score's top-ranked encoder per setting and seed (existing Task 4 scores).
- Labels: the head pairs of step A (analysis/nlp_head_pairs.py), from the published tuned results
  of Bassignana et al. (2022) with the Task 4 tie rule: 180 pairs (71 stable, 66 near-tie,
  43 reversal); 20 of 30 recommendations reversed, from 4 of 5 tasks. Airline has none and is the
  control a method must not flag.
- Early fine-tuning (scripts/task4_early_ft.py): every encoder, setting and seed fine-tuned in full
  for 3 epochs at learning rate 5e-5, batch 32, AdamW without weight decay or schedule, maximum
  length 256, on exactly the estimators' training sample (10,000 examples; RTE's whole training
  split), validation accuracy after every epoch on at most 2,000 validation rows shared by all
  encoders (the official validation split where one exists, otherwise training rows outside the
  sample); no test split is read. 210 runs, all on V100-SXM2.

## Methods compared

1. Always follow H-score.
2. The fixed rule, fixed on 2026-10-13 before these data existed: follow H-score unless its pick is
   not the validation leader among the 7 encoders at epoch 3 of the fine-tune with the estimator's
   seed; otherwise abstain. Nothing is fitted; it is applied to all 30 recommendations.
3. The learned head-pair model of version 2 (analysis/reversal_model_v2.py): L2 logistic regression
   on the head pairs, noisy-OR recommendation risk, penalty and threshold (Youden's J) chosen inside
   each fold from the training tasks only, a fixed 0.5 threshold reported beside it. Features:
   frozen (H-score and LogME margins standardized within the ranking, ranks), early fine-tuning
   (validation accuracy at epochs 1 to 3, slopes, ranks among the 7, rank change, gap to the epoch-3
   leader, whether the alternative closes in), history (each encoder's published tuned rank on the
   other tasks). No layer features.
4. Reference rows: the best-average rule of Task 4 and the frozen margin to the runner-up.

Evaluation is leave-one-task-out (5 folds; both poolings of a task are held out together because
they share data). The rule needs no folds; it is still reported per held-out task.

## Outcomes

Recommendation level: reversals caught, false alarms, coverage, regret and selective regret (tuned
accuracy points against the best encoder), AURC, calibration (Brier score and bins with counts).
Compute cost: GPU-minutes per recommendation for the early fine-tunes (7 encoders x epochs), from
the measured run times, against the cost of fine-tuning all 7 encoders in full.

Primary comparisons: the fixed rule against always following H-score, on selective regret and
coverage; the learned model against the fixed rule, on the same outcomes. Secondary: the rule and
the model with epochs 1 and 2 only (the minimum useful budget), and the robustness of every result
to dropping Airline. With 5 tasks and 30 recommendations every estimate is coarse; counts are
reported with every rate.

## Steps

1. Commit this file with scripts/task4_early_ft.py, its tests and the generator.
2. Timing and protocol check: 5 runs (bert-base-uncased, first-token pooling, seed 42, one per
   task). Pass: every run finishes with 3 epoch records, validation accuracy above the majority
   class on every task by epoch 3, no out-of-memory error; the measured time sets the walltime.
3. The full 210 runs; then the collector and the evaluation, written and committed before the
   collector first reads the full set.

## Commit note (added when this file was committed)

This file was delivered before the 5-run timing and protocol check, but it was committed only after
that check had run and after the full set of 210 runs had been submitted, because the first commit
step was skipped. Everything above this note is unchanged from the delivered version: the first
4,266 bytes of this file have md5 2454fcd5ac5f7beb5e92d5d20194da30. The check ran one encoder per
task, so it could not show whether H-score's pick led at epoch 3, and no output of the full set was
read before this commit. The epoch-3 rule and every other choice above are therefore still fixed
before the data they are tested on.
