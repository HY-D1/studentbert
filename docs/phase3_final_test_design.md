# Phase 3 design: the final fresh-seed test of the three-epoch framework

Prof. Hazra's decision after phase 2 (RESULTS.md 12.18): the diagnosis and the warning signal are
the main contribution; correction stays secondary unless the three-epoch version holds on further
fresh seeds. She asked for one final fresh-seed evaluation with the rule fixed before any result is
seen. This file, the job generator and the evaluation code are committed together before any
phase 3 job is submitted, and none of them is changed afterwards.

## 1. The rule

The full framework at epoch 3, exactly as defined in phase 2 (docs/phase2_amendment1.md):

1. Take H-score's top-ranked encoder.
2. Fine-tune all 7 encoders for 3 epochs on the estimator's sample with the phase 2 settings
   (scripts/task4_early_ft.py: learning rate 5e-5, batch 32, at most 2,000 validation rows shared by
   all encoders).
3. Switch to the epoch-3 validation leader only if its validation accuracy exceeds that of
   H-score's pick by more than delta = 1.96 x sqrt(2 p (1 - p) / n), where p is the mean of the two
   accuracies and n the number of validation rows; otherwise keep H-score's pick. Ties for the
   leader go to the encoder with the higher H-score.

## 2. Data

- Fresh estimator seeds 6, 7 and 8, never used before (earlier seeds: 42, 1, 2, 3, 4, 5).
- Everything else as in phase 2: the 7 encoders, the 5 tasks with 2 poolings, frozen features and
  H-score and LogME scores extracted and scored into new folders, the published fine-tuned results
  as gold with the Task 4 tie rule, and 210 early fine-tunes of 3 epochs.
- The same limitation as in phase 2: only AG News, MNLI and QNLI draw new samples; Airline's
  samples overlap almost entirely with earlier ones, and RTE's are identical, so for those two only
  the fine-tuning randomness is new.

## 3. Primary test and what "holds" means

Primary outcome: the mean, over the 30 recommendations, of the full framework's regret minus
H-score's regret (published fine-tuned points), with a task-cluster bootstrap interval (the 5 tasks
resampled with replacement, 20,000 times, random.Random(0)).

**The three-epoch version holds if the 95% interval lies entirely below zero.** If it does, the
correction can be presented as a contribution, alongside the diagnosis and the warning signal. If it
does not, correction stays a secondary, exploratory result, whatever the point estimate. With five
tasks the interval is wide; that is accepted in advance and not a reason to change the criterion.

## 4. Secondary analyses, also fixed now

- The same difference for the early leader alone at epoch 3 (the ablation without H-score), the
  full framework and the early leader at epochs 1 and 2, LogME alone and the best-average rule.
- Reversal recovery, false corrections, harmless switches, wins, ties and losses against H-score,
  top-1, ties with the best, stability across the 3 seeds, and compute, as in phase 2.
- The warning signal at epochs 1 to 3 (reversals caught, false alarms).
- AG News, MNLI and QNLI alone.
- Phase 2 and phase 3 pooled (60 recommendations, seeds 3 to 8), for the primary difference only,
  with the same bootstrap. This is secondary because the phase 2 half has been seen.

## 5. Steps

1. Commit this file, slurm/generators/gen_phase3_jobs.sh and analysis/phase3_eval.py with its tests.
2. Extract and score the fresh seeds and build the head pairs (labels only; no fine-tuning result).
3. Run the 210 early fine-tunes.
4. Run analysis/phase3_eval.py once and record the result in RESULTS.md as it comes out.

## Commit note (added when this file was committed)

This file, the job generator and the evaluation code were delivered together before any phase 3
job, but the commit step did not run at the time. They were committed after the extractions, the
scoring, the head pairs and the submission of the 210 early fine-tunes, while no fine-tune had
finished and before any fine-tune output was read. Everything above this note is unchanged from
the delivered version: the first 3,567 bytes of this file have md5
d4bd10936c2e4b142c057697ac5adb01. The head pairs are labels (H-score's picks and the published
results) and say nothing about the early fine-tunes, so the rule and the criterion above are still
fixed before the data they are tested on.
