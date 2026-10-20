# Phase 2 design: from a warning signal to a complete decision rule

Agreed with Prof. Hazra on 2026-10-20, after the results were frozen (pipeline results write-up of
2026-10-20): test whether the early fine-tuning warning can become a complete decision rule that
keeps H-score's pick when it is reliable and uses the early fine-tuning leader when it is flagged,
evaluated on fresh estimator seeds. This file is committed before any fresh-seed run starts and is
not changed afterwards; anything added later is labelled as added after the data.

## 1. A property of the rule, stated before the data

With the warning used so far (flag when H-score's pick is not the validation leader), the complete
rule always ends at the early leader: when the pick leads, keeping it is choosing the leader, and
when it does not, the rule takes the leader. So the rule as stated is the same as always taking the
early fine-tuning leader, and H-score has no influence on the final choice. For H-score to
contribute, the warning has to tolerate small gaps. Both versions are therefore tested:

- **R1, the complete rule as stated.** At epoch k, if H-score's pick is the validation leader among
  the 7 encoders, keep it; otherwise take the leader. (Equivalently, take the epoch-k leader.)
- **R2, the margin-gated rule.** At epoch k, take the leader only if its validation accuracy
  exceeds the pick's by more than the validation noise, delta = 1.96 x sqrt(2 p (1 - p) / n), where
  p is the mean of the two accuracies and n the number of validation rows; otherwise keep H-score's
  pick. Delta is a fixed formula, not a tuned value. On 2,000 rows at an accuracy near 0.8 it is
  about 2.5 accuracy points.

Ties for the leader go to the encoder with the higher H-score.

## 2. Data

- Candidates, tasks and poolings as in the NLP test: 7 encoders; AG News, MNLI, QNLI, RTE and
  Airline; first-token and mean pooling.
- Fresh estimator seeds: 3, 4 and 5, with new frozen features and new H-score and LogME scores,
  extracted and scored exactly as in Task 4 into separate output folders, so the frozen results are
  not touched. The samples are only partly new: AG News, MNLI and QNLI draw new 10,000-example
  samples from large training splits; Airline's 10,000 of 10,248 rows overlap almost entirely with
  the earlier samples; RTE uses its whole training split for every seed, so its frozen scores and
  H-score's pick repeat (up to tiny GPU differences) and only the fine-tuning randomness is new.
  Results are therefore also reported for AG News, MNLI and QNLI alone.
- Early fine-tunes: the same script and settings as the NLP test (3 epochs, learning rate 5e-5,
  batch 32, validation accuracy after every epoch on at most 2,000 rows shared by all encoders),
  on the fresh samples: 210 runs, written to a separate folder.
- Gold: the published fine-tuned results of Bassignana et al. (2022), as before. The gold does not
  depend on the estimator seed, so what is fresh is the sample, the H-score recommendation and the
  early fine-tuning, not the tasks. 30 fresh recommendations.

## 3. Methods compared

1. Always follow H-score.
2. R1 at epoch 1 (primary), and at epochs 2 and 3.
3. R2 at epoch 1, and at epochs 2 and 3.
4. The best-average rule of Task 4 (mean published fine-tuned rank on the other tasks); it does not
   depend on the seed.
5. For reference: the warning-only rule of the NLP test (abstain when flagged), reported with its
   coverage, so the fresh seeds also replicate the frozen result.

Every method chooses one encoder per recommendation, so coverage is 30 of 30 except for item 5.

## 4. Outcomes

- Mean regret in published fine-tuned points against the best encoder, with a task-cluster
  bootstrap interval (5 tasks, 20,000 resamples).
- How often the choice is tied with the best under the Task 4 tie rule, and how often it is the
  best (top-1).
- Per recommendation, wins, ties and losses against always following H-score.
- Compute: GPU-minutes per recommendation for the early fine-tunes at each budget, as in 12.16.

Primary comparison: R1 at epoch 1 against always following H-score, on mean regret. Key secondary:
R2 against R1 at epoch 1 (does H-score add anything once the early fine-tune is run?). Epochs 2 and
3, the best-average rule and the warning-only replication are secondary.

## 5. StudentBERT

R1 and R2 are also applied to the 7 StudentBERT targets with the existing early fine-tuning records
(validation AUC instead of accuracy; delta from the same formula is not defined for AUC, so R2 uses
the gap between the two encoders' validation AUC against 0.001, the benchmark's practical tie
margin). This is retrospective evidence, as before.

## 6. Steps

1. Commit this file before any fresh-seed job is submitted.
2. Extract and score the fresh seeds (about 1 GPU-hour), build the fresh head pairs, then run the
   210 early fine-tunes (about 8 GPU-hours).
3. Write and commit the evaluation before it first reads the fresh early fine-tunes.
4. Run it, check it, and record the result in a new RESULTS.md section.
