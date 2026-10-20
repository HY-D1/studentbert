# Reversal-aware transfer guidance: results so far (frozen 2026-10-20)

This write-up freezes the results of the reversal-aware pipeline before the next phase. It covers
the StudentBERT findings, the NLP validation of the fixed early fine-tuning rule, and the comparison
of fine-tuning budgets (epochs 1 to 3). Every number comes from RESULTS.md sections 12.13 to 12.16;
the record at the end says where the files are kept. Nothing here will be changed; later results go into new
sections.

## 1. The question

H-score is the strongest transferability estimator on StudentBERT (Tasks 1 to 4), but it ranks
candidates from frozen features, and full fine-tuning sometimes overturns its choice. The pipeline
asks whether we can tell, cheaply and in advance, when H-score's recommendation will be overturned.
A recommendation is *reversed* when another candidate beats H-score's pick after fine-tuning by more
than the benchmark's tie rule allows. *Selective regret* is the regret of the recommendations a
method chooses to follow; *coverage* is how many it follows.

## 2. StudentBERT findings (7 KT targets, 21 recommendations)

Seven of the 21 recommendations are reversed. They come from three events only: ASSISTments 2017
(all three estimator seeds), Algebra 2006 (all three) and Algebra 2005 (one seed). Always following
H-score gives a mean regret of 0.0012 AUC. Evaluation is leave-one-target-out throughout.

| Method | Followed | Reversals caught | False alarms | Selective regret (AUC) |
|---|---|---|---|---|
| Always follow H-score | 21/21 | 0/7 | 0 | 0.0012 |
| Version 1: all pairs (12.13) | 16/21 | 1/7 | 4 | 0.0015 |
| Version 2: H-score's pick against each alternative (12.14) | 9/21 | 0/7 | 12 | 0.0027 |
| Fixed epoch-3 rule, retrospective (step B) | 12/21 | 6/7 | 3 | 0.0009 |

- Version 1 predicts reversals between pairs well (pair AUC 0.83), but that does not carry over to
  the recommendation.
- Version 2 aligned the target with the recommendation, as asked, with its primary comparison fixed
  before the run. Its held-out risks ran against the outcome (AUC 0.12 for detecting reversed
  recommendations), and the penalty sat at the top of its grid in 6 of 7 folds: each held-out
  reversal event had to be predicted from the other two, and they did not carry a signal that
  transfers.
- The fixed rule (follow H-score unless its pick is not the validation leader at epoch 3 of a
  short fine-tune; otherwise abstain) catches 6 of 7, missing Algebra 2005, where the overtaken pick
  leads early. Because the rule was suggested by an earlier diagnostic on these same targets, this
  row is retrospective evidence only.

## 3. NLP validation (Bassignana et al. benchmark, 30 recommendations)

**Set-up.** Seven encoders, five tasks (AG News, MNLI, QNLI, RTE, Airline), two poolings and three
estimator seeds: 30 H-score recommendations, labelled against the published fine-tuned results with
the Task 4 tie rule. Twenty are reversed, from four of the five tasks; Airline has none and is the
control. Each encoder was fine-tuned for 3 epochs on the estimators' own 10,000-example sample (210
runs, 7.6 GPU-hours), with validation accuracy recorded after every epoch. Evaluation holds out one
task at a time, with both poolings together.

**Why it counts as a prospective test of the rule.** The rule was fixed on Oct 13, before any NLP
early fine-tune existed. The design (rule, comparison, outcomes) was written before the 5-run
check and committed before any result of the full set was read; the commit came after the runs had
started, which the design file and RESULTS.md record. The check ran one encoder per task, so it
could not show the rule's outcome. Nothing was tuned after the results.

| Method (epoch 3) | Followed | Reversals caught | False alarms | Selective regret (points) |
|---|---|---|---|---|
| Always follow H-score | 30/30 | 0/20 | 0/10 | 2.70 [1.11, 4.15] |
| Fixed epoch-3 rule | 8/30 | 18/20 [70%, 97%] | 4/10 [17%, 69%] | 0.54 [0.00, 2.15] |
| Learned head-pair model | 24/30 | 3/20 [5%, 36%] | 3/10 [11%, 60%] | 2.51 [0.46, 4.49] |

Intervals are Wilson 95% for counts and a task-cluster bootstrap for regret; with five tasks they
are wide.

- **The rule** catches 18 of 20 reversed recommendations. It misses AG News with mean pooling on two
  seeds (2.15 points each); its false alarms are RTE with first-token pooling on all three seeds and
  Airline once. Its simple calibration is good: 82% of the recommendations it flags are reversed,
  against 25% of those it follows. Without Airline, it follows 3 of 24 with selective regret 1.43
  points against 3.37.
- **The learned head-pair model** fails as on StudentBERT. At its held-out thresholds it abstains
  only on RTE, it makes the same decisions at every budget, and its risks are poorly calibrated:
  five recommendations given a risk near 0.04 were all reversed.
- **References.** The best-average rule of Task 4, which never abstains, has regret 1.37 points.
  H-score's own margin to its runner-up orders the recommendations worse than chance (AURC 3.71
  against 2.70).

## 4. Fine-tuning budget: epochs 1 to 3 (NLP)

| Epochs | Caught | False alarms | Followed | Selective regret (points) | GPU-minutes per recommendation | Share of one full-data epoch |
|---|---|---|---|---|---|---|
| 1 | 19/20 | 4/10 | 7/30 | 0.31 | 5.1 | 7% |
| 2 | 18/20 | 7/10 | 5/30 | 2.76 | 10.2 | 14% |
| 3 | 18/20 | 4/10 | 8/30 | 0.54 | 15.2 | 21% |

- Epoch 1 makes the same decision as epoch 3 on 29 of 30 recommendations, at a third of the cost.
  Epoch 2 agrees with epoch 3 on only 23, so the budgets are not ordered by the counts alone.
- The cost of full fine-tuning was bounded rather than run: one epoch over each task's full training
  split costs at least 72.5 GPU-minutes per recommendation (7 encoders), and the authors' tuned
  setting allows up to 50 epochs with early stopping, so the real saving is larger than the shares
  above.

## 5. What the results support

1. A fixed check, whether H-score's pick leads after a short fine-tune, is a strong warning signal:
   it caught 18 of 20 reversed recommendations on data that did not exist when it was fixed, and 6
   of 7 retrospectively on StudentBERT.
2. One epoch appears sufficient for that signal on the NLP benchmark.
3. Learned reversal models do not beat the fixed check when reversal events are few: 3 events on
   StudentBERT, 4 tasks on NLP.
4. The signal alone is not a decision rule: it abstains on 22 of 30 NLP recommendations, and 4 of
   10 correct recommendations are flagged.

## 6. Limitations

- Small samples: 21 StudentBERT and 30 NLP recommendations, from 3 and 4 reversal events.
- The StudentBERT evidence for the rule is retrospective.
- The epoch 1 and 2 results are the secondary analysis named in the design; epoch 3 is the
  pre-specified primary.
- The early fine-tunes use one learning rate and batch size (the authors' tuned setting) and a
  10,000-example sample; other settings were not tried.
- The full fine-tuning cost is a lower bound, not a measurement.

## 7. Next phase (agreed 2026-10-20)

Turn the warning signal into a complete decision rule: keep H-score's pick when it leads after the
short fine-tune, and otherwise take the early fine-tuning leader. Evaluate it on fresh estimator
seeds, with the design, the budget and the comparison fixed before any fresh result exists.

## Record

The numbers above are RESULTS.md sections 12.13 (version 1), 12.14 (version 2), 12.15 (NLP test)
and 12.16 (intervals, calibration, budgets and compute). The output files behind them, and the 210
early fine-tune records, are archived together with this document; a checksum manifest is committed
next to it, and the repository is tagged at this point so the frozen state can be recovered exactly.
