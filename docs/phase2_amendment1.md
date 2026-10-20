# Phase 2 amendment 1: Prof. Hazra's phase 2 task list (2026-10-20)

Prof. Hazra set the phase 2 tasks after the design (phase2_complete_rule_design.md) was committed.
This amendment is committed after the 210 fresh-seed early fine-tunes were submitted and before any
of them is read. The design file is unchanged; where the two differ, this file takes precedence.
The fresh H-score picks and labels were built before this amendment (18 of 30 fresh
recommendations reversed, from 4 of 5 tasks); they involve no fine-tuning result.

## 1. The full framework

Her task list asks for the full decision rule (H-score, then a one-epoch check, then keep H-score's
pick or switch to the early leader) and compares it with one-epoch fine-tuning alone. With a strict
check (switch whenever the pick is not the leader), the two are the same method, because both end
at the early leader in every case. So:

- **Full framework (primary):** H-score's pick, then one epoch of fine-tuning for the 7 encoders;
  switch to the epoch-1 leader only when its validation accuracy exceeds the pick's by more than
  the validation noise (R2 of the design, delta = 1.96 x sqrt(2 p (1 - p) / n)); otherwise keep
  H-score's pick.
- **One-epoch fine-tuning alone (the ablation without H-score):** take the epoch-1 leader (R1 of
  the design), ties broken by H-score.

## 2. Methods compared on the 30 fresh recommendations

| Method | Role in her list |
|---|---|
| H-score alone | baseline; the ablation without the early signal |
| LogME alone (its own top pick) | baseline |
| One-epoch fine-tuning alone (R1) | baseline; the ablation without H-score |
| H-score + abstention (follow H-score unless the one-epoch check flags it) | baseline; abstention against correction |
| Full framework (R2) | the method |
| Best-average rule of Task 4 | reference |

Every method except H-score + abstention chooses an encoder for all 30 recommendations. Epochs 1, 2
and 3 are evaluated for the full framework, one-epoch fine-tuning alone and H-score + abstention
(the 1 against 2 against 3 epochs ablation).

## 3. Outcomes

- Top-1: the choice is the published best. Tied with the best under the Task 4 tie rule.
- Mean regret in published fine-tuned points, with a task-cluster bootstrap interval.
- Reversal recovery: of the recommendations where H-score's pick is overtaken, the share where the
  method's choice is tied with the best.
- False corrections: of the recommendations where H-score's pick is not overtaken, the share where
  the method switches to an encoder decisively worse than the pick under the tie rule. Switches to
  an encoder tied with the pick are counted separately as unnecessary but harmless.
- Coverage and calibration of H-score + abstention: recommendations followed, and the reversal rate
  among flagged and among followed recommendations.
- Stability: for each of the 10 settings, whether the method picks the same encoder on all 3 fresh
  seeds, and the spread of its regret across seeds.
- Compute: GPU-minutes per recommendation for the early fine-tunes at each budget, against the
  full fine-tuning lower bound of 12.16.

Primary comparison: the full framework at epoch 1 against H-score alone, on mean regret. Key
secondary comparisons: the full framework against one-epoch fine-tuning alone (does H-score add
anything?) and against H-score + abstention (correction against abstention), and epochs 1 to 3.
All results are also reported for AG News, MNLI and QNLI alone, the tasks with new samples.

## 4. Failures and near-ties

Every fresh recommendation where the full framework's choice is not tied with the best is listed
with its task, pooling, seed, H-score's pick, the epoch-1 leader, their epoch-1 gap and their
published gap. Separately: how often the framework switches between two encoders that the published
results cannot separate (a near-tie), which changes nothing in practice.

## 5. Other items of her list

- Freezing the current StudentBERT and NLP results in RESULTS.md, including the one-epoch rule:
  recorded in RESULTS.md 12.17 before the phase 2 results.
- StudentBERT: the full framework and one-epoch fine-tuning alone are applied retrospectively to
  the 7 targets, as in the design.
- A focused literature check for directly related methods and baselines (transferability
  estimation, model selection with short fine-tuning, successive halving and learning-curve
  extrapolation) is done separately, for the journal paper.

## 6. A finding that bears on the one-epoch budget (StudentBERT, retrospective)

Checked while writing this amendment, from the existing StudentBERT records: after one epoch the
check catches only 1 of the 7 reversed StudentBERT recommendations (17 of 21 followed, 3 false
alarms), against 6 of 7 after three epochs. The in-domain picks of ASSISTments 2017 and Algebra 2006
lead after one epoch on every seed and are overtaken later. So one epoch is sufficient on the NLP
benchmark (12.16) but not on StudentBERT, and the budget may depend on the domain. The fresh NLP
test keeps epoch 1 as its primary budget, as set; epochs 1, 2 and 3 are reported with equal
prominence for every method; and the conclusion on the budget will be stated per domain. This
result is recorded in RESULTS.md 12.17.

