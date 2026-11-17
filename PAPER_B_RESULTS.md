# Paper B: reliable transfer guidance (estimators, warning signal, fresh-seed validation)

Generated from RESULTS.md (md5 eeeb20becddcab752782e7f64e505fe5) by tools/split_results_by_paper.py. Do not edit: RESULTS.md stays the record, and every section below is copied from it unchanged. Sections, with the md5 of their text:

- 12.3: a990d4030a606f0669466ec1b8ceb3c4
- 12.4: 9c1bf912b795dbb1c64da6500d423ae1
- 12.5: 24111b7ea0383395d7c906bb2780db46
- 12.6: 440a9b1835a44f2486fb8ae07aa83ae9
- 12.7: b7233a5471120b8e692b66ab3c782c75
- 12.8: 8a6add40be551df0effca0961fefddec
- 12.9: 98f98c9786934378fc8ec730384444a6
- 12.10: 4c84cfa67e4a63d9c8b9b7326bf8a4e4
- 12.11: 739dd9d539ed35f074582cce0b85f6eb
- 12.12: ca439dbf3de60c8ff48a9ad1fb8c7721
- 12.13: 34dbf5effaea0c61fe7e34a9d9699441
- 12.14: 4e4a013e55831003f86033795d82510f
- 12.15: a30942c60bbb418813e7b2e53c1fe022
- 12.16: 9d110d55fea3e0fc82540026fc0f37bc
- 12.17: 8eae065b957bd2524b473bb8f6557d7a
- 12.18: b592641247dec5199fd3444364ff5f64
- 12.19: 0765011ab405c3a9a7dfd0eecd69f5fb
- 12.20: 5aa1046402a2904e5165dc5fc800ebce
- 12.21: 35befca5abd759e40fc16884fb8f2600
- 13: d6160eb467f0df7fd498b04c329f4ca4
- 13.1: d7da7737332501c1243c1c91d8ca69c0

---

### 12.3 Transferability estimators, KT, the seven N=3000 cells (pretrained view; 7 targets x 3 estimator seeds; the practical view, with scratch as an option, gives the same ordering)

| estimator | rankings | rho | top-1 | tied with best | regret mean / max | negative picks | own-encoder picks |
|---|---|---|---|---|---|---|---|
| hscore_kt_causal | 21 | +0.5221 | 11/21 | 14/21 | 0.0012 / 0.0103 | 0 | 15 |
| logme_kt_causal | 21 | +0.4507 | 10/21 | 13/21 | 0.0017 / 0.0103 | 0 | 14 |
| largest source | 7 | +0.4031 | 3/7 | 3/7 | 0.0038 / 0.0103 | 0 | 1 |
| fewshot_ft_fit200_e5 | 21 | +0.3027 | 5/21 | 8/21 | 0.0036 / 0.0103 | 3 | 12 |
| fewshot_ft_fit50_e5 | 21 | +0.2211 | 3/21 | 4/21 | 0.0038 / 0.0098 | 3 | 13 |
| logme_kt_causal_per_skill | 21 | +0.1003 | 2/21 | 5/21 | 0.0077 / 0.0278 | 5 | 8 |
| nleep_k4_kt_causal | 21 | -0.1837 | 1/21 | 2/21 | 0.0090 / 0.0278 | 6 | 1 |
| nleep_k8_kt_causal | 21 | -0.1531 | 2/21 | 2/21 | 0.0078 / 0.0278 | 4 | 0 |
| nleep_k16_kt_causal | 21 | -0.3197 | 2/21 | 2/21 | 0.0080 / 0.0278 | 2 | 0 |

hscore_shrunk makes the same pick as hscore in all 21 rankings (top-1 11/21, tied with best 14/21,
regret 0.0012 / 0.0103) at rho +0.5238; the largest-source rule has one ranking per target. Scores
use the leakage-safe causal features of src/estimators/features.py on the fine-tune's learner draw
(3,000 learners on EdNet and Junyi, the whole train split on the five smaller targets), capped at
50,000 positions.

_Read: H-score is the strongest estimator, narrowly ahead of LogME. Both beat the largest-source
rule on regret and rank correlation and never pick a negative-transfer source. NLEEP ranks the
candidates backwards, and a short fine-tuning proxy (5 epochs on 50 or 200 learners) is weak and
unstable._

### 12.4 The repeated failure: preference for the target's own encoder

H-score picks the target's own encoder in 15 of 21 rankings, LogME in 14 and the few-shot proxy in
12. That is right on EdNet and ASSISTments 2009, where in-domain is the best source (on ASSISTments
2009 Junyi and scratch are tied with it), not an error on Bridge 2006, where in-domain is tied with
Junyi, the best, and wrong for all three estimators on ASSISTments 2017 (EdNet best, also across
evaluation learners, 12.12) and Algebra 2006 (Junyi best). Ruled out as causes by the LogME
diagnostics (scripts/diagnose_logme.py, three
original targets): the trained skill table, since re-scoring the in-domain encoder with its skill
table at the random start leaves the ASSISTments 2017 pick unchanged on 3 of 3 seeds; and layer
choice, with rank correlation by layer L0 +0.20, L1 -0.40, L2 -0.40, L3 +0.04, L4 +0.24, L5 +0.62,
L6 +0.42, where L5 still picks in-domain on ASSISTments 2017. On Algebra 2006 the same diagnostics
point the other way (12.9). Per-skill LogME fails at every layer.

### 12.5 Budget hypothesis for that failure (not supported)

Hypothesis: the estimators see little target data, the in-domain encoder leads at small budgets,
and a larger foreign source overtakes it by N=3000. The KT budget sweep on ASSISTments 2017 (full
fine-tuning; N=25 to 1000 from the W3/W4 curves, 3 seeds, batch 32; N=3000 from the pinned grid,
6 seeds; gains against scratch, with the count of seeds in the direction of the mean):

| budget | in-domain | EdNet | Junyi | top-equivalent |
|---|---|---|---|---|
| 25 | +0.0103 (2/3) | +0.0149 (2/3) | -0.0012 (1/3) | ednet, assist2017, scratch |
| 50 | -0.0095 (3/3) | -0.0028 (3/3) | -0.0149 (3/3) | scratch |
| 100 | +0.0002 (1/3) | +0.0048 (3/3) | -0.0035 (3/3) | ednet |
| 200 | +0.0023 (2/3) | +0.0125 (3/3) | +0.0041 (2/3) | ednet |
| 500 | +0.0121 (3/3) | +0.0236 (3/3) | +0.0122 (3/3) | ednet |
| 1000 | +0.0205 (3/3) | +0.0269 (3/3) | +0.0170 (3/3) | ednet |
| 3000 | +0.0231 (6/6) | +0.0261 (6/6) | +0.0214 (6/6) | ednet |

_Read: EdNet leads at every budget from N=100 to 3,000 and the in-domain encoder never leads, so
scarce target data does not favour the in-domain encoder. The few-shot proxy still ranks it first
on both failure targets at seed 42, by small margins (ASSISTments 2017 at 200 learners, 0.6288
against 0.6270 for EdNet), which points to its short training rather than to data size. One
mechanism remains open and testable: every estimator samples target learners from the train split
(src/estimators/features.py), the in-domain encoder's own pretraining partition, while the gold is
measured on unseen test learners. Scoring the estimators on validation learners would test it._

### 12.6 Exposure test for that failure (validation learners against a size-matched train draw; 3 estimator seeds; exposure_v1/)

Both frozen-feature scorers re-run on the 7 x 7 grid with two more target samples: the
validation learners (--split val), which no encoder saw (pretraining reads the train split and
keeps its checkpoint by train loss), and a train draw with as many learners (--match_split val).
Same candidates, seeds and 50,000-position cap. Code 6ed04fe, with the decision rule fixed in
analysis/exposure_report.py before any validation score existed; walltimes 6f8c756. 28 jobs,
Slurm 10590371 to 10594649 (the Junyi validation Task 2 job 10591596 stalled on node d1020, was
cancelled after 1:03:23 with no output, and ran again as 10594649). Margin: in-domain score minus
the best other pretrained score, in each estimator's own units. Preferred: strictly top on at
least 2 of 3 seeds. Verdict: exposure if preferred on the matched train draw and not on
validation learners, representation if preferred on both, no preference if not preferred on the
train draw. Where the matched draw is the train draw (EdNet, Junyi) it reproduces the train-draw
scores exactly.

Files (md5): exposure_summary.tsv adfd97ebdd98d70c618e37ef385bd0f9,
estimators_summary.tsv 79715b91069b33d6bdbd7d88b5a0bdef,
estimators_cells.tsv 0801251bb479829d7a461fb1446923cf,
exposure_report.md 30c2dfd03a9c59cea6d5f6cf104e659d.

| estimator | target | learners train / matched / val | preferred train / matched / val | margin train / matched / val | verdict |
|---|---|---|---|---|---|
| hscore_kt_causal | assist2017 | 1366 / 170 / 170 | 3/3 / 3/3 / 1/3 | +0.004011 / +0.004526 / +0.000132 | exposure |
| hscore_kt_causal | ednet | 3000 / 3000 / 3000 | 3/3 / 3/3 / 3/3 | +0.004880 / +0.004880 / +0.006801 | representation |
| hscore_kt_causal | junyi | 3000 / 3000 / 3000 | 0/3 / 0/3 / 0/3 | -0.007527 / -0.007527 / -0.005422 | no preference |
| hscore_kt_causal | algebra2005 | 453 / 56 / 56 | 0/3 / 1/3 / 1/3 | -0.004664 / -0.001162 / -0.002097 | no preference |
| hscore_kt_causal | bridge2006 | 904 / 113 / 113 | 3/3 / 3/3 / 1/3 | +0.001512 / +0.002433 / -0.001275 | exposure |
| hscore_kt_causal | assist2009 | 2495 / 311 / 311 | 3/3 / 3/3 / 3/3 | +0.013193 / +0.014235 / +0.016371 | representation |
| hscore_kt_causal | algebra2006 | 1048 / 131 / 131 | 3/3 / 3/3 / 3/3 | +0.002125 / +0.003730 / +0.006013 | representation |
| logme_kt_causal | assist2017 | 1366 / 170 / 170 | 3/3 / 3/3 / 1/3 | +0.000939 / +0.001184 / -0.000965 | exposure |
| logme_kt_causal | ednet | 3000 / 3000 / 3000 | 2/3 / 2/3 / 3/3 | +0.000677 / +0.000677 / +0.001533 | representation |
| logme_kt_causal | junyi | 3000 / 3000 / 3000 | 0/3 / 0/3 / 0/3 | -0.008027 / -0.008027 / -0.007034 | no preference |
| logme_kt_causal | algebra2005 | 453 / 56 / 56 | 0/3 / 2/3 / 1/3 | -0.001788 / +0.000672 / -0.000900 | no preference |
| logme_kt_causal | bridge2006 | 904 / 113 / 113 | 3/3 / 3/3 / 2/3 | +0.001200 / +0.001844 / -0.000039 | representation |
| logme_kt_causal | assist2009 | 2495 / 311 / 311 | 3/3 / 3/3 / 3/3 | +0.008098 / +0.007405 / +0.010668 | representation |
| logme_kt_causal | algebra2006 | 1048 / 131 / 131 | 3/3 / 3/3 / 3/3 | +0.001744 / +0.002315 / +0.003353 | representation |

Estimator quality by target sample (the seven N=3000 cells, pretrained view, 21 rankings each):

| estimator | sample | rho | top-1 | tied with best | regret mean / max |
|---|---|---|---|---|---|
| hscore_kt_causal | train draw | +0.5221 | 11/21 | 14/21 | 0.0012 / 0.0103 |
| hscore_kt_causal | matched train draw | +0.4711 | 10/21 | 13/21 | 0.0015 / 0.0103 |
| hscore_kt_causal | validation learners | +0.5051 | 12/21 | 13/21 | 0.0019 / 0.0103 |
| logme_kt_causal | train draw | +0.4507 | 10/21 | 13/21 | 0.0017 / 0.0103 |
| logme_kt_causal | matched train draw | +0.4082 | 8/21 | 11/21 | 0.0021 / 0.0092 |
| logme_kt_causal | validation learners | +0.4473 | 11/21 | 13/21 | 0.0019 / 0.0103 |

Picks on the two failure targets and on Bridge 2006 (seeds 42, 1, 2; regret against the best source):

| target | estimator | sample | picks | mean regret |
|---|---|---|---|---|
| assist2017 | hscore_kt_causal | train draw | assist2017, assist2017, assist2017 | 0.0030 |
| assist2017 | hscore_kt_causal | matched train draw | assist2017, assist2017, assist2017 | 0.0030 |
| assist2017 | hscore_kt_causal | validation learners | assist2017, ednet, ednet | 0.0010 |
| assist2017 | logme_kt_causal | train draw | assist2017, assist2017, assist2017 | 0.0030 |
| assist2017 | logme_kt_causal | matched train draw | assist2017, assist2017, assist2017 | 0.0030 |
| assist2017 | logme_kt_causal | validation learners | assist2017, ednet, ednet | 0.0010 |
| algebra2006 | hscore_kt_causal | train draw | algebra2006, algebra2006, algebra2006 | 0.0017 |
| algebra2006 | hscore_kt_causal | matched train draw | algebra2006, algebra2006, algebra2006 | 0.0017 |
| algebra2006 | hscore_kt_causal | validation learners | algebra2006, algebra2006, algebra2006 | 0.0017 |
| algebra2006 | logme_kt_causal | train draw | algebra2006, algebra2006, algebra2006 | 0.0017 |
| algebra2006 | logme_kt_causal | matched train draw | algebra2006, algebra2006, algebra2006 | 0.0017 |
| algebra2006 | logme_kt_causal | validation learners | algebra2006, algebra2006, algebra2006 | 0.0017 |
| bridge2006 | hscore_kt_causal | train draw | bridge2006, bridge2006, bridge2006 | 0.0005 |
| bridge2006 | hscore_kt_causal | matched train draw | bridge2006, bridge2006, bridge2006 | 0.0005 |
| bridge2006 | hscore_kt_causal | validation learners | junyi, junyi, bridge2006 | 0.0002 |
| bridge2006 | logme_kt_causal | train draw | bridge2006, bridge2006, bridge2006 | 0.0005 |
| bridge2006 | logme_kt_causal | matched train draw | bridge2006, bridge2006, bridge2006 | 0.0005 |
| bridge2006 | logme_kt_causal | validation learners | bridge2006, junyi, bridge2006 | 0.0004 |

_Read: pretraining exposure explains the in-domain preference on ASSISTments 2017 and not on
Algebra 2006. On ASSISTments 2017 both estimators prefer the in-domain encoder on 3 of 3 seeds of
the matched train draw and on 1 of 3 with validation learners, where the other two seeds pick
EdNet, the best source. On Algebra 2006 the preference holds on every seed of every sample and its
margin grows on unseen learners, so it is not exposure; Junyi is the only top-equivalent source
across fine-tuning seeds (12.1) but not across evaluation learners (12.8, 12.11). On
Bridge 2006 the estimators disagree (exposure for
H-score, representation for LogME), but every pick there is tied with the best. Scoring
validation learners is not a fix: on the five smaller targets a validation split holds an eighth
as many learners as the train split, and in aggregate both estimators lose rank correlation and
gain regret; at matched size, unseen learners give the higher rank correlation and top-1. On
those five targets the train draw and the validation sample hold every learner of their split, so
there the three seeds differ only in the position subsample and the random skill tables; the
matched train draw is a different eighth of the train split at each seed._

### 12.7 Confidence signals, the foreign-source view and two simple rules (the seven N=3000 cells; analysis/estimator_signals.py, analysis/similarity_baseline.py, analysis/best_average_source.py)

Signals from analysis/estimator_signals.py over the 3 estimator seeds of each target: the modal
pick, the seeds that agree with it, and the margin of the pick over the runner-up divided by the
per-candidate seed SD, as that script writes it (6 dp). Mean regret is recomputed from
estimators_cells.tsv. The foreign-source view, from analysis/similarity_baseline.py, leaves the
target's own encoder out and judges every choice against the best foreign source. The best-average
rule (analysis/best_average_source.py, commit 5471edc) scores each source by minus its mean rank as
a foreign source on the other six targets, so no target's results enter its own scores; it re-judges
the largest-source rule as a check, which reproduces 12.3 exactly.

Files (md5): estimators_summary.tsv a82f6da1cd995a4d13fd4c6a825969a6, estimators_cells.tsv
3a69311ffa9b0d90befdb7287f4ce632, signals_main.tsv a3f4c0d80490904618a69ede169f5858,
similarity_summary.tsv 14ef5a915ca203eed44649da767ec66d, similarity_cells.tsv
f1ca3ab393d465df3dab7ef8c6d28ddd, best_average_cells.tsv 37a2d6d02d2232da2d06b73248b19596,
best_average_report.md 028f88245621a2bf83da55b9f346c4c9.

| estimator | target | modal pick | agreement | margin / seed SD | mean regret |
|---|---|---|---|---|---|
| hscore_kt_causal | assist2017 | assist2017 | 3/3 | 2.898999 | 0.0030 |
| hscore_kt_causal | ednet | ednet | 3/3 | 2.328283 | 0.0000 |
| hscore_kt_causal | junyi | ednet | 3/3 | 0.352872 | 0.0000 |
| hscore_kt_causal | algebra2005 | junyi | 2/3 | 1.442697 | 0.0034 |
| hscore_kt_causal | bridge2006 | bridge2006 | 3/3 | 0.768078 | 0.0005 |
| hscore_kt_causal | assist2009 | assist2009 | 3/3 | 3.045132 | 0.0000 |
| hscore_kt_causal | algebra2006 | algebra2006 | 3/3 | 0.748992 | 0.0017 |
| logme_kt_causal | assist2017 | assist2017 | 3/3 | 1.003681 | 0.0030 |
| logme_kt_causal | ednet | ednet | 2/3 | 0.784816 | 0.0031 |
| logme_kt_causal | junyi | ednet | 3/3 | 0.202392 | 0.0000 |
| logme_kt_causal | algebra2005 | junyi | 2/3 | 1.800205 | 0.0034 |
| logme_kt_causal | bridge2006 | bridge2006 | 3/3 | 0.276234 | 0.0005 |
| logme_kt_causal | assist2009 | assist2009 | 3/3 | 2.590958 | 0.0000 |
| logme_kt_causal | algebra2006 | algebra2006 | 3/3 | 1.105673 | 0.0017 |

Foreign-source view (the target's own encoder out; 7 targets):

| estimator | rankings | rho | top-1 | tied with best | regret mean / max | negative picks |
|---|---|---|---|---|---|---|
| best average (leave one target out) | 7 | +0.4699 | 5/7 | 5/7 | 0.0020 / 0.0093 | 1 |
| largest source | 7 | +0.3959 | 3/7 | 3/7 | 0.0037 / 0.0103 | 0 |
| hscore_shrunk_kt_causal | 21 | +0.3660 | 14/21 | 14/21 | 0.0024 / 0.0233 | 0 |
| hscore_kt_causal | 21 | +0.3578 | 14/21 | 14/21 | 0.0024 / 0.0233 | 0 |
| logme_kt_causal | 21 | +0.3442 | 13/21 | 13/21 | 0.0028 / 0.0233 | 0 |
| fewshot_ft_fit200_e5 | 21 | +0.1864 | 11/21 | 11/21 | 0.0028 / 0.0103 | 4 |
| similarity_labels | 7 | +0.1429 | 3/7 | 3/7 | 0.0058 / 0.0278 | 1 |
| similarity_inputs | 7 | +0.1184 | 3/7 | 3/7 | 0.0062 / 0.0278 | 1 |
| fewshot_ft_fit50_e5 | 21 | +0.0721 | 8/21 | 8/21 | 0.0048 / 0.0229 | 5 |
| logme_kt_causal_per_skill | 21 | +0.0503 | 8/21 | 8/21 | 0.0064 / 0.0278 | 6 |
| nleep_k8_kt_causal | 21 | -0.0721 | 2/21 | 3/21 | 0.0071 / 0.0278 | 4 |
| nleep_k4_kt_causal | 21 | -0.1755 | 1/21 | 2/21 | 0.0084 / 0.0278 | 6 |
| nleep_k16_kt_causal | 21 | -0.1891 | 2/21 | 2/21 | 0.0073 / 0.0278 | 2 |

Best-average rule against H-score, pretrained view (regret per target; the rule has one ranking per
target, H-score three):

| target | best-average pick | rule regret | H-score regret |
|---|---|---|---|
| assist2017 | junyi | 0.0047 | 0.0030 |
| ednet | junyi | 0.0045 | 0.0000 |
| junyi | junyi | 0.0023 | 0.0000 |
| algebra2005 | junyi | 0.0000 | 0.0034 |
| bridge2006 | junyi | 0.0000 | 0.0005 |
| assist2009 | junyi | 0.0004 | 0.0000 |
| algebra2006 | junyi | 0.0000 | 0.0017 |

Pretrained view, all targets: best average rho +0.4447, top-1 3/7, tied with best 4/7, regret 0.0017
/ 0.0047, 0 negative picks. Taking the lower of the two regrets on each target, a bound that reads
the gold, gives 0.0004.

_Read: the logged confidence signals do not flag the two wrong picks. On ASSISTments 2017 and
Algebra 2006, H-score and LogME pick the in-domain encoder on 3 of 3 seeds, and the 200-learner
few-shot proxy makes the same pick on at least 2 of 3; H-score's ASSISTments 2017 margin is 2.898999
seed SDs, while its correct Junyi pick has a margin of 0.352872. With the own encoder removed,
H-score picks Junyi on Algebra 2006 on every seed; the largest-source rule orders the foreign
sources better than H-score but H-score picks the winner more often; the dataset-similarity rules
are weak. The best-average rule picks Junyi on every target in the pretrained view: it is right on
Algebra 2006's test learners (12.11), has the highest rank correlation in the foreign-source
view, and trails H-score in the
pretrained view. H-score and the rule lose regret on different targets, and only ASSISTments 2017
defeats both. The foreign-view maximum of H-score, its shrunk form and LogME comes from one ranking,
ASSISTments 2017 at seed 42, where all three pick Bridge 2006._
### 12.8 Fine-tuning trajectories and the validation check (the two failure targets, N=3000, 6 seeds; analysis/finetune_trajectories.py)

Validation AUC per epoch from the fine-tune logs (4 dp, as scripts/finetune_edubert.py prints it);
best validation AUC from the W&B export at full precision, each value checked against its log; test
AUC is the benchmark value. The winner is the best source by mean test AUC (EdNet on ASSISTments
2017, Junyi on Algebra 2006).

Files (md5): trajectories_valtest.tsv 9088a58c14422e0a415d7e276dacf65c, trajectories_epochs.tsv
b493726aeac64d6f2f53c86a93954af7.

| target | winner minus own encoder | mean | seeds positive | per seed |
|---|---|---|---|---|
| assist2017 | best validation AUC | +0.0035 | 6/6 | +0.0008 +0.0033 +0.0037 +0.0069 +0.0015 +0.0048 |
| assist2017 | test AUC | +0.0030 | 6/6 | +0.0025 +0.0015 +0.0022 +0.0047 +0.0021 +0.0049 |
| algebra2006 | best validation AUC | +0.0001 | 3/6 | -0.0014 -0.0001 +0.0006 +0.0007 -0.0012 +0.0020 |
| algebra2006 | test AUC | +0.0017 | 6/6 | +0.0009 +0.0004 +0.0002 +0.0032 +0.0029 +0.0028 |

| target | epoch | leader | winner minus own | seeds winner ahead |
|---|---|---|---|---|
| assist2017 | 1 | assist2017 | -0.0103 | 1/6 |
| assist2017 | 2 | ednet | +0.0039 | 5/6 |
| assist2017 | 3 | ednet | +0.0137 | 6/6 |
| assist2017 | 5 | ednet | +0.0077 | 6/6 |
| assist2017 | 10 | ednet | +0.0029 | 6/6 |
| assist2017 | 15 | ednet | +0.0015 | 5/6 |
| assist2017 | 20 | ednet | +0.0022 | 5/6 |
| algebra2006 | 1 | algebra2006 | -0.0113 | 0/6 |
| algebra2006 | 2 | algebra2006 | -0.0002 | 2/6 |
| algebra2006 | 3 | junyi | +0.0024 | 6/6 |
| algebra2006 | 5 | junyi | +0.0023 | 6/6 |
| algebra2006 | 10 | algebra2006 | -0.0001 | 1/6 |
| algebra2006 | 15 | algebra2006 | -0.0014 | 2/6 |
| algebra2006 | 20 | algebra2006 | -0.0021 | 1/6 |

_Read: at epoch 1 of fine-tuning the own encoder leads on both targets, as the frozen scores say.
EdNet leads ASSISTments 2017 from epoch 2 on, and its validation and test learners agree (6 of 6
seeds on both). On Algebra 2006 Junyi leads the validation learners at epochs 3 to 9 and the own
encoder from epoch 10 on; at the best checkpoints the validation learners tie the two (+0.0001, 3/6
seeds) while the test learners favour Junyi (+0.0017, 6/6)._

### 12.9 Layer and skill-table diagnostic on Algebra 2006 (3 seeds, scratch and 7 encoders; scripts/diagnose_logme.py with H-score, analysis/layer_diag_report.py)

Each ranking of the 7 sources is judged against the fine-tuned gold as in 12.3. 'Skill table reset'
re-scores the in-domain encoder with its skill table at the fine-tune's random start, the table
every foreign encoder is scored with; nothing else changes. The final layer reproduces the 7 x 7
scores (within 6.64e-09 for H-score and 2.94e-09 for LogME).

Files (md5): layer_diag_algebra2006_report.md b40fdea796ce83f5ac146661ea7eafee,
layer_diag_algebra2006_cells.tsv e11f95f387d6eca99d6becf78e36df1b.

H-score:

| layer | variant | picks (seeds 1 2 42) | top-1 | tied with best | mean regret | mean rho |
|---|---|---|---|---|---|---|
| L0 | trained skill table | algebra2006, assist2009, bridge2006 | 0/3 | 0/3 | 0.0048 | -0.0595 |
| L0 | skill table reset | ednet, assist2009, bridge2006 | 0/3 | 0/3 | 0.0065 | -0.2500 |
| L1 | trained skill table | algebra2005, algebra2005, algebra2005 | 0/3 | 0/3 | 0.0102 | +0.0833 |
| L1 | skill table reset | algebra2005, algebra2005, algebra2005 | 0/3 | 0/3 | 0.0102 | +0.0714 |
| L2 | trained skill table | algebra2006, algebra2006, junyi | 1/3 | 1/3 | 0.0012 | +0.5357 |
| L2 | skill table reset | junyi, junyi, junyi | 3/3 | 3/3 | 0.0000 | +0.4405 |
| L3 | trained skill table | junyi, junyi, junyi | 3/3 | 3/3 | 0.0000 | +0.7143 |
| L3 | skill table reset | junyi, junyi, junyi | 3/3 | 3/3 | 0.0000 | +0.5952 |
| L4 | trained skill table | algebra2006, algebra2006, junyi | 1/3 | 1/3 | 0.0012 | +0.7619 |
| L4 | skill table reset | junyi, junyi, junyi | 3/3 | 3/3 | 0.0000 | +0.6310 |
| L5 | trained skill table | algebra2006, algebra2006, junyi | 1/3 | 1/3 | 0.0012 | +0.7381 |
| L5 | skill table reset | junyi, junyi, junyi | 3/3 | 3/3 | 0.0000 | +0.6071 |
| L6 | trained skill table | algebra2006, algebra2006, algebra2006 | 0/3 | 0/3 | 0.0017 | +0.7262 |
| L6 | skill table reset | junyi, junyi, algebra2006 | 2/3 | 2/3 | 0.0006 | +0.5952 |

LogME:

| layer | variant | picks (seeds 1 2 42) | top-1 | tied with best | mean regret | mean rho |
|---|---|---|---|---|---|---|
| L0 | trained skill table | algebra2006, bridge2006, bridge2006 | 0/3 | 0/3 | 0.0042 | +0.0952 |
| L0 | skill table reset | bridge2006, bridge2006, bridge2006 | 0/3 | 0/3 | 0.0055 | -0.0476 |
| L1 | trained skill table | algebra2005, algebra2005, algebra2005 | 0/3 | 0/3 | 0.0102 | +0.0238 |
| L1 | skill table reset | algebra2005, algebra2005, algebra2005 | 0/3 | 0/3 | 0.0102 | +0.0238 |
| L2 | trained skill table | algebra2006, algebra2006, junyi | 1/3 | 1/3 | 0.0012 | +0.4048 |
| L2 | skill table reset | junyi, algebra2005, junyi | 2/3 | 2/3 | 0.0034 | +0.3095 |
| L3 | trained skill table | algebra2006, algebra2006, junyi | 1/3 | 1/3 | 0.0012 | +0.6310 |
| L3 | skill table reset | junyi, junyi, junyi | 3/3 | 3/3 | 0.0000 | +0.5357 |
| L4 | trained skill table | algebra2006, algebra2006, junyi | 1/3 | 1/3 | 0.0012 | +0.7381 |
| L4 | skill table reset | junyi, junyi, junyi | 3/3 | 3/3 | 0.0000 | +0.6429 |
| L5 | trained skill table | algebra2006, algebra2006, junyi | 1/3 | 1/3 | 0.0012 | +0.7143 |
| L5 | skill table reset | junyi, junyi, junyi | 3/3 | 3/3 | 0.0000 | +0.5952 |
| L6 | trained skill table | algebra2006, algebra2006, algebra2006 | 0/3 | 0/3 | 0.0017 | +0.6548 |
| L6 | skill table reset | junyi, assist2017, algebra2006 | 1/3 | 1/3 | 0.0030 | +0.5238 |

_Read: on Algebra 2006, unlike ASSISTments 2017 (12.4), the trained skill table carries much of the
in-domain preference. With it reset, H-score's final layer picks Junyi on 2 of 3 seeds and LogME's
on 1 of 3, and at layers L2 to L5 H-score picks Junyi on every seed, as LogME does from L3. With the
trained table only H-score's L3 picks Junyi on every seed, and the final layer never does. The
frozen view grants a trained skill table to the in-domain encoder only; fine-tuning trains one for
every candidate._

### 12.10 Frozen gold: logistic readouts on the estimators' features, scored on test learners (all 7 targets, 3 seeds; scripts/frozen_gold_kt.py, analysis/frozen_gold_report.py)

A logistic readout of the estimators' own causal features (the same learner draw, position cap,
vocabulary loading and random start), fit on the train draw and scored on a fixed draw of test
learners, gives each candidate's frozen performance. The frozen oracle ranks the candidates by it
and is judged against the fine-tuned gold as in 12.3. The main readout is shared across skills with
an L2 penalty of 1e-4; the report also holds a 1e-2 penalty and a next-skill-bias readout. The
frozen-gold features match the estimators' within 4.3e-09.

Files (md5): frozen_gold_cells.tsv 46008bbac2b0e3fba6b62b7f52359d73, frozen_gold_report.md
10a1661f378626db33ebe8d537fd0f47.

| target | frozen best | own-encoder rank | tuned best | oracle rho | oracle top-1 | oracle regret | H-score rho, frozen / tuned | LogME rho, frozen / tuned |
|---|---|---|---|---|---|---|---|---|
| algebra2005 | src:junyi | 4 | src:junyi | +0.0714 | True | 0.0000 | +0.6548 / +0.3810 | +0.4881 / +0.3571 |
| algebra2006 | src:algebra2006 | 1 | src:junyi | +0.6429 | False | 0.0017 | +0.9524 / +0.7262 | +0.9405 / +0.6548 |
| assist2009 | src:assist2009 | 1 | src:assist2009 | +0.6071 | True | 0.0000 | +0.7738 / +0.4762 | +0.7738 / +0.4524 |
| assist2017 | src:assist2017 | 1 | src:ednet | +0.4286 | False | 0.0030 | +0.8214 / +0.1429 | +0.8214 / +0.1429 |
| bridge2006 | src:junyi | 2 | src:junyi | +0.7500 | True | 0.0000 | +0.7738 / +0.7976 | +0.7738 / +0.7976 |
| ednet | src:ednet | 1 | src:ednet | +0.4286 | True | 0.0000 | +0.8333 / +0.4524 | +0.8095 / +0.4762 |
| junyi | src:ednet | 3 | src:ednet | +0.9286 | True | 0.0000 | +0.8095 / +0.6786 | +0.5119 / +0.2738 |
| mean over 7 | | | | +0.5510 | 5/7 | 0.0007 | +0.8027 / +0.5221 | +0.7313 / +0.4507 |

_Read: knowing frozen performance exactly picks the fine-tuned winner on 5 of 7 targets (rank
correlation +0.5510, mean regret 0.0007, against 0.0012 for H-score) and misses exactly on
ASSISTments 2017 and Algebra 2006, where frozen performance on test learners prefers the own
encoder. H-score and LogME track frozen performance (+0.8027, +0.7313) far better than fine-tuned
performance (+0.5221, +0.4507, the 12.3 values reproduced). The estimators measure frozen
performance well; frozen performance is an imperfect guide to fine-tuned performance._

### 12.11 Learner-level bootstrap of the Algebra 2006 validation and test splits (in-domain against Junyi, 6 seeds; scripts/split_bootstrap_kt.py)

The 12 saved best checkpoints are re-scored on both splits as the fine-tune scores its test set;
every one reproduces its recorded test AUC within 1.7e-08. Learners are resampled 2,000 times (seed
0), the same resample for every checkpoint; the statistic is the mean over seeds of Junyi's AUC
minus the in-domain encoder's. The two splits hold disjoint learners and are resampled
independently.

Files (md5): split_bootstrap_algebra2006_summary.tsv 65d86026e7623b8887487935c7a5fcdd,
split_bootstrap_algebra2006_checks.json f07e604fc0bea2225f144f66dbde0e76.

| split | learners | positions | Junyi minus in-domain | 95% interval | P(>0) | seeds positive |
|---|---|---|---|---|---|---|
| val | 131 | 61866 | +0.0001 | [-0.0019, +0.0022] | 0.5255 | 3/6 |
| test | 131 | 58800 | +0.0017 | [-0.0001, +0.0035] | 0.9705 | 6/6 |
| test minus val |  |  | +0.0016 | [-0.0010, +0.0044] | 0.8835 |  |

_Read: once learners are resampled, neither split separates Junyi from the in-domain encoder: the
test interval includes zero, and the gap between the splits is within learner sampling. The Algebra
2006 difference of 12.1 holds across fine-tuning seeds, whose intervals the gold uses, but not
across evaluation learners, which the gold does not resample._
### 12.12 Learner-level bootstrap of the other two regret-bearing pairs (ASSISTments 2017 in-domain against EdNet, Algebra 2005 EdNet against Junyi, 6 seeds; scripts/split_bootstrap_kt.py), and H-score's regret by evidence status

The 12.11 design on the two other pairs where H-score is charged regret on a small target. On
ASSISTments 2017, a is the in-domain encoder and b is EdNet; on Algebra 2005, a is EdNet, H-score's
pick on its one wrong seed, and b is Junyi. The 24 saved best checkpoints reproduce their recorded
test AUC within 5.7e-08; 2,000 learner resamples (seed 0), the same for every checkpoint; the
statistic is the mean over seeds of b minus a. Each checkpoint was selected on the validation
learners, so the test interval is the clean one.

Files (md5): split_bootstrap_assist2017_summary.tsv 854f586525daaa32b90af656c357912b,
split_bootstrap_assist2017_checks.json 6fb893cae0c03e73d3b51664e0cde348,
split_bootstrap_algebra2005_summary.tsv 5570b3f70342d562607f4995f0a99b3c,
split_bootstrap_algebra2005_checks.json 761628948e0c744f4e65fa32b4c1e401, signals_main.tsv
a3f4c0d80490904618a69ede169f5858.

| target | b minus a | split | learners | positions | difference | 95% interval | P(>0) | seeds positive |
|---|---|---|---|---|---|---|---|---|
| assist2017 | EdNet minus in-domain | val | 170 | 60757 | +0.0035 | [+0.0010, +0.0061] | 0.9945 | 6/6 |
| assist2017 | EdNet minus in-domain | test | 172 | 63066 | +0.0030 | [+0.0003, +0.0057] | 0.9870 | 6/6 |
| assist2017 | EdNet minus in-domain | test minus val |  |  | -0.0005 | [-0.0043, +0.0032] | 0.3900 |  |
| algebra2005 | Junyi minus EdNet | val | 56 | 20936 | +0.0103 | [+0.0066, +0.0141] | 1.0000 | 6/6 |
| algebra2005 | Junyi minus EdNet | test | 58 | 22238 | +0.0103 | [+0.0075, +0.0132] | 1.0000 | 6/6 |
| algebra2005 | Junyi minus EdNet | test minus val |  |  | +0.0000 | [-0.0047, +0.0047] | 0.4910 |  |

H-score's mean regret over the 21 rankings of 12.3 (0.0012) by target: mean regret over its 3 seeds
as benchmark_final/signals_main.tsv writes it, and that value over 7.

| target | H-score pick, seeds | mean regret | share of the 21-ranking mean | evidence |
|---|---|---|---|---|
| assist2017 | assist2017, 3/3 | 0.002971 | 0.0004 | robust: across fine-tuning seeds (12.1) and evaluation learners on both splits (this section) |
| algebra2005 | junyi, 2/3 | 0.003445 | 0.0005 | gold robust across learners (this section); the miss is one estimator seed of 3 |
| algebra2006 | algebra2006, 3/3 | 0.001727 | 0.0002 | not separable across evaluation learners (12.11) |
| bridge2006 | bridge2006, 3/3 | 0.000527 | 0.0001 | the pick is tied with the best (12.1) |
| assist2009 | assist2009, 3/3 | 0.000000 | 0.0000 | no regret |
| ednet | ednet, 3/3 | 0.000000 | 0.0000 | no regret |
| junyi | ednet, 3/3 | 0.000000 | 0.0000 | no regret |
| all 7 | | | 0.0012 | |

_Read: on ASSISTments 2017 EdNet beats the in-domain encoder across evaluation learners on both
splits, +0.0035 [+0.0010, +0.0061] on validation and +0.0030 [+0.0003, +0.0057] on test, 6 of 6
seeds each, and the test gap is above the 0.001 margin, so the two are not top-equivalent at learner
level either; the re-scored differences equal 12.8 seed by seed. On Algebra 2005 Junyi beats EdNet
on both splits, +0.0103 [+0.0066, +0.0141] and +0.0103 [+0.0075, +0.0132], so H-score's one wrong
seed there is an estimator error (EdNet, last of 7 there in 12.1), not gold noise. Of H-score's
0.0012, 0.0004 comes from ASSISTments 2017, 0.0005 from the Algebra 2005 seed, 0.0002 from Algebra
2006, which the evaluation learners do not separate, and 0.0001 from the Bridge 2006 tie._
### 12.13 Reversal-aware signal pipeline, first version: pairs, leave-one-target-out reversal risk and a selective rule (2026-10-09; analysis/reversal_table.py, analysis/reversal_model.py, tools/reversal_pipeline.sh, jobs 10943035 and 10943165)

Task: Prof. Hazra's reversal-aware signal pipeline (Project_idea.pdf, added 2026-10-06), with her
decisions of 2026-10-08: candidate pairs as rows, every pair of a held-out target held out together,
near-ties their own class, the benchmark tie rule for the main label, Algebra 2006 ambiguous in the
robustness label, the 7 KT targets only.

Design. One row per target, estimator seed (42, 1, 2) and pair of the 8 candidates (7 encoders and
scratch), a being the candidate H-score ranks higher: 588 rows. Label from the Track B N=3000 gold,
6 fine-tuning seeds: near_tie if the paired bootstrap interval of gold(a) - gold(b) includes zero or
|d| <= 0.001 (the benchmark rule), otherwise stable or reversal: 314 stable, 162 near-tie, 112
reversal; the robustness label marks the Algebra 2006 in-domain/Junyi pair ambiguous (3 rows).
Features, none reading the held-out target's gold or any test split: frozen (H-score and LogME at
the final layer, margins, ranks), early fine-tuning (validation AUC at epochs 1 to 3 of the
fine-tune with the estimator's seed, slopes, ranks, gaps), history (each candidate as a foreign
source on the other targets), layers (H-score at layer 3, layer agreement, skill-table reset).
Model: L2-regularized logistic regression for P(reversal), fitted on stable and reversal pairs of
the training targets, features standardized on the training fold, penalty chosen inside each fold by
an inner leave-one-target-out. Recommendation: H-score's top pick, its risk the largest predicted
risk over the pairs it heads, abstain above 0.5 (fixed before any result).

Reproduction checks: H-score's pretrained regret is 0.0012 over 21 rankings with every per-target
value equal to 12.3 and 12.12; the skill-table reset flips Algebra 2006 to Junyi on seeds 1 and 2
(12.9); the ASSISTments 2017 in-domain encoder leads at epoch 1 and trails at epoch 3 (12.8). The
two pipeline runs wrote identical logs.

Files (md5): reversal_pairs.tsv 8f57558d142869b3e34a35ad18d095f7, reversal_pairs_targets.tsv
88746eb17c78250bd47a75d850ef8ec0, reversal_model_report.md d6684f62a402e4b7852e673bbde4d4e8,
reversal_model_recs.tsv d7b874b710acc7a2d611e987917fc426, reversal_model_pairs.tsv
aecb1dc3263881b08269010636c3be6a.

Report as written by analysis/reversal_model.py (headings lowered one level):


#### Main analysis (benchmark label)

Pairs fitted: 314 stable, 112 reversal; near-tie pairs predicted only (162, mean predicted risk 0.3257). Penalty per held-out target: algebra2005 10.0, algebra2006 10.0, assist2009 100.0, assist2017 100.0, bridge2006 100.0, ednet 100.0, junyi 10.0.

Pair-level AUC 0.8278, Brier 0.1406; per held-out target AUC: algebra2005 0.6833, algebra2006 0.8908, assist2009 0.7857, assist2017 0.8613, bridge2006 0.9344, ednet 1.0000, junyi 0.8864.

| predicted risk | pairs | mean predicted | observed reversal rate |
|---|---|---|---|
| 0.0 to 0.2 | 239 | 0.0744 | 0.0795 |
| 0.2 to 0.4 | 69 | 0.2827 | 0.2899 |
| 0.4 to 0.6 | 45 | 0.4904 | 0.5111 |
| 0.6 to 0.8 | 47 | 0.6998 | 0.6383 |
| 0.8 to 1.0 | 26 | 0.9044 | 0.7692 |

Recommendations judged: 21 (target x estimator seed).

| rule | followed | mean regret of followed | reversals caught | false alarms | missed |
|---|---|---|---|---|---|
| always follow H-score | 21 | 0.0012 | 0 | 0 | 7 |
| follow when risk <= 0.5 | 16 | 0.0015 | 1 | 4 | 6 |

AURC (lower is better): model 0.0015, frozen margin 0.0017, random 0.0012, oracle 0.0002.

#### Robustness (Algebra 2006 pair ambiguous)

Pairs fitted: 314 stable, 109 reversal; near-tie pairs predicted only (162, mean predicted risk 0.3186). Penalty per held-out target: algebra2005 10.0, algebra2006 10.0, assist2009 100.0, assist2017 100.0, bridge2006 100.0, ednet 100.0, junyi 10.0.

Pair-level AUC 0.8326, Brier 0.1381; per held-out target AUC: algebra2005 0.6991, algebra2006 0.9146, assist2009 0.8080, assist2017 0.8549, bridge2006 0.9385, ednet 1.0000, junyi 0.8794.

| predicted risk | pairs | mean predicted | observed reversal rate |
|---|---|---|---|
| 0.0 to 0.2 | 243 | 0.0724 | 0.0782 |
| 0.2 to 0.4 | 64 | 0.2886 | 0.2812 |
| 0.4 to 0.6 | 45 | 0.4911 | 0.5111 |
| 0.6 to 0.8 | 47 | 0.6953 | 0.6596 |
| 0.8 to 1.0 | 24 | 0.9018 | 0.7500 |

Recommendations judged: 18 (target x estimator seed).

| rule | followed | mean regret of followed | reversals caught | false alarms | missed |
|---|---|---|---|---|---|
| always follow H-score | 18 | 0.0012 | 0 | 0 | 4 |
| follow when risk <= 0.5 | 15 | 0.0014 | 0 | 3 | 4 |

AURC (lower is better): model 0.0017, frozen margin 0.0018, random 0.0012, oracle 0.0002.

#### Feature groups (main label)

| features | pair AUC | Brier | AURC |
|---|---|---|---|
| all | 0.8278 | 0.1406 | 0.0015 |
| frozen only | 0.7196 | 0.1796 | 0.0009 |
| all but frozen | 0.8852 | 0.1134 | 0.0010 |
| early only | 0.8848 | 0.1185 | 0.0012 |
| all but early | 0.8025 | 0.1472 | 0.0013 |
| history only | 0.7669 | 0.1602 | 0.0016 |
| all but history | 0.8037 | 0.1469 | 0.0014 |
| layers only | 0.4978 | 0.2026 | 0.0012 |
| all but layers | 0.8543 | 0.1278 | 0.0014 |

With 7 targets and 21 recommendations every calibration and selective estimate is coarse; read the counts, not only the rates.

Files: reversal_model_pairs.tsv aecb1dc3263881b08269010636c3be6a, reversal_model_recs.tsv d7b874b710acc7a2d611e987917fc426

| target | selected (seeds 42 / 1 / 2) | reversal probability | abstained | reversed |
|---|---|---|---|---|
| algebra2005 | ednet / junyi / junyi | 0.20 / 0.01 / 0.00 | 0/3 | 1/3 |
| algebra2006 | algebra2006 / algebra2006 / algebra2006 | 0.55 / 0.08 / 0.22 | 1/3 | 3/3 |
| assist2009 | assist2009 / assist2009 / assist2009 | 0.49 / 0.35 / 0.40 | 0/3 | 0/3 |
| assist2017 | assist2017 / assist2017 / assist2017 | 0.21 / 0.19 / 0.18 | 0/3 | 3/3 |
| bridge2006 | bridge2006 / bridge2006 / bridge2006 | 0.52 / 0.38 / 0.36 | 1/3 | 0/3 |
| ednet | ednet / ednet / ednet | 0.23 / 0.21 / 0.19 | 0/3 | 0/3 |
| junyi | ednet / ednet / ednet | 0.92 / 0.76 / 0.91 | 3/3 | 0/3 |

_Read: reversals between candidate pairs are predictable out of target (pair AUC 0.8278; calibration
close in the lower bins and overconfident in the top one, 0.9044 predicted against 0.7692 observed).
The recommendation rule does not yet help: abstaining above 0.5 follows 16 of 21 recommendations
with mean regret 0.0015, against 0.0012 for always following H-score, catching 1 reversal (Algebra
2006, seed 42) with 4 false alarms; it follows all three ASSISTments 2017 recommendations, which
reverse, and abstains on all three Junyi recommendations, whose EdNet pick is right. Its AURC
(0.0015) is above random (0.0012), while the frozen features alone give 0.0009. Early fine-tuning is
the strongest group for pairs (AUC 0.8848 alone) and the layer group carries no signal (0.4978
alone). With 7 targets and 21 recommendations every estimate here is coarse. Nothing was tuned after
these results were seen; changing the model to fit them would be research-history leakage in the
sense of the project constraints._
### 12.14 Reversal-aware signal pipeline, version 2: the H-score recommendation as the target (analysis/reversal_model_v2.py, design frozen in commit ccd1655 before its first run, job 11042633)

Task: Prof. Hazra's feedback on 12.13. Rows are only the pairs headed by H-score's pick, 7
alternatives per recommendation (147 rows; 123 stable, 13 near-tie, 11 reversal on the main label);
the target is whether the alternative overtakes the pick after fine-tuning; the pairwise risks are
combined by noisy-OR into a recommendation-level risk (the largest pairwise risk reported beside
it); features are frozen, early fine-tuning and history, without the layer group (early fine-tuning
alone as a variant); the all-pairs model of 12.13, and the same model with version 2's features, are
the baselines. Leave-one-target-out throughout; the penalty and the threshold are chosen inside each
fold from the training targets' out-of-fold predictions (the threshold by Youden's J), and a fixed
0.5 is reported beside it. Primary, fixed before the run: the head-pair model with frozen, early and
history features, noisy-OR risk and the fold-chosen threshold.

Reproduction check: the 12.13 all-pairs model, scored by its own rule (largest pairwise risk,
threshold 0.5), follows 16 of 21 recommendations, catches 1 of 7 reversals with 4 false alarms,
selective regret 0.0015, as in 12.13.

Files (md5): reversal_model_v2.py 47d206c63749bdc2eb0cee6fdda62d7b, reversal_model_v2_report.md
f145e45fee795193ad885100af7295be, reversal_model_v2_recs.tsv 1f8d84f8a598452c4ba74404526b2438,
reversal_model_v2_pairs.tsv 34fff18c059bab71fc1c6ce29ab3bd34.

Report as written by analysis/reversal_model_v2.py (headings lowered one level):


Primary (fixed before results): head: frozen + early + history, noisy-OR risk, fold-chosen threshold.

#### Main label

Recommendations judged: 21, of which 7 reversed. Always following H-score: mean regret 0.0012 (also the AURC of a random order); frozen margin AURC 0.0017.

| model | risk | threshold | followed | caught | false alarms | selective regret | AURC | Brier |
|---|---|---|---|---|---|---|---|---|
| head: frozen + early + history | noisy-or | fold-chosen | 9/21 | 0/7 | 12 | 0.0027 | 0.0027 | 0.3294 |
| head: frozen + early + history | noisy-or | fixed 0.5 | 15/21 | 0/7 | 6 | 0.0017 | 0.0027 | 0.3294 |
| head: frozen + early + history | max | fold-chosen | 6/21 | 3/7 | 12 | 0.0026 | 0.0027 | 0.3059 |
| head: frozen + early + history | max | fixed 0.5 | 21/21 | 0/7 | 0 | 0.0012 | 0.0027 | 0.3059 |
| head: early only | noisy-or | fold-chosen | 14/21 | 0/7 | 7 | 0.0019 | 0.0028 | 0.3503 |
| head: early only | noisy-or | fixed 0.5 | 13/21 | 0/7 | 8 | 0.0020 | 0.0028 | 0.3503 |
| head: early only | max | fold-chosen | 19/21 | 0/7 | 2 | 0.0014 | 0.0031 | 0.3381 |
| head: early only | max | fixed 0.5 | 20/21 | 0/7 | 1 | 0.0013 | 0.0031 | 0.3381 |
| all pairs: version 1 (12.13) | noisy-or | fold-chosen | 3/21 | 6/7 | 12 | 0.0034 | 0.0016 | 0.4285 |
| all pairs: version 1 (12.13) | noisy-or | fixed 0.5 | 7/21 | 4/7 | 10 | 0.0020 | 0.0016 | 0.4285 |
| all pairs: version 1 (12.13) | max | fold-chosen | 2/21 | 7/7 | 12 | 0.0000 | 0.0015 | 0.3665 |
| all pairs: version 1 (12.13) | max | fixed 0.5 | 16/21 | 1/7 | 4 | 0.0015 | 0.0015 | 0.3665 |
| all pairs: version 2 features | noisy-or | fold-chosen | 3/21 | 6/7 | 12 | 0.0034 | 0.0016 | 0.3815 |
| all pairs: version 2 features | noisy-or | fixed 0.5 | 9/21 | 3/7 | 9 | 0.0017 | 0.0016 | 0.3815 |
| all pairs: version 2 features | max | fold-chosen | 2/21 | 7/7 | 12 | 0.0000 | 0.0014 | 0.3261 |
| all pairs: version 2 features | max | fixed 0.5 | 16/21 | 0/7 | 5 | 0.0016 | 0.0014 | 0.3261 |

Primary model, fold-chosen thresholds (noisy-OR): algebra2005 0.0699, algebra2006 1.0000, assist2009 0.2088, assist2017 1.0000, bridge2006 0.2150, ednet 0.2098, junyi 0.2238; penalties: algebra2005 1.0, algebra2006 100.0, assist2009 100.0, assist2017 100.0, bridge2006 100.0, ednet 100.0, junyi 100.0.

| primary, noisy-OR risk | recommendations | mean predicted | observed reversal rate |
|---|---|---|---|
| 0.00 to 0.25 | 3 | 0.0076 | 0.3333 |
| 0.25 to 0.50 | 12 | 0.4203 | 0.5000 |
| 0.50 to 0.75 | 6 | 0.5828 | 0.0000 |
| 0.75 to 1.00 | 0 | nan | nan |

| target | seed | selected | noisy-OR risk | fold threshold | abstain | reversed |
|---|---|---|---|---|---|---|
| algebra2005 | 1 | src:junyi | 0.0028 | 0.0699 | 0 | 0 |
| algebra2005 | 2 | src:junyi | 0.0188 | 0.0699 | 0 | 0 |
| algebra2005 | 42 | src:ednet | 0.0011 | 0.0699 | 0 | 1 |
| algebra2006 | 1 | src:algebra2006 | 0.2855 | 1.0000 | 0 | 1 |
| algebra2006 | 2 | src:algebra2006 | 0.3339 | 1.0000 | 0 | 1 |
| algebra2006 | 42 | src:algebra2006 | 0.3232 | 1.0000 | 0 | 1 |
| assist2009 | 1 | src:assist2009 | 0.6407 | 0.2088 | 1 | 0 |
| assist2009 | 2 | src:assist2009 | 0.6075 | 0.2088 | 1 | 0 |
| assist2009 | 42 | src:assist2009 | 0.5734 | 0.2088 | 1 | 0 |
| assist2017 | 1 | src:assist2017 | 0.4245 | 1.0000 | 0 | 1 |
| assist2017 | 2 | src:assist2017 | 0.3940 | 1.0000 | 0 | 1 |
| assist2017 | 42 | src:assist2017 | 0.3932 | 1.0000 | 0 | 1 |
| bridge2006 | 1 | src:bridge2006 | 0.4673 | 0.2150 | 1 | 0 |
| bridge2006 | 2 | src:bridge2006 | 0.4773 | 0.2150 | 1 | 0 |
| bridge2006 | 42 | src:bridge2006 | 0.4731 | 0.2150 | 1 | 0 |
| ednet | 1 | src:ednet | 0.4970 | 0.2098 | 1 | 0 |
| ednet | 2 | src:ednet | 0.4756 | 0.2098 | 1 | 0 |
| ednet | 42 | src:ednet | 0.4991 | 0.2098 | 1 | 0 |
| junyi | 1 | src:ednet | 0.5180 | 0.2238 | 1 | 0 |
| junyi | 2 | src:ednet | 0.5468 | 0.2238 | 1 | 0 |
| junyi | 42 | src:ednet | 0.6106 | 0.2238 | 1 | 0 |

#### Robustness label (Algebra 2006 pair ambiguous)

Recommendations judged: 18, of which 4 reversed. Always following H-score: mean regret 0.0012 (also the AURC of a random order); frozen margin AURC 0.0018.

| model | risk | threshold | followed | caught | false alarms | selective regret | AURC | Brier |
|---|---|---|---|---|---|---|---|---|
| head: frozen + early + history | noisy-or | fold-chosen | 18/18 | 0/4 | 0 | 0.0012 | 0.0024 | 0.4188 |
| head: frozen + early + history | noisy-or | fixed 0.5 | 15/18 | 0/4 | 3 | 0.0014 | 0.0024 | 0.4188 |
| head: frozen + early + history | max | fold-chosen | 18/18 | 0/4 | 0 | 0.0012 | 0.0024 | 0.3137 |
| head: frozen + early + history | max | fixed 0.5 | 16/18 | 0/4 | 2 | 0.0013 | 0.0024 | 0.3137 |
| head: early only | noisy-or | fold-chosen | 18/18 | 0/4 | 0 | 0.0012 | 0.0024 | 0.3249 |
| head: early only | noisy-or | fixed 0.5 | 18/18 | 0/4 | 0 | 0.0012 | 0.0024 | 0.3249 |
| head: early only | max | fold-chosen | 18/18 | 0/4 | 0 | 0.0012 | 0.0024 | 0.2270 |
| head: early only | max | fixed 0.5 | 18/18 | 0/4 | 0 | 0.0012 | 0.0024 | 0.2270 |
| all pairs: version 1 (12.13) | noisy-or | fold-chosen | 3/18 | 3/4 | 12 | 0.0034 | 0.0016 | 0.3844 |
| all pairs: version 1 (12.13) | noisy-or | fixed 0.5 | 7/18 | 2/4 | 9 | 0.0019 | 0.0016 | 0.3844 |
| all pairs: version 1 (12.13) | max | fold-chosen | 2/18 | 4/4 | 12 | 0.0000 | 0.0017 | 0.3220 |
| all pairs: version 1 (12.13) | max | fixed 0.5 | 15/18 | 0/4 | 3 | 0.0014 | 0.0017 | 0.3220 |
| all pairs: version 2 features | noisy-or | fold-chosen | 2/18 | 4/4 | 12 | 0.0000 | 0.0015 | 0.3307 |
| all pairs: version 2 features | noisy-or | fixed 0.5 | 7/18 | 3/4 | 8 | 0.0015 | 0.0015 | 0.3307 |
| all pairs: version 2 features | max | fold-chosen | 2/18 | 4/4 | 12 | 0.0000 | 0.0012 | 0.2788 |
| all pairs: version 2 features | max | fixed 0.5 | 15/18 | 0/4 | 3 | 0.0014 | 0.0012 | 0.2788 |

Primary model, fold-chosen thresholds (noisy-OR): algebra2005 1.0000, algebra2006 1.0000, assist2009 1.0000, assist2017 1.0000, bridge2006 1.0000, ednet 1.0000, junyi 1.0000; penalties: algebra2005 0.01, algebra2006 100.0, assist2009 100.0, assist2017 0.01, bridge2006 100.0, ednet 100.0, junyi 100.0.

| primary, noisy-OR risk | recommendations | mean predicted | observed reversal rate |
|---|---|---|---|
| 0.00 to 0.25 | 4 | 0.0001 | 1.0000 |
| 0.25 to 0.50 | 11 | 0.3939 | 0.0000 |
| 0.50 to 0.75 | 1 | 0.5136 | 0.0000 |
| 0.75 to 1.00 | 2 | 0.8721 | 0.0000 |

| target | seed | selected | noisy-OR risk | fold threshold | abstain | reversed |
|---|---|---|---|---|---|---|
| algebra2005 | 1 | src:junyi | 0.7736 | 1.0000 | 0 | 0 |
| algebra2005 | 2 | src:junyi | 0.9705 | 1.0000 | 0 | 0 |
| algebra2005 | 42 | src:ednet | 0.0002 | 1.0000 | 0 | 1 |
| algebra2006 | 1 | src:algebra2006 | 0.2855 | 1.0000 | 0 | 0 (ambiguous) |
| algebra2006 | 2 | src:algebra2006 | 0.3339 | 1.0000 | 0 | 0 (ambiguous) |
| algebra2006 | 42 | src:algebra2006 | 0.3232 | 1.0000 | 0 | 0 (ambiguous) |
| assist2009 | 1 | src:assist2009 | 0.4512 | 1.0000 | 0 | 0 |
| assist2009 | 2 | src:assist2009 | 0.4167 | 1.0000 | 0 | 0 |
| assist2009 | 42 | src:assist2009 | 0.3910 | 1.0000 | 0 | 0 |
| assist2017 | 1 | src:assist2017 | 0.0000 | 1.0000 | 0 | 1 |
| assist2017 | 2 | src:assist2017 | 0.0000 | 1.0000 | 0 | 1 |
| assist2017 | 42 | src:assist2017 | 0.0000 | 1.0000 | 0 | 1 |
| bridge2006 | 1 | src:bridge2006 | 0.3127 | 1.0000 | 0 | 0 |
| bridge2006 | 2 | src:bridge2006 | 0.3205 | 1.0000 | 0 | 0 |
| bridge2006 | 42 | src:bridge2006 | 0.3138 | 1.0000 | 0 | 0 |
| ednet | 1 | src:ednet | 0.4220 | 1.0000 | 0 | 0 |
| ednet | 2 | src:ednet | 0.4053 | 1.0000 | 0 | 0 |
| ednet | 42 | src:ednet | 0.4212 | 1.0000 | 0 | 0 |
| junyi | 1 | src:ednet | 0.4310 | 1.0000 | 0 | 0 |
| junyi | 2 | src:ednet | 0.4473 | 1.0000 | 0 | 0 |
| junyi | 42 | src:ednet | 0.5136 | 1.0000 | 0 | 0 |

With 7 targets and 21 recommendations every estimate here is coarse.

Files: reversal_model_v2_recs.tsv 1f8d84f8a598452c4ba74404526b2438, reversal_model_v2_pairs.tsv 34fff18c059bab71fc1c6ce29ab3bd34

_Read: aligning the target with the recommendation did not improve the decision. The primary rule
follows 9 of 21 recommendations, catches 0 of 7 reversals and raises 12 false alarms; the
recommendations it follows have mean regret 0.0027, against 0.0012 for always following H-score, and
its AURC (0.0027) is above the 0.0012 of a random order. Its held-out risks run against the outcome:
as a detector of reversed recommendations their AUC is 0.12 (7 reversed, 14 not). The penalty sits
at the top of the grid in 6 of 7 folds, and the folds that hold out ASSISTments 2017 and Algebra
2006 choose a threshold of 1.0 (follow everything), so those reversals are missed by construction.
On the robustness label the primary rule follows all 18 recommendations, which is always following
H-score. The reversed recommendations come from three events (ASSISTments 2017 on three seeds,
Algebra 2006 on three, Algebra 2005 on one), so holding out a target removes its event entirely, and
the remaining events do not carry a signal that transfers across targets. With 7 targets this is a
sample-size limit as much as a modelling result. Nothing was tuned after the run._
### 12.15 Option 1: the recommendation-level test on the NLP benchmark (analysis/nlp_option1_eval.py, job 11261879; design docs/option1_nlp_design.md)

Task: Prof. Hazra approved 210 early fine-tunes (7 encoders x 5 tasks x 2 poolings x 3 estimator
seeds, 3 epochs each on the estimators' own 10,000-example samples, scripts/task4_early_ft.py),
starting with a 5-run timing and protocol check, and asked for the fixed epoch-3 rule to be kept
fixed for a test on new data. Labels are the head pairs of step A (published tuned gold, Task 4 tie
rule): 30 recommendations, 20 reversed, from 4 of 5 tasks. Evaluation is leave-one-task-out, both
poolings of a task held out together.

Timeline of the design. The rule was fixed on 2026-10-13 (analysis/fixed_rule_eval.py), before any
NLP early fine-tune existed. The design file was delivered before the 5-run check and committed in
359b78d after the check and after the full set had been submitted, before any output of the full set
was read; its first 4,266 bytes are the delivered version and its last section records this. The
check ran one encoder per task, so it could not show whether H-score's pick led at epoch 3.

Files (md5): option1_nlp_design.md 198ce6a6ecede9a26db35adf0cce6059, nlp_option1_eval.py
76f268952cc363be25271e4df738b5d7, nlp_head_pairs.tsv d023dd9346eb1020afa11d1dacbaf579,
nlp_head_recs.tsv e533ea40073f9ff4926af3194570008c, nlp_option1_report.md
011c052c9166199daa417efe0a6cf173, nlp_option1_recs.tsv 8a146de30d1cc103d8df324f983d99cb.

Report as written by analysis/nlp_option1_eval.py (headings lowered one level):


Early fine-tunes: 210 runs, all present, 3 epochs, V100-SXM2, identical training and validation rows across encoders; 7.6 GPU-hours in total. Recommendations: 30 (5 tasks x 2 poolings x 3 seeds), 20 reversed.

#### Primary comparison (epoch 3)

| method | followed | caught | false alarms | selective regret (points) |
|---|---|---|---|---|
| always follow H-score | 30/30 | 0/20 | 0 | 2.6967 |
| fixed epoch-3 rule | 8/30 | 18/20 | 4 | 0.5375 |
| learned head-pair, fold-chosen threshold | 24/30 | 3/20 | 3 | 2.5083 |
| learned head-pair, threshold 0.5 | 6/30 | 14/20 | 10 | 1.3850 |

Best-average rule (never abstains): mean regret 1.3670 points over the 10 settings (reproduces RESULTS.md 13.1). AURC: learned 2.5323, frozen margin 3.7104, random 2.6967. Learned model Brier 0.4830.

| learned risk (epoch 3) | recommendations | mean predicted | observed reversal rate |
|---|---|---|---|
| 0.00 to 0.25 | 5 | 0.0410 | 1.0000 |
| 0.25 to 0.50 | 1 | 0.4552 | 1.0000 |
| 0.50 to 0.75 | 0 | nan | nan |
| 0.75 to 1.00 | 24 | 0.9560 | 0.5833 |

#### Epochs 1 to 3: the minimum useful budget

| epochs | GPU-minutes per recommendation | rule followed | rule caught | rule false alarms | rule selective regret | learned followed | learned caught | learned false alarms | learned selective regret | learned AURC |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5.1 | 7/30 | 19/20 | 4 | 0.3071 | 24/30 | 3/20 | 3 | 2.5083 | 2.8069 |
| 2 | 10.2 | 5/30 | 18/20 | 7 | 2.7600 | 24/30 | 3/20 | 3 | 2.5083 | 2.6976 |
| 3 | 15.2 | 8/30 | 18/20 | 4 | 0.5375 | 24/30 | 3/20 | 3 | 2.5083 | 2.5323 |

#### Per held-out task (epoch 3)

| task | recommendations | reversed | rule abstains | rule caught | learned abstains | learned caught |
|---|---|---|---|---|---|---|
| agnews | 6 | 6 | 4 | 4 | 0 | 0 |
| mnli | 6 | 5 | 5 | 5 | 0 | 0 |
| qnli | 6 | 6 | 6 | 6 | 0 | 0 |
| rte | 6 | 3 | 6 | 3 | 6 | 3 |
| airline | 6 | 0 | 1 | 0 | 0 | 0 |

Without Airline (24 recommendations): the epoch-3 rule follows 3, catches 18 of 20 with 3 false alarms, selective regret 1.4333, against 3.3708 for always following H-score.

With 5 tasks and 30 recommendations every estimate here is coarse.

Files: nlp_option1_recs.tsv 8a146de30d1cc103d8df324f983d99cb

_Read: the fixed epoch-3 rule, set before these data existed, flags 18 of the 20 reversed
recommendations and 4 of the 10 that hold (Airline once, RTE on all three first-token rankings); it
follows 8 of 30, and those have mean regret 0.5375 points against 2.6967 for always following
H-score. The two reversals it follows are AG News with mean pooling (seeds 1 and 42, regret 2.15
points each). Its coverage is low: 22 of 30 recommendations fall back to full fine-tuning. The
learned head-pair model does not beat the rule: at its fold-chosen thresholds it abstains only on
RTE, so it catches 3 of 20, and the same decisions come out at every budget; its calibration is poor
(5 recommendations given a risk near 0.04 were all reversed). On budget, the rule at epoch 1 flags
19 of 20 with 4 false alarms at 5.1 GPU-minutes per recommendation, against 15.2 at epoch 3; epoch 2
is worse (18 of 20, 7 false alarms), so these counts do not order the budgets reliably. The cost of
fine-tuning all 7 encoders in full was not measured here, so the saving is not quantified. With 5
tasks and 30 recommendations every estimate is coarse._
### 12.16 Option 1 supplement: intervals, calibration, budget agreement and compute (analysis/nlp_option1_supplement.py, job 11263574; added after the 12.15 results, describing the same decisions)

Files (md5): nlp_option1_supplement.py b6de6a5e5859dfe02fb9b5df9cdc182c, nlp_option1_recs.tsv
8a146de30d1cc103d8df324f983d99cb, nlp_option1_supplement_report.md d404e124397cd141a82c431874bb0df7,
nlp_option1_supplement.tsv 58b7229abe69d88bb0dea920e77e7c93.

Report as written by analysis/nlp_option1_supplement.py:


Added after the 12.15 results were seen; it describes the same decisions and changes no rule, threshold or model. Intervals: Wilson 95% for counts, task-cluster bootstrap for selective regret (5 tasks, so the intervals are wide).

| method | epochs | caught [95% CI] | false alarms [95% CI] | followed | selective regret [95% CI] | GPU-minutes per recommendation |
|---|---|---|---|---|---|---|
| fixed rule | 1 | 19/20 [76%, 99%] | 4/10 [17%, 69%] | 7/30 | 0.31 [0.00, 2.15] | 5.1 |
| fixed rule | 2 | 18/20 [70%, 97%] | 7/10 [40%, 89%] | 5/30 | 2.76 [0.00, 6.90] | 10.2 |
| fixed rule | 3 | 18/20 [70%, 97%] | 4/10 [17%, 69%] | 8/30 | 0.54 [0.00, 2.15] | 15.2 |
| learned head-pair | 1 | 3/20 [5%, 36%] | 3/10 [11%, 60%] | 24/30 | 2.51 [0.46, 4.49] | 5.1 |
| learned head-pair | 2 | 3/20 [5%, 36%] | 3/10 [11%, 60%] | 24/30 | 2.51 [0.46, 4.49] | 10.2 |
| learned head-pair | 3 | 3/20 [5%, 36%] | 3/10 [11%, 60%] | 24/30 | 2.51 [0.46, 4.49] | 15.2 |

Always following H-score: regret 2.70 points [1.11, 4.15] over all 30 recommendations.

Rule calibration (observed reversal rate):

| epochs | among flagged | among followed |
|---|---|---|
| 1 | 19/23 (83%) | 1/7 (14%) |
| 2 | 18/25 (72%) | 2/5 (40%) |
| 3 | 18/22 (82%) | 2/8 (25%) |

Rule decisions agree between epoch 1 and epoch 3 on 29 of 30 recommendations, between epoch 2 and epoch 3 on 23 of 30.

Compute per recommendation (GPU-minutes, 7 encoders):

| budget | GPU-minutes | share of one full-data epoch |
|---|---|---|
| early, 1 epoch on the 10,000-example sample | 5.1 | 7% |
| early, 2 epochs on the 10,000-example sample | 10.2 | 14% |
| early, 3 epochs on the 10,000-example sample | 15.2 | 21% |
| full fine-tuning, one epoch over the full training split (lower bound) | 72.5 | 100% |

Training-split sizes: agnews 120,000, mnli 392,702, qnli 104,743, rte 2,490, airline 10,248. The full-data figure scales the measured training time per example; the published fine-tunes ran several epochs with early stopping, so it understates their cost.

Files: nlp_option1_supplement.tsv 58b7229abe69d88bb0dea920e77e7c93

_Read: as a detector the fixed rule is strong on these data. At epoch 3 it flags 18 of 20 wrong
recommendations (95% interval 70% to 97%); 82% of the recommendations it flags are wrong, against
25% of those it follows; and those it follows have regret 0.54 points (task-cluster interval 0.00 to
2.15) against 2.70 (1.11 to 4.15) for always following H-score. Its weaknesses are the false-alarm
rate, 4 of 10 correct recommendations (17% to 69%), and coverage, 8 of 30. Epoch 1 makes the same
decision as epoch 3 on 29 of 30 recommendations at a third of the cost (5.1 against 15.2 GPU-minutes
per recommendation, 7% and 21% of one epoch of full fine-tuning over the full training splits, a
lower bound because the authors' tuned setting allows up to 50 epochs with early stopping); epoch 2
agrees on only 23, so the budgets are not ordered by the counts alone. The learned head-pair model
catches 3 of 20 (5% to 36%) at every budget. Five tasks make every interval wide._
### 12.17 Frozen state before phase 2, and the fixed rule on StudentBERT at one and three epochs

Freeze. Sections 12.13 to 12.16 are frozen together with the write-up
docs/pipeline_results_20261020.md and its checksum manifest, and the repository is tagged
pipeline-results-20261020 (commit bcb20aa, 2026-10-20). Later results go into new sections. The rule
carried into phase 2 is the same check after one epoch: flag H-score's recommendation when its pick
is not the validation leader after one epoch of fine-tuning (docs/phase2_complete_rule_design.md and
docs/phase2_amendment1.md).

Files (md5): fixed_rule_eval.py 94fc5f277d758c6b64fc8ebd5e169de2, reversal_pairs.tsv
8f57558d142869b3e34a35ad18d095f7, fixed_rule_report.md 9c9d0608a9c5f75e205b6f9e92aa34d3,
fixed_rule_recs.tsv f2373c453120d3301a724410ccaa3706, fixed_rule_e1_report.md
67e44aacb5ef638cad2a7b19c412c9fb, fixed_rule_e1_recs.tsv bae76c73c9558791b4e67c91d604f19f,
pipeline_results_20261020.md 68f5ec603cd8e523974b621282e3d569, phase2_complete_rule_design.md
254538daf4b91503fe1cfe767f32ce0f, phase2_amendment1.md ce8e827bfe9ffee6a074b498915c29cc.

The fixed rule on StudentBERT at epoch 3 (analysis/fixed_rule_eval.py, as fixed on 2026-10-13):


Rule: abstain when H-score's pick is not the validation leader at epoch 3 of the fine-tune with the estimator's seed; otherwise follow H-score. Nothing is fitted.

#### Main label

| rule | followed | caught | false alarms | selective regret |
|---|---|---|---|---|
| always follow H-score | 21/21 | 0/7 | 0 | 0.0012 |
| fixed epoch-3 rule | 12/21 | 6/7 | 3 | 0.0009 |

| target | seed | selected | epoch-3 leader | abstain | reversed |
|---|---|---|---|---|---|
| algebra2005 | 1 | src:junyi | 1 | 0 | 0 |
| algebra2005 | 2 | src:junyi | 1 | 0 | 0 |
| algebra2005 | 42 | src:ednet | 1 | 0 | 1 |
| algebra2006 | 1 | src:algebra2006 | 0 | 1 | 1 |
| algebra2006 | 2 | src:algebra2006 | 0 | 1 | 1 |
| algebra2006 | 42 | src:algebra2006 | 0 | 1 | 1 |
| assist2009 | 1 | src:assist2009 | 1 | 0 | 0 |
| assist2009 | 2 | src:assist2009 | 1 | 0 | 0 |
| assist2009 | 42 | src:assist2009 | 0 | 1 | 0 |
| assist2017 | 1 | src:assist2017 | 0 | 1 | 1 |
| assist2017 | 2 | src:assist2017 | 0 | 1 | 1 |
| assist2017 | 42 | src:assist2017 | 0 | 1 | 1 |
| bridge2006 | 1 | src:bridge2006 | 0 | 1 | 0 |
| bridge2006 | 2 | src:bridge2006 | 1 | 0 | 0 |
| bridge2006 | 42 | src:bridge2006 | 0 | 1 | 0 |
| ednet | 1 | src:ednet | 1 | 0 | 0 |
| ednet | 2 | src:ednet | 1 | 0 | 0 |
| ednet | 42 | src:ednet | 1 | 0 | 0 |
| junyi | 1 | src:ednet | 1 | 0 | 0 |
| junyi | 2 | src:ednet | 1 | 0 | 0 |
| junyi | 42 | src:ednet | 1 | 0 | 0 |

#### Robustness label (Algebra 2006 pair ambiguous)

| rule | followed | caught | false alarms | selective regret |
|---|---|---|---|---|
| always follow H-score | 18/18 | 0/4 | 0 | 0.0012 |
| fixed epoch-3 rule | 12/18 | 3/4 | 3 | 0.0009 |

| target | seed | selected | epoch-3 leader | abstain | reversed |
|---|---|---|---|---|---|
| algebra2005 | 1 | src:junyi | 1 | 0 | 0 |
| algebra2005 | 2 | src:junyi | 1 | 0 | 0 |
| algebra2005 | 42 | src:ednet | 1 | 0 | 1 |
| algebra2006 | 1 | src:algebra2006 | 0 | 1 | 0 (ambiguous) |
| algebra2006 | 2 | src:algebra2006 | 0 | 1 | 0 (ambiguous) |
| algebra2006 | 42 | src:algebra2006 | 0 | 1 | 0 (ambiguous) |
| assist2009 | 1 | src:assist2009 | 1 | 0 | 0 |
| assist2009 | 2 | src:assist2009 | 1 | 0 | 0 |
| assist2009 | 42 | src:assist2009 | 0 | 1 | 0 |
| assist2017 | 1 | src:assist2017 | 0 | 1 | 1 |
| assist2017 | 2 | src:assist2017 | 0 | 1 | 1 |
| assist2017 | 42 | src:assist2017 | 0 | 1 | 1 |
| bridge2006 | 1 | src:bridge2006 | 0 | 1 | 0 |
| bridge2006 | 2 | src:bridge2006 | 1 | 0 | 0 |
| bridge2006 | 42 | src:bridge2006 | 0 | 1 | 0 |
| ednet | 1 | src:ednet | 1 | 0 | 0 |
| ednet | 2 | src:ednet | 1 | 0 | 0 |
| ednet | 42 | src:ednet | 1 | 0 | 0 |
| junyi | 1 | src:ednet | 1 | 0 | 0 |
| junyi | 2 | src:ednet | 1 | 0 | 0 |
| junyi | 42 | src:ednet | 1 | 0 | 0 |

Retrospective: the rule was suggested by diagnostic D8 (RESULTS.md 12.8) on these same targets, so this is development evidence, not a held-out test.

Files: fixed_rule_recs.tsv f2373c453120d3301a724410ccaa3706

The same rule at epoch 1 (analysis/fixed_rule_eval.py --epoch 1):


Rule: abstain when H-score's pick is not the validation leader at epoch 1 of the fine-tune with the estimator's seed; otherwise follow H-score. Nothing is fitted.

#### Main label

| rule | followed | caught | false alarms | selective regret |
|---|---|---|---|---|
| always follow H-score | 21/21 | 0/7 | 0 | 0.0012 |
| fixed epoch-1 rule | 17/21 | 1/7 | 3 | 0.0009 |

| target | seed | selected | epoch-1 leader | abstain | reversed |
|---|---|---|---|---|---|
| algebra2005 | 1 | src:junyi | 0 | 1 | 0 |
| algebra2005 | 2 | src:junyi | 1 | 0 | 0 |
| algebra2005 | 42 | src:ednet | 0 | 1 | 1 |
| algebra2006 | 1 | src:algebra2006 | 1 | 0 | 1 |
| algebra2006 | 2 | src:algebra2006 | 1 | 0 | 1 |
| algebra2006 | 42 | src:algebra2006 | 1 | 0 | 1 |
| assist2009 | 1 | src:assist2009 | 0 | 1 | 0 |
| assist2009 | 2 | src:assist2009 | 1 | 0 | 0 |
| assist2009 | 42 | src:assist2009 | 1 | 0 | 0 |
| assist2017 | 1 | src:assist2017 | 1 | 0 | 1 |
| assist2017 | 2 | src:assist2017 | 1 | 0 | 1 |
| assist2017 | 42 | src:assist2017 | 1 | 0 | 1 |
| bridge2006 | 1 | src:bridge2006 | 1 | 0 | 0 |
| bridge2006 | 2 | src:bridge2006 | 1 | 0 | 0 |
| bridge2006 | 42 | src:bridge2006 | 1 | 0 | 0 |
| ednet | 1 | src:ednet | 1 | 0 | 0 |
| ednet | 2 | src:ednet | 1 | 0 | 0 |
| ednet | 42 | src:ednet | 1 | 0 | 0 |
| junyi | 1 | src:ednet | 1 | 0 | 0 |
| junyi | 2 | src:ednet | 1 | 0 | 0 |
| junyi | 42 | src:ednet | 0 | 1 | 0 |

#### Robustness label (Algebra 2006 pair ambiguous)

| rule | followed | caught | false alarms | selective regret |
|---|---|---|---|---|
| always follow H-score | 18/18 | 0/4 | 0 | 0.0012 |
| fixed epoch-1 rule | 14/18 | 1/4 | 3 | 0.0007 |

| target | seed | selected | epoch-1 leader | abstain | reversed |
|---|---|---|---|---|---|
| algebra2005 | 1 | src:junyi | 0 | 1 | 0 |
| algebra2005 | 2 | src:junyi | 1 | 0 | 0 |
| algebra2005 | 42 | src:ednet | 0 | 1 | 1 |
| algebra2006 | 1 | src:algebra2006 | 1 | 0 | 0 (ambiguous) |
| algebra2006 | 2 | src:algebra2006 | 1 | 0 | 0 (ambiguous) |
| algebra2006 | 42 | src:algebra2006 | 1 | 0 | 0 (ambiguous) |
| assist2009 | 1 | src:assist2009 | 0 | 1 | 0 |
| assist2009 | 2 | src:assist2009 | 1 | 0 | 0 |
| assist2009 | 42 | src:assist2009 | 1 | 0 | 0 |
| assist2017 | 1 | src:assist2017 | 1 | 0 | 1 |
| assist2017 | 2 | src:assist2017 | 1 | 0 | 1 |
| assist2017 | 42 | src:assist2017 | 1 | 0 | 1 |
| bridge2006 | 1 | src:bridge2006 | 1 | 0 | 0 |
| bridge2006 | 2 | src:bridge2006 | 1 | 0 | 0 |
| bridge2006 | 42 | src:bridge2006 | 1 | 0 | 0 |
| ednet | 1 | src:ednet | 1 | 0 | 0 |
| ednet | 2 | src:ednet | 1 | 0 | 0 |
| ednet | 42 | src:ednet | 1 | 0 | 0 |
| junyi | 1 | src:ednet | 1 | 0 | 0 |
| junyi | 2 | src:ednet | 1 | 0 | 0 |
| junyi | 42 | src:ednet | 0 | 1 | 0 |

Retrospective: the rule was suggested by diagnostic D8 (RESULTS.md 12.8) on these same targets, so this is development evidence, not a held-out test.

Files: fixed_rule_e1_recs.tsv bae76c73c9558791b4e67c91d604f19f

_Read: both rows are retrospective, because the rule was suggested by diagnostic D8 on these
targets. At epoch 3 the rule catches 6 of 7 reversed StudentBERT recommendations with 3 false
alarms; at epoch 1 it catches 1 of 7 (Algebra 2005, seed 42), also with 3 false alarms, because the
in-domain picks of ASSISTments 2017 and Algebra 2006 lead after one epoch on every seed and are
overtaken later. On the NLP benchmark one epoch matches three (12.16: 19 of 20 caught at epoch 1, 18
of 20 at epoch 3). So the one-epoch budget is supported on NLP but not on StudentBERT, and the
budget may depend on the domain._
### 12.18 Phase 2: the full framework on fresh seeds (analysis/phase2_eval.py, job 11271624; design docs/phase2_complete_rule_design.md and docs/phase2_amendment1.md, both committed before the fresh fine-tunes were read)

Task: Prof. Hazra's phase 2 list. Fresh estimator seeds 3, 4 and 5 (new samples on AG News, MNLI and
QNLI; Airline's samples overlap almost entirely with the old ones and RTE's are identical, so only
their fine-tuning randomness is new): new frozen features and scores, 30 fresh H-score
recommendations (18 overtaken after fine-tuning, from 4 of 5 tasks) and 210 early fine-tunes of 3
epochs. The full framework keeps H-score's pick unless the epoch-k leader's validation accuracy
exceeds it by more than 1.96 x sqrt(2 p (1 - p) / n); with a strict check it would equal the early
leader alone, which is reported as the ablation without H-score. Primary, fixed before the run: the
full framework at epoch 1 against H-score alone on mean regret. Four jobs ran twice after scheduler
errors on explorer-02; each of the 210 outputs is one complete run, checked by the evaluation.

Files (md5): phase2_complete_rule_design.md 254538daf4b91503fe1cfe767f32ce0f, phase2_amendment1.md
ce8e827bfe9ffee6a074b498915c29cc, phase2_eval.py d0ecd67479378b4d3dcbf3457253f5dc,
phase2_supplement.py bf1756e7072fd077aad769a4f452b50c, p2_head_report.md
62702fa6156404a02f6d5f34cf856d1d, p2_head_pairs.tsv 9123ad27c17496af7f5f51cdb1987bc1,
p2_head_recs.tsv a776f71b11a673d815dc6f486e4eb668, p2_eval_report.md
205f72d95e20cf2208a51428175a2e86, p2_eval_choices.tsv 21df978b07059d31198046609f88e1d0,
p2_supplement_report.md fb051bca22405d4d505d06b46f2b695d.

Report as written by analysis/phase2_eval.py:


30 fresh recommendations (seeds 3, 4, 5), 18 with H-score's pick overtaken. Early fine-tunes: 210 runs, all present, 3 epochs, V100-SXM2, identical rows across encoders; 7.6 GPU-hours. Regret in published fine-tuned points.

#### All five tasks

| method | top-1 | tied with best | mean regret [95% CI] | reversal recovery | false corrections | harmless switches | wins/ties/losses vs H-score |
|---|---|---|---|---|---|---|---|
| H-score alone | 12/30 | 12/30 | 2.37 [0.97, 3.56] | 0/18 [0%, 18%] | 0/12 [0%, 24%] | 0 | 0/30/0 |
| LogME alone | 15/30 | 18/30 | 2.08 [0.68, 3.47] | 7/18 [20%, 61%] | 1/12 [1%, 35%] | 3 | 7/19/4 |
| early leader, epoch 1 | 12/30 | 19/30 | 1.79 [0.55, 3.27] | 7/18 [20%, 61%] | 0/12 [0%, 24%] | 6 | 9/10/11 |
| early leader, epoch 2 | 12/30 | 18/30 | 1.74 [0.44, 3.90] | 6/18 [16%, 56%] | 0/12 [0%, 24%] | 5 | 12/8/10 |
| early leader, epoch 3 | 12/30 | 19/30 | 1.12 [0.45, 2.07] | 9/18 [29%, 71%] | 2/12 [5%, 45%] | 6 | 14/6/10 |
| full framework, epoch 1 | 12/30 | 16/30 | 1.95 [0.52, 3.45] | 4/18 [9%, 45%] | 0/12 [0%, 24%] | 3 | 5/22/3 |
| full framework, epoch 2 | 12/30 | 15/30 | 1.96 [0.54, 3.65] | 3/18 [6%, 39%] | 0/12 [0%, 24%] | 3 | 5/22/3 |
| full framework, epoch 3 | 15/30 | 18/30 | 1.42 [0.33, 3.02] | 6/18 [16%, 56%] | 0/12 [0%, 24%] | 3 | 9/18/3 |
| best-average rule | 6/30 | 12/30 | 1.37 [0.51, 2.33] | 1/18 [1%, 26%] | 1/12 [1%, 35%] | 5 | 12/12/6 |

| abstention rule | followed (coverage) | reversed among flagged | reversed among followed | selective regret [95% CI] |
|---|---|---|---|---|
| H-score + abstention, epoch 1 | 10/30 | 14/20 | 4/10 | 0.55 [0.00, 1.38] |
| H-score + abstention, epoch 2 | 8/30 | 17/22 | 1/8 | 0.27 [0.00, 1.61] |
| H-score + abstention, epoch 3 | 6/30 | 16/24 | 2/6 | 0.72 [0.00, 2.15] |

#### AG News, MNLI and QNLI (new samples)

| method | top-1 | tied with best | mean regret [95% CI] | reversal recovery | false corrections | harmless switches | wins/ties/losses vs H-score |
|---|---|---|---|---|---|---|---|
| H-score alone | 3/18 | 3/18 | 2.81 [1.38, 3.88] | 0/15 [0%, 20%] | 0/3 [0%, 56%] | 0 | 0/18/0 |
| LogME alone | 6/18 | 6/18 | 2.81 [0.72, 3.88] | 4/15 [11%, 52%] | 1/3 [6%, 79%] | 0 | 4/13/1 |
| early leader, epoch 1 | 6/18 | 9/18 | 1.43 [0.81, 2.55] | 6/15 [20%, 64%] | 0/3 [0%, 56%] | 2 | 7/5/6 |
| early leader, epoch 2 | 6/18 | 8/18 | 0.88 [0.66, 1.03] | 5/15 [15%, 58%] | 0/3 [0%, 56%] | 1 | 11/3/4 |
| early leader, epoch 3 | 7/18 | 10/18 | 0.73 [0.36, 1.22] | 8/15 [30%, 75%] | 1/3 [6%, 79%] | 2 | 11/2/5 |
| full framework, epoch 1 | 4/18 | 6/18 | 1.72 [0.62, 3.16] | 3/15 [7%, 45%] | 0/3 [0%, 56%] | 1 | 4/13/1 |
| full framework, epoch 2 | 5/18 | 6/18 | 1.51 [0.66, 2.49] | 3/15 [7%, 45%] | 0/3 [0%, 56%] | 1 | 5/12/1 |
| full framework, epoch 3 | 8/18 | 9/18 | 0.87 [0.42, 1.38] | 6/15 [20%, 64%] | 0/3 [0%, 56%] | 1 | 8/9/1 |
| best-average rule | 0/18 | 3/18 | 1.24 [1.08, 1.38] | 1/15 [1%, 30%] | 1/3 [6%, 79%] | 2 | 9/6/3 |

| abstention rule | followed (coverage) | reversed among flagged | reversed among followed | selective regret [95% CI] |
|---|---|---|---|---|
| H-score + abstention, epoch 1 | 5/18 | 11/13 | 4/5 | 1.11 [0.00, 1.38] |
| H-score + abstention, epoch 2 | 3/18 | 14/15 | 1/3 | 0.72 [0.00, 2.15] |
| H-score + abstention, epoch 3 | 2/18 | 13/16 | 2/2 | 2.15 [2.15, 2.15] |

#### Stability across the 3 fresh seeds

| method | settings with the same choice on all seeds | mean spread of regret (SD) |
|---|---|---|
| H-score alone | 8/10 | 0.46 |
| LogME alone | 8/10 | 0.37 |
| early leader, epoch 1 | 3/10 | 1.02 |
| early leader, epoch 2 | 2/10 | 0.96 |
| early leader, epoch 3 | 0/10 | 0.91 |
| full framework, epoch 1 | 5/10 | 0.96 |
| full framework, epoch 2 | 6/10 | 0.59 |
| full framework, epoch 3 | 5/10 | 0.77 |
| best-average rule | 10/10 | 0.00 |

#### Compute

| budget | GPU-minutes per recommendation |
|---|---|
| 1 epoch | 5.1 |
| 2 epochs | 10.2 |
| 3 epochs | 15.2 |

#### Failures of the full framework (epoch 1): choice not tied with the best

| task | pooling | seed | H-score's pick | epoch-1 leader | epoch-1 gap | delta | choice | regret |
|---|---|---|---|---|---|---|---|---|
| agnews | cls | 3 | cardiffnlp/twitter-roberta-base | cardiffnlp/twitter-roberta-base | 0.0000 | 0.0176 | cardiffnlp/twitter-roberta-base | 0.62 |
| agnews | cls | 4 | cardiffnlp/twitter-roberta-base | cardiffnlp/twitter-roberta-base | 0.0000 | 0.0171 | cardiffnlp/twitter-roberta-base | 0.62 |
| agnews | cls | 5 | cardiffnlp/twitter-roberta-base | distilbert-base-uncased | 0.0015 | 0.0176 | cardiffnlp/twitter-roberta-base | 0.62 |
| agnews | mean | 3 | roberta-base | distilbert-base-uncased | 0.0150 | 0.0182 | roberta-base | 2.15 |
| agnews | mean | 4 | roberta-base | roberta-base | 0.0000 | 0.0176 | roberta-base | 2.15 |
| agnews | mean | 5 | roberta-base | roberta-base | 0.0000 | 0.0167 | roberta-base | 2.15 |
| mnli | cls | 3 | dmis-lab/biobert-v1.1 | cardiffnlp/twitter-roberta-base | 0.0510 | 0.0281 | cardiffnlp/twitter-roberta-base | 1.39 |
| qnli | cls | 3 | dmis-lab/biobert-v1.1 | bert-base-uncased | 0.0040 | 0.0215 | dmis-lab/biobert-v1.1 | 3.66 |
| qnli | cls | 4 | dmis-lab/biobert-v1.1 | bert-base-uncased | 0.0065 | 0.0218 | dmis-lab/biobert-v1.1 | 3.66 |
| qnli | cls | 5 | dmis-lab/biobert-v1.1 | bert-base-uncased | 0.0095 | 0.0219 | dmis-lab/biobert-v1.1 | 3.66 |
| qnli | mean | 3 | dmis-lab/biobert-v1.1 | roberta-base | 0.0185 | 0.0211 | dmis-lab/biobert-v1.1 | 3.98 |
| qnli | mean | 4 | dmis-lab/biobert-v1.1 | bert-base-uncased | 0.0105 | 0.0222 | dmis-lab/biobert-v1.1 | 3.98 |
| rte | mean | 3 | bert-base-uncased | dmis-lab/biobert-v1.1 | 0.0397 | 0.0812 | bert-base-uncased | 6.90 |
| rte | mean | 4 | bert-base-uncased | cardiffnlp/twitter-roberta-base | 0.0181 | 0.0831 | bert-base-uncased | 6.90 |

Switches of the full framework (epoch 1) between encoders the published results cannot separate: 3.

With 5 tasks and 30 recommendations every estimate here is coarse.

Files: p2_eval_choices.tsv 21df978b07059d31198046609f88e1d0

Supplement, added after the results (analysis/phase2_supplement.py; part B is retrospective):


#### A. NLP fresh seeds: descriptive additions

| method | maximum regret | minus H-score, all tasks [95% CI] | minus H-score, AG News, MNLI, QNLI [95% CI] |
|---|---|---|---|
| H-score alone | 6.90 | | |
| LogME alone | 5.87 | -0.30 [-0.99, +0.27] | -0.00 [-0.67, +0.66] |
| early leader, epoch 1 | 9.01 | -0.58 [-1.94, +0.49] | -1.38 [-3.07, -0.46] |
| early leader, epoch 2 | 9.01 | -0.64 [-2.34, +1.07] | -1.93 [-3.22, -0.44] |
| early leader, epoch 3 | 6.91 | -1.26 [-2.55, +0.01] | -2.08 [-3.26, -0.17] |
| full framework, epoch 1 | 6.91 | -0.42 [-1.96, +0.69] | -1.09 [-3.26, +0.00] |
| full framework, epoch 2 | 6.91 | -0.42 [-1.93, +0.95] | -1.30 [-3.22, +0.00] |
| full framework, epoch 3 | 6.91 | -0.95 [-2.54, +0.42] | -1.93 [-3.46, +0.00] |
| best-average rule | 3.84 | -1.01 [-1.98, -0.07] | -1.56 [-2.61, +0.00] |

| warning signal | reversals caught | false alarms |
|---|---|---|
| epoch 1 | 14/18 [55%, 91%] | 6/12 [25%, 75%] |
| epoch 2 | 17/18 [74%, 99%] | 5/12 [19%, 68%] |
| epoch 3 | 16/18 [67%, 97%] | 8/12 [39%, 86%] |

#### B. StudentBERT, retrospective (21 recommendations, 7 with H-score's pick overtaken)

| method | top-1 | tied with best | mean regret (AUC) [95% CI] | reversal recovery | false corrections | stable across seeds |
|---|---|---|---|---|---|---|
| H-score alone | 11/21 | 14/21 | 0.0012 [0.0003, 0.0023] | 0/7 [0%, 35%] | 0/14 [0%, 22%] | 6/7 |
| LogME alone | 10/21 | 13/21 | 0.0017 [0.0006, 0.0027] | 0/7 [0%, 35%] | 1/14 [1%, 31%] | 5/7 |
| early leader, epoch 1 | 8/21 | 12/21 | 0.0031 [0.0007, 0.0070] | 0/7 [0%, 35%] | 2/14 [4%, 40%] | 4/7 |
| early leader, epoch 2 | 13/21 | 16/21 | 0.0010 [0.0002, 0.0025] | 5/7 [36%, 92%] | 3/14 [8%, 48%] | 5/7 |
| early leader, epoch 3 | 18/21 | 19/21 | 0.0006 [0.0000, 0.0016] | 6/7 [49%, 97%] | 1/14 [1%, 31%] | 4/7 |
| full framework, epoch 1 | 8/21 | 12/21 | 0.0031 [0.0007, 0.0070] | 0/7 [0%, 35%] | 2/14 [4%, 40%] | 4/7 |
| full framework, epoch 2 | 13/21 | 16/21 | 0.0011 [0.0002, 0.0026] | 4/7 [25%, 84%] | 2/14 [4%, 40%] | 5/7 |
| full framework, epoch 3 | 17/21 | 19/21 | 0.0006 [0.0001, 0.0016] | 6/7 [49%, 97%] | 1/14 [1%, 31%] | 4/7 |

| abstention rule | followed | reversed among flagged | reversed among followed |
|---|---|---|---|
| H-score + abstention, epoch 1 | 17/21 | 1/4 | 6/17 |
| H-score + abstention, epoch 2 | 12/21 | 6/9 | 1/12 |
| H-score + abstention, epoch 3 | 12/21 | 6/9 | 1/12 |

Every interval here is coarse: 5 NLP tasks and 7 StudentBERT targets.

_Read: the pre-declared primary comparison is inconclusive. The full framework at epoch 1 lowers
mean regret from 2.37 to 1.95 points (5 wins, 22 ties, 3 losses against H-score), but the paired
difference, -0.42, has a task-cluster interval from -1.96 to +0.69. It recovers 4 of 18 overtaken
recommendations and makes no false correction; the early leader alone recovers 7 of 18, also without
false corrections, with regret 1.79, so at one epoch H-score adds stability (5 of 10 settings choose
the same encoder on every seed, against 3) but not lower regret. More early fine-tuning helps
correction: at epoch 3 the early leader reaches 1.12 (difference -1.26, interval -2.55 to +0.01) and
the full framework 1.42. The best-average rule, which uses the published fine-tuned results of the
other tasks, has 1.37 and is the only method whose difference from H-score excludes zero (-1.01,
-1.98 to -0.07). The warning signal replicates less cleanly on fresh seeds: 14 of 18 overtaken
recommendations caught at epoch 1 with 6 of 12 false alarms (originally 19 of 20 and 4 of 10), 17 of
18 with 5 at epoch 2. The full framework's remaining failures at epoch 1 are AG News (6), where
H-score's pick leads after one epoch in 4 cases and is overtaken later; QNLI (5) and RTE (2), where
the leader's advantage is inside the validation noise (RTE has 277 validation rows); and one wrong
switch on MNLI. On StudentBERT (retrospective) one epoch makes things worse (regret 0.0031 against
0.0012, no reversal recovered), while three epochs recover 6 of 7 at regret 0.0006. In both domains,
then, one epoch is not enough to correct H-score, and the evidence does not yet support a complete
selection framework at that budget. Five tasks and seven targets make every interval wide._
### 12.19 Phase 3: the final fresh-seed test of the three-epoch framework (analysis/phase3_eval.py, job 11395653; design docs/phase3_final_test_design.md)

Task: Prof. Hazra's decision after 12.18 (the diagnosis and the warning signal are the main
contribution; correction stays secondary unless the three-epoch version holds on further fresh
seeds). Rule: the full framework at epoch 3 of 12.18. Fresh seeds 6, 7 and 8 (new samples on AG
News, MNLI and QNLI; Airline's overlap almost entirely with earlier ones and RTE's are identical).
Criterion, fixed in advance: the three-epoch version holds if the 95% task-cluster interval of its
regret minus H-score's lies entirely below zero.

Timeline, from git and sacct. The design, the generator and the evaluation were delivered before any
phase 3 job, but their commit step did not run. The 210 fine-tunes were submitted at 17:29 on
2026-10-26; the first job ended at 17:40:13; the design was committed at 17:45:25 (687613d), five
minutes later, and the evaluation read the fine-tunes afterwards (job 11395653). The design file's
commit note says no fine-tune had finished by then; that is wrong, as one job (6 runs) had. No
output was read before the commit, and the first 3,567 bytes of the design are the delivered
version.

H-score's picks on seeds 6 to 8 equal those on seeds 42, 1 and 2 in all 10 settings, although the
scores differ (new samples), so the 30 fresh labels match the original ones: 20 reversed, from 4 of
5 tasks.

Files (md5): phase3_final_test_design.md c10e130c866e721ef96d7d0d66fda637, gen_phase3_jobs.sh
c4d16748aca88164f70219dace81b3d1, phase3_eval.py a143007d38fe0a72bbd6c80e833bf37b, p3_head_report.md
84b62b7cbdc4bb91b17cbd4b008367a7, p3_head_pairs.tsv 9bad0afce43c426c8105473a6b8f918c,
p3_head_recs.tsv 1aaaca3159db41dc1ad1deffcc6d3245, p3_eval_report.md
0532591f84e64d2b2634b07faa7b0aee, p3_eval_choices.tsv e92570b7749f3fe4deab19f4250a7670.

Report as written by analysis/phase3_eval.py (headings lowered one level):


30 recommendations, 20 with H-score's pick overtaken; head-pair report md5 84b62b7cbdc4bb91b17cbd4b008367a7. Early fine-tunes: 210 runs, all present, 3 epochs, V100-SXM2, identical rows across encoders; 7.6 GPU-hours. Regret in published points.

#### Primary test (fixed before the run)

Full framework at epoch 3 minus H-score: -1.09 points, 95% task-cluster interval [-3.02, +0.00]. The three-epoch version DOES NOT HOLD (criterion: the interval lies entirely below zero).

#### Secondary: other methods minus H-score

| method | all five tasks [95% CI] | AG News, MNLI, QNLI [95% CI] |
|---|---|---|
| full framework, epoch 3 | -1.09 [-3.02, +0.00] | -1.81 [-4.83, +0.00] |
| early leader, epoch 3 | -1.95 [-3.67, -0.41] | -2.90 [-4.83, -0.57] |
| full framework, epoch 1 | -0.55 [-2.50, +1.00] | -1.53 [-4.17, +0.00] |
| early leader, epoch 1 | -1.05 [-2.76, +0.40] | -2.22 [-4.17, -0.80] |
| full framework, epoch 2 | -0.79 [-2.78, +0.39] | -1.54 [-4.63, +0.00] |
| early leader, epoch 2 | -1.59 [-3.25, -0.15] | -2.64 [-4.44, -0.68] |
| LogME alone | -0.53 [-1.08, -0.04] | -0.38 [-0.95, +0.00] |
| best-average rule | -1.33 [-2.68, -0.07] | -2.10 [-3.55, +0.00] |

Phase 2 and phase 3 pooled (60 recommendations, seeds 3 to 8; secondary): full framework at epoch 3 minus H-score -1.02 [-2.68, +0.21].

#### Outcomes, all five tasks

| method | top-1 | tied with best | mean regret [95% CI] | reversal recovery | false corrections | harmless switches | wins/ties/losses vs H-score |
|---|---|---|---|---|---|---|---|
| H-score alone | 10/30 | 10/30 | 2.70 [1.04, 4.15] | 0/20 [0%, 16%] | 0/10 [0%, 28%] | 0 | 0/30/0 |
| LogME alone | 13/30 | 16/30 | 2.17 [0.86, 3.47] | 6/20 [15%, 52%] | 0/10 [0%, 28%] | 3 | 6/21/3 |
| early leader, epoch 1 | 12/30 | 19/30 | 1.65 [0.52, 3.18] | 10/20 [30%, 70%] | 1/10 [2%, 40%] | 4 | 15/8/7 |
| early leader, epoch 2 | 16/30 | 23/30 | 1.10 [0.34, 2.21] | 13/20 [43%, 82%] | 0/10 [0%, 28%] | 4 | 16/7/7 |
| early leader, epoch 3 | 18/30 | 22/30 | 0.75 [0.14, 1.57] | 12/20 [39%, 78%] | 0/10 [0%, 28%] | 3 | 17/9/4 |
| full framework, epoch 1 | 10/30 | 14/30 | 2.14 [0.54, 3.84] | 4/20 [8%, 42%] | 0/10 [0%, 28%] | 3 | 7/20/3 |
| full framework, epoch 2 | 13/30 | 16/30 | 1.90 [0.35, 3.45] | 6/20 [15%, 52%] | 0/10 [0%, 28%] | 2 | 6/21/3 |
| full framework, epoch 3 | 16/30 | 17/30 | 1.61 [0.28, 2.94] | 7/20 [18%, 57%] | 0/10 [0%, 28%] | 1 | 7/22/1 |
| best-average rule | 6/30 | 12/30 | 1.37 [0.51, 2.33] | 2/20 [3%, 30%] | 0/10 [0%, 28%] | 4 | 14/12/4 |

| abstention rule | followed (coverage) | reversed among flagged | reversed among followed | selective regret [95% CI] |
|---|---|---|---|---|
| H-score + abstention, epoch 1 | 8/30 | 17/22 | 3/8 | 1.22 [0.00, 3.27] |
| H-score + abstention, epoch 2 | 7/30 | 19/23 | 1/7 | 0.31 [0.00, 2.15] |
| H-score + abstention, epoch 3 | 9/30 | 18/21 | 2/9 | 0.31 [0.00, 1.19] |

#### Outcomes, AG News, MNLI and QNLI

| method | top-1 | tied with best | mean regret [95% CI] | reversal recovery | false corrections | harmless switches | wins/ties/losses vs H-score |
|---|---|---|---|---|---|---|---|
| H-score alone | 1/18 | 1/18 | 3.34 [1.38, 4.83] | 0/17 [0%, 18%] | 0/1 [0%, 79%] | 0 | 0/18/0 |
| LogME alone | 4/18 | 4/18 | 2.96 [1.18, 3.88] | 3/17 [6%, 41%] | 0/1 [0%, 79%] | 0 | 3/15/0 |
| early leader, epoch 1 | 6/18 | 9/18 | 1.13 [0.59, 2.14] | 8/17 [26%, 69%] | 0/1 [0%, 79%] | 0 | 13/4/1 |
| early leader, epoch 2 | 8/18 | 12/18 | 0.70 [0.39, 1.03] | 11/17 [41%, 83%] | 0/1 [0%, 79%] | 1 | 14/1/3 |
| early leader, epoch 3 | 9/18 | 10/18 | 0.44 [0.00, 0.82] | 9/17 [31%, 74%] | 0/1 [0%, 79%] | 0 | 14/3/1 |
| full framework, epoch 1 | 3/18 | 4/18 | 1.82 [0.66, 3.41] | 3/17 [6%, 41%] | 0/1 [0%, 79%] | 0 | 6/12/0 |
| full framework, epoch 2 | 5/18 | 6/18 | 1.80 [0.19, 3.82] | 5/17 [13%, 53%] | 0/1 [0%, 79%] | 0 | 5/12/1 |
| full framework, epoch 3 | 7/18 | 7/18 | 1.53 [0.00, 3.21] | 6/17 [17%, 59%] | 0/1 [0%, 79%] | 0 | 6/12/0 |
| best-average rule | 0/18 | 3/18 | 1.24 [1.08, 1.38] | 2/17 [3%, 34%] | 0/1 [0%, 79%] | 1 | 11/6/1 |

| abstention rule | followed (coverage) | reversed among flagged | reversed among followed | selective regret [95% CI] |
|---|---|---|---|---|
| H-score + abstention, epoch 1 | 4/18 | 14/14 | 3/4 | 2.45 [0.00, 3.82] |
| H-score + abstention, epoch 2 | 1/18 | 16/17 | 1/1 | 2.15 [2.15, 2.15] |
| H-score + abstention, epoch 3 | 3/18 | 15/15 | 2/3 | 0.92 [0.00, 1.38] |

| warning signal | reversals caught | false alarms |
|---|---|---|
| epoch 1 | 17/20 [64%, 95%] | 5/10 [24%, 76%] |
| epoch 2 | 19/20 [76%, 99%] | 4/10 [17%, 69%] |
| epoch 3 | 18/20 [70%, 97%] | 3/10 [11%, 60%] |

#### Stability across the 3 seeds

| method | same choice on all seeds | mean spread of regret (SD) |
|---|---|---|
| H-score alone | 9/10 | 0.27 |
| LogME alone | 8/10 | 0.30 |
| early leader, epoch 1 | 0/10 | 1.23 |
| early leader, epoch 2 | 2/10 | 1.16 |
| early leader, epoch 3 | 4/10 | 0.58 |
| full framework, epoch 1 | 5/10 | 0.70 |
| full framework, epoch 2 | 6/10 | 0.66 |
| full framework, epoch 3 | 7/10 | 0.82 |
| best-average rule | 10/10 | 0.00 |

| budget | GPU-minutes per recommendation |
|---|---|
| 1 epoch | 5.1 |
| 2 epochs | 10.1 |
| 3 epochs | 15.2 |

#### Failures of the full framework at epoch 3: choice not tied with the best

| task | pooling | seed | H-score's pick | epoch-3 leader | epoch-3 gap | delta | choice | regret |
|---|---|---|---|---|---|---|---|---|
| agnews | cls | 6 | cardiffnlp/twitter-roberta-base | roberta-base | 0.0150 | 0.0162 | cardiffnlp/twitter-roberta-base | 0.62 |
| agnews | cls | 7 | cardiffnlp/twitter-roberta-base | cardiffnlp/twitter-roberta-base | 0.0000 | 0.0163 | cardiffnlp/twitter-roberta-base | 0.62 |
| agnews | cls | 8 | cardiffnlp/twitter-roberta-base | bert-base-uncased | 0.0025 | 0.0165 | cardiffnlp/twitter-roberta-base | 0.62 |
| agnews | mean | 6 | roberta-base | roberta-base | 0.0000 | 0.0162 | roberta-base | 2.15 |
| agnews | mean | 7 | roberta-base | bert-base-uncased | 0.0060 | 0.0173 | roberta-base | 2.15 |
| agnews | mean | 8 | roberta-base | bert-base-uncased | 0.0040 | 0.0170 | roberta-base | 2.15 |
| qnli | cls | 7 | dmis-lab/biobert-v1.1 | roberta-base | 0.0085 | 0.0213 | dmis-lab/biobert-v1.1 | 3.66 |
| qnli | cls | 8 | dmis-lab/biobert-v1.1 | cardiffnlp/twitter-roberta-base | 0.0160 | 0.0209 | dmis-lab/biobert-v1.1 | 3.66 |
| qnli | mean | 6 | dmis-lab/biobert-v1.1 | roberta-base | 0.0130 | 0.0221 | dmis-lab/biobert-v1.1 | 3.98 |
| qnli | mean | 7 | dmis-lab/biobert-v1.1 | cardiffnlp/twitter-roberta-base | 0.0070 | 0.0215 | dmis-lab/biobert-v1.1 | 3.98 |
| qnli | mean | 8 | dmis-lab/biobert-v1.1 | cardiffnlp/twitter-roberta-base | 0.0160 | 0.0211 | dmis-lab/biobert-v1.1 | 3.98 |
| rte | mean | 7 | bert-base-uncased | roberta-base | 0.0578 | 0.0775 | bert-base-uncased | 6.90 |
| rte | mean | 8 | bert-base-uncased | roberta-base | 0.0108 | 0.0802 | bert-base-uncased | 6.90 |

With 5 tasks and 30 recommendations every estimate here is coarse.

Files: p3_eval_choices.tsv e92570b7749f3fe4deab19f4250a7670

_Read: by the criterion fixed in advance, the three-epoch framework does not hold: its regret minus
H-score's is -1.09 points with an interval from -3.02 to +0.00, which does not lie below zero, so
correction stays secondary. The noise margin is what holds it back: the framework never makes a
false correction, but of its 13 failures, 11 keep H-score's pick because the early leader's
advantage is inside the margin (AG News, QNLI, RTE) and in 2 (AG News) H-score's pick is itself the
epoch-3 leader. The early leader alone at epoch 3, a secondary analysis fixed in the design, reduces
regret by 1.95 points (-3.67 to -0.41) with no false correction, but its choice changes across seeds
in 6 of 10 settings; in phase 2 the same comparison gave -1.26 (-2.55 to +0.01). The best-average
rule (-1.33) and LogME (-0.53) also improve on H-score here. The warning signal replicates: 17 of 20
overtaken recommendations flagged at epoch 1, 19 at epoch 2 and 18 at epoch 3, with 5, 4 and 3 false
alarms out of 10. Pooled with phase 2, the framework at epoch 3 gives -1.02 (-2.68 to +0.21). Five
tasks make every interval wide._
### 12.20 The warning signal as a detector, and at matched coverage (analysis/warning_metrics.py, commit 632d8df, job 11421958)

Task: Prof. Hazra (2026-10-27) asked for a more formal evaluation of the warning signal, with
precision and recall and AUROC and AUPRC across epochs. The analysis was defined and run after every
warning result in 12.15 to 12.19 had been seen; it changes no rule and no decision. Graded score:
the best alternative's validation score after k epochs minus that of H-score's pick, so the fixed
rule flags exactly when it is above zero. Comparison without fine-tuning: H-score's standardized
margin over its runner-up, negated. The matched-coverage comparison follows the rule's own number of
recommendations and takes the same number with the largest frozen margins.

Ties: the original NLP evaluation (analysis/nlp_option1_eval.py) ranked by validation accuracy and
then by encoder name; the replications (analysis/phase2_eval.py) and StudentBERT break a tie in
H-score's favour. The script checks each set against its own implementation, and its counts
reproduce 12.15 to 12.19. Only one tie changes a reported count: AG News, first-token pooling, seed
1, epoch 3, where the original rule's 18 of 20 becomes 17 of 20 under the replications' tie
handling.

Files (md5): warning_metrics.py fb52474862374e4eadc13f2890ac21cb, test_warning_metrics.py
f715ef05c0d5651fed07363034cb3560, warning_metrics_report.md bd6d07d9a0ea2c604bed8a0030167f8c,
warning_metrics.tsv e0f40b73114194830b854e155a47c4b7, warning_metrics_coverage.tsv
4865675cb63b56d50ed18ccb719bb505, reversal_pairs.tsv 8f57558d142869b3e34a35ad18d095f7.

Report as written by analysis/warning_metrics.py (headings lowered two levels):


Added after the warning results were seen; no rule or decision changes. Score: the best alternative's validation score after k epochs minus that of H-score's pick; the rule flags when it is above zero. Precision and recall at that threshold reproduce RESULTS.md 12.15 to 12.19. AUROC and AUPRC: threshold-free, with a cluster bootstrap over tasks (NLP) or targets (StudentBERT). Frozen margin: H-score's standardized margin over its runner-up, negated, which needs no fine-tuning.

| data | signal | precision (flagged) | recall (overtaken) | AUROC [95% CI] | AUPRC [95% CI] |
|---|---|---|---|---|---|
| StudentBERT (retrospective) | epoch 1 | 1/4 (0.25) | 1/7 (0.14) | 0.53 [0.22, 1.00] | 0.45 [0.16, 1.00] |
| StudentBERT (retrospective) | epoch 2 | 6/9 (0.67) | 6/7 (0.86) | 0.84 [0.46, 1.00] | 0.60 [0.33, 1.00] |
| StudentBERT (retrospective) | epoch 3 | 6/9 (0.67) | 6/7 (0.86) | 0.83 [0.00, 1.00] | 0.85 [0.05, 1.00] |
| StudentBERT (retrospective) | frozen margin | -- | -- | 0.56 [0.05, 0.91] | 0.47 [0.05, 0.90] |
| NLP, seeds 42, 1, 2 (prospective) | epoch 1 | 19/23 (0.83) | 19/20 (0.95) | 0.69 [0.00, 1.00] | 0.71 [0.45, 1.00] |
| NLP, seeds 42, 1, 2 (prospective) | epoch 2 | 18/25 (0.72) | 18/20 (0.90) | 0.54 [0.00, 1.00] | 0.66 [0.33, 1.00] |
| NLP, seeds 42, 1, 2 (prospective) | epoch 3 | 17/21 (0.81) | 17/20 (0.85) | 0.66 [0.00, 1.00] | 0.70 [0.44, 1.00] |
| NLP, seeds 42, 1, 2 (prospective) | frozen margin | -- | -- | 0.66 [0.23, 0.90] | 0.84 [0.37, 0.99] |
| NLP, seeds 3, 4, 5 | epoch 1 | 14/20 (0.70) | 14/18 (0.78) | 0.62 [0.17, 0.95] | 0.64 [0.47, 0.97] |
| NLP, seeds 3, 4, 5 | epoch 2 | 17/22 (0.77) | 17/18 (0.94) | 0.66 [0.19, 0.95] | 0.65 [0.47, 0.96] |
| NLP, seeds 3, 4, 5 | epoch 3 | 16/24 (0.67) | 16/18 (0.89) | 0.60 [0.33, 0.88] | 0.63 [0.44, 0.89] |
| NLP, seeds 3, 4, 5 | frozen margin | -- | -- | 0.56 [0.15, 0.82] | 0.67 [0.30, 0.95] |
| NLP, seeds 6, 7, 8 | epoch 1 | 17/22 (0.77) | 17/20 (0.85) | 0.60 [0.03, 1.00] | 0.69 [0.45, 1.00] |
| NLP, seeds 6, 7, 8 | epoch 2 | 19/23 (0.83) | 19/20 (0.95) | 0.77 [0.36, 1.00] | 0.78 [0.57, 1.00] |
| NLP, seeds 6, 7, 8 | epoch 3 | 18/21 (0.86) | 18/20 (0.90) | 0.81 [0.39, 1.00] | 0.82 [0.62, 1.00] |
| NLP, seeds 6, 7, 8 | frozen margin | -- | -- | 0.71 [0.29, 0.93] | 0.88 [0.44, 0.99] |
| NLP, all three seed sets | epoch 1 | 50/65 (0.77) | 50/58 (0.86) | 0.64 [0.11, 0.94] | 0.67 [0.43, 0.95] |
| NLP, all three seed sets | epoch 2 | 54/70 (0.77) | 54/58 (0.93) | 0.65 [0.17, 0.94] | 0.67 [0.41, 0.95] |
| NLP, all three seed sets | epoch 3 | 51/66 (0.77) | 51/58 (0.88) | 0.68 [0.26, 0.93] | 0.69 [0.45, 0.94] |
| NLP, all three seed sets | frozen margin | -- | -- | 0.65 [0.20, 0.88] | 0.78 [0.34, 0.97] |

AUPRC of a random score: the share overtaken, StudentBERT (retrospective) 0.33, NLP, seeds 42, 1, 2 (prospective) 0.67, NLP, seeds 3, 4, 5 0.60, NLP, seeds 6, 7, 8 0.67, NLP, all three seed sets 0.64.

With 7 targets and 5 tasks the intervals are wide; the pooled NLP row reuses the same 5 tasks three times, so its interval resamples tasks, not seed sets.

Precision and recall above break ties in H-score's favour (s_k > 0) for every set. Exact ties at the top, where an alternative's validation score equals that of H-score's pick:
- StudentBERT (retrospective), epoch 3: assist2009 seed 2 (not overtaken; flagged by neither rule)
- NLP, seeds 42, 1, 2 (prospective), epoch 2: airline mean seed 2 (not overtaken; flagged by neither rule)
- NLP, seeds 42, 1, 2 (prospective), epoch 3: agnews cls seed 1 (overtaken; flagged by the original rule only)
- NLP, seeds 3, 4, 5, epoch 1: airline cls seed 3 (not overtaken; flagged by neither rule)
- NLP, seeds 6, 7, 8, epoch 1: airline cls seed 6 (not overtaken; flagged by neither rule)
- NLP, seeds 6, 7, 8, epoch 1: qnli cls seed 8 (overtaken; flagged by the original rule only)

Files: warning_metrics.tsv e0f40b73114194830b854e155a47c4b7

#### The same signals at matched coverage

Mean regret of the recommendations followed (overtaken among them in brackets): those the rule lets through, and the same number with the largest frozen margins. Regret in AUC on StudentBERT and in points on NLP; the original NLP test uses its own tie handling, as above.

| data | epoch | followed | always follow | rule follows | frozen margin, same number |
|---|---|---|---|---|---|
| StudentBERT (retrospective) | epoch 1 | 17/21 | 0.0012 | 0.0009 (6) | 0.0014 (6) |
| StudentBERT (retrospective) | epoch 2 | 12/21 | 0.0012 | 0.0003 (1) | 0.0014 (3) |
| StudentBERT (retrospective) | epoch 3 | 12/21 | 0.0012 | 0.0009 (1) | 0.0014 (3) |
| NLP, seeds 42, 1, 2 (prospective) | epoch 1 | 7/30 | 2.70 | 0.31 (1) | 3.80 (4) |
| NLP, seeds 42, 1, 2 (prospective) | epoch 2 | 5/30 | 2.70 | 2.76 (2) | 5.31 (4) |
| NLP, seeds 42, 1, 2 (prospective) | epoch 3 | 8/30 | 2.70 | 0.54 (2) | 3.32 (4) |
| NLP, seeds 3, 4, 5 | epoch 1 | 10/30 | 2.37 | 0.55 (4) | 3.24 (5) |
| NLP, seeds 3, 4, 5 | epoch 2 | 8/30 | 2.37 | 0.27 (1) | 4.05 (5) |
| NLP, seeds 3, 4, 5 | epoch 3 | 6/30 | 2.37 | 0.72 (2) | 3.45 (3) |
| NLP, seeds 6, 7, 8 | epoch 1 | 8/30 | 2.70 | 1.22 (3) | 3.08 (4) |
| NLP, seeds 6, 7, 8 | epoch 2 | 7/30 | 2.70 | 0.31 (1) | 2.96 (3) |
| NLP, seeds 6, 7, 8 | epoch 3 | 9/30 | 2.70 | 0.31 (2) | 3.39 (5) |

Files: warning_metrics_coverage.tsv 4865675cb63b56d50ed18ccb719bb505

_Read: as a threshold-free detector of overtaken recommendations, the early signal is moderate on
NLP (pooled AUROC 0.64 to 0.68, every interval including 0.5) and no better than the frozen margin
(0.65; AUPRC 0.78 against 0.67 to 0.69). Its precision at the rule's threshold, 0.67 to 0.86, sits
only modestly above the share overtaken, 0.60 to 0.67. On StudentBERT it separates well after two
and three epochs (AUROC 0.84 and 0.83 against 0.56), but only three targets have an overtaken
recommendation, so the intervals are uninformative, and this evidence is retrospective. At matched
coverage the two signals differ sharply: after three epochs the recommendations the rule follows
have mean regret 0.31 to 0.72 points on the three NLP seed sets, against 3.32 to 3.45 for the same
number with the largest frozen margins, more than following every recommendation (2.37 to 2.70); on
StudentBERT 0.0009 AUC against 0.0014. The exception is the prospective set after two epochs (2.76
points, two overtaken among five followed). The early signal is therefore not a sharper detector
than the frozen margin on NLP, but the recommendations it lets through are the ones whose errors are
cheap._
### 12.21 PARC as an additional transferability baseline (analysis/parc_report.py; design docs/parc_design.md, commit 6d841b9)

Task: Prof. Hazra (2026-10-27) asked to include PARC (Bolya, Mittapalli and Hoffman, NeurIPS 2021)
as an additional baseline. Settings, fixed in the design file and committed with the code before any
PARC job ran: a probe of 500 examples with at least two per class, PCA to 32 dimensions fitted on
the probe, distances one minus the Pearson correlation of the features and of the one-hot labels,
and the Spearman correlation of the two upper triangles; no depth heuristic. Features: the same
frozen causal features as the other StudentBERT estimators (7 GPU jobs, 168 scores), and the saved
NLP features of 13.1 (210 scores). Evaluation: the existing evaluators, unchanged, whose other rows
reproduce 12.7 and 13.1 exactly.

Files (md5): parc_design.md 359effa3f201685de2cfcd48dbcb5bc8, parc.py
7a0cc68bd4b9392c70070b31190badc5, score_parc.py 4bcfccd6a14b6add6b9a4af26d6c0293,
task4_score_parc.py ff529ab75b1e1bb0a19ce2bbbe23f33d, parc_report.py
2aeaf5ec71a244dccb7418d04cd0e331, parc_report.md 9168d99796667a3cb8f05c512604027c,
parc_estimators_summary.tsv d2a05f591c19c208778469ba542179ff, task4_parc_summary.tsv
9f83da2b69115ea0d0d6d6d406ba102b, task4_parc_scores.jsonl 5ebda1c51835cba7eca0b45a4b33e6a6,
tg1_parc_kt_algebra2005.jsonl 79d7a17779247908f5f60aca72bcae6f, tg1_parc_kt_algebra2006.jsonl
042568523fa3b24cd7826d8399c9eb8c, tg1_parc_kt_assist2009.jsonl e7b98c82b1ee1af19fcce299be4d5869,
tg1_parc_kt_assist2017.jsonl 50338441f0699dacd1f61279e957844e, tg1_parc_kt_bridge2006.jsonl
ba9a9deea80a50e2fe2a17a687b731c8, tg1_parc_kt_ednet.jsonl 10fef3f7fcb6ee9e6dbe0930c5d42eb5,
tg1_parc_kt_junyi.jsonl b08e7ac34a877e28e760b2c77b083c17.

Report as written by analysis/parc_report.py (headings lowered two levels):


Settings fixed in docs/parc_design.md before any score: probe of 500 with at least two per class, PCA to 32, no depth heuristic. Evaluated with the existing evaluators.

#### StudentBERT (7 targets x 3 estimator seeds)

| view | rankings | rho | tau | decisive pairs | top-1 | tied with best | mean regret | max regret | negative picks | seed stability |
|---|---|---|---|---|---|---|---|---|---|---|
| pretrained | 21 | -0.2347 | -0.1973 | 0.3801 | 1/21 | 1/21 | 0.0067 | 0.0103 | 5 | 0.3651 |
| practical | 21 | -0.2596 | -0.2075 | 0.3520 | 1/21 | 1/21 | 0.0092 | 0.0369 | 3 | 0.3878 |

Own-encoder picks: 2/21. PARC's top pretrained candidate per target at seeds 42, 1 and 2: algebra2005 ednet, bridge2006, assist2017; algebra2006 algebra2005, ednet, ednet; assist2009 ednet, assist2017, assist2017; assist2017 own, own, ednet; bridge2006 ednet, algebra2005, ednet; ednet bridge2006, bridge2006, algebra2005; junyi algebra2005, algebra2005, assist2017.

Check: the other 28 rows of the same evaluation equal benchmark_final/estimators_summary.tsv (RESULTS.md 12.7) exactly.

#### NLP benchmark (10 settings x 3 seeds)

| gold | rankings | rho | tau | pairwise | top-1 | tied with best | mean regret | max regret | seed stability |
|---|---|---|---|---|---|---|---|---|---|
| frozen | 30 | +0.2821 | +0.1968 | 0.6360 | 10/30 | 14/30 | 1.65 | 5.00 | 0.6317 |
| tuned | 30 | -0.0131 | -0.0063 | 0.5107 | 8/30 | 10/30 | 3.87 | 10.00 | 0.6317 |

Check: the same evaluator on the 13.1 score file reproduces all 6 estimator rows of RESULTS.md 13.1.

_Read: PARC, in the version fixed before scoring, is weak on both benchmarks. On StudentBERT its
rankings run against the fine-tuned gold (rho -0.23; top choice the best in 1 of 21 rankings; 5
negatively transferring picks; mean regret 0.0067 AUC, against H-score's +0.52, 11 of 21 and
0.0012), and it picks the target's own encoder in only 2 of 21 rankings, so it does not share the
own-encoder preference of H-score and LogME. On the NLP benchmark it agrees only weakly with frozen
performance (+0.28) and not with fine-tuned performance (-0.01; the fine-tuned best in 8 of 30
rankings; mean regret 3.87 points against H-score's 2.70). Its rankings also change more across
seeds (stability 0.37 on StudentBERT and 0.63 on NLP). Bolya et al. report PARC at its best with a
depth ensemble on a benchmark of diverse architectures; these numbers characterize the core method
without that ensemble, as fixed in the design._
## 13. Cross-domain check: NLP encoder selection (Bassignana et al., EMNLP 2022; analysis/task4_evaluate.py, analysis/task4_ceiling.py; /projects/algl/dai.hany/task4/)

Seven Hugging Face encoders (bert-base-uncased, roberta-base, distilbert-base-uncased,
emilyalsentzer/Bio_ClinicalBERT, dmis-lab/biobert-v1.1, cardiffnlp/twitter-roberta-base,
allenai/scibert_scivocab_uncased) on four of the paper's classification tasks (AGNews, MNLI, QNLI,
RTE), with [CLS] and mean pooling. Features from scripts/task4_extract.py at its defaults: the last
hidden layer, maximum length 256, a seeded sample of 10,000 training examples that does not depend
on the encoder (RTE's whole training split of 2,490, so its three seeds see the same examples); 3
sample seeds; 504 scores. Code 6dd0f0c; extraction Slurm 10673386 to 10673392 (one gpu-partition job
per encoder), scoring 10673657 (short partition). Gold: the paper's Table 2
(configs/task4/bassignana2022_table2.tsv, checked against the paper's own correlations by
analysis/task4_gold_check.py), frozen and fully fine-tuned means over 5 seeds. A choice ties with
the best within 0.1 points or inside a two-sided 95% interval from the published SDs, a rule fixed
before any score. The paper's Airline and SciERC tasks were not run in this pass; on 2026-09-30
Airline was added and SciERC excluded (13.1). Published-table issues: the RTE mean-pooled rows
reproduce the paper's Pearson only to 0.605
and 0.581 against the printed 0.616 and 0.597, and the MNLI mean-pooled tuned SDs of the last four
models repeat SciERC's.

Files (md5): task4_eval_summary.tsv 17d4dfd7116bd9d2a0a4642719210c8d, task4_eval_cells.tsv
d1f2c4fe25d7d51d0a253dc50f3a3260, task4_eval_repro.tsv dab14daf29eb1ba817ab824f7b3e75f3,
task4_scores.jsonl 52ed7805e369eb479f5ca45afcc1cb34, task4_ceiling_report.md
dc37d1cc158b39359a646340948a52ba.

LogME reproduction (Spearman between our LogME, mean over seeds, and the published LogME over the 7
encoders):

| task | pooling | Spearman |
|---|---|---|
| agnews | cls | +0.9286 |
| agnews | mean | +0.8214 |
| mnli | cls | +0.4144 |
| mnli | mean | +0.5000 |
| qnli | cls | +0.6786 |
| qnli | mean | +0.5714 |
| rte | cls | +0.5714 |
| rte | mean | +0.6487 |

Estimators against the published gold (4 tasks x 2 poolings x 3 seeds; regret in points):

| estimator | gold | rho | tau-b | pairwise accuracy | top-1 | tied with best | regret mean / max | seed stability |
|---|---|---|---|---|---|---|---|---|
| hscore | frozen | +0.7649 | +0.6111 | 0.8548 | 16/24 | 19/24 | 1.05 / 4.06 | 0.9444 |
| hscore | tuned | +0.2098 | +0.1825 | 0.5931 | 4/24 | 4/24 | 3.37 / 6.90 | 0.9444 |
| hscore_shrunk | frozen | +0.7619 | +0.6310 | 0.8437 | 15/24 | 21/24 | 0.86 / 4.06 | 0.9286 |
| hscore_shrunk | tuned | +0.2143 | +0.1944 | 0.6194 | 2/24 | 5/24 | 3.84 / 6.90 | 0.9286 |
| logme | frozen | +0.7440 | +0.6071 | 0.8317 | 14/24 | 17/24 | 0.94 / 4.06 | 0.9524 |
| logme | tuned | +0.2068 | +0.1865 | 0.6070 | 7/24 | 10/24 | 3.07 / 5.87 | 0.9524 |

Reference rankings from the published table alone (analysis/task4_ceiling.py, commit af372a3; one
ranking per task and pooling): the frozen oracle ranks by the published frozen means; best average
ranks by mean tuned rank on the other published tasks, same pooling.

| ranking | gold | rho | tau-b | pairwise accuracy | top-1 | tied with best | regret mean / max |
|---|---|---|---|---|---|---|---|
| published_logme | frozen | +0.7829 | +0.6824 | 0.8920 | 3/8 | 5/8 | 1.38 / 4.61 |
| loto_best_average | frozen | +0.3770 | +0.2717 | 0.6447 | 2/8 | 3/8 | 2.43 / 5.54 |
| frozen_oracle | tuned | +0.3884 | +0.2857 | 0.6821 | 2/8 | 3/8 | 2.31 / 5.87 |
| published_logme | tuned | +0.5659 | +0.4428 | 0.7626 | 2/8 | 3/8 | 2.84 / 6.16 |
| loto_best_average | tuned | +0.5363 | +0.4598 | 0.7275 | 0/8 | 2/8 | 1.71 / 3.84 |

Tuned winners: RoBERTa on all six NLI settings, DistilBERT on both AGNews settings. BioBERT, never a
tuned winner, is H-score's pick in 11 of 24 rankings and LogME's in 15.

_Read: against frozen performance H-score ranks the encoders best (rho +0.7649, top-1 16/24),
narrowly ahead of LogME (+0.7440), as on StudentBERT. Against full fine-tuning both fall (+0.2098
and +0.2068) and pick the tuned winner in 4 and 7 of 24 rankings, with seed stability 0.9444 and
0.9524, so seed spread does not flag the failure. Exact knowledge of frozen performance picks the
tuned winner in 2 of 8 settings (rho +0.3884), so frozen performance is itself a weak guide to
fine-tuned performance here. The best-average rule has the lowest mean regret against tuned gold
(1.71 points) but never picks the winner. Our LogME reproduces the published ranking strongly on
AGNews and partly on the NLI tasks, and ranks tuned performance worse than the published LogME
(+0.5659 on the same settings)._
### 13.1 Airline added, SciERC excluded (2026-09-30; scripts/task4_extract.py at 7b1cfb7; /projects/algl/dai.hany/task4/)

Data: Kaggle's Tweets.csv (crowdflower/twitter-airline-sentiment, md5
2fa808ea99b32814ccd64d7097d935f2, 14,640 tweets), split by the authors' own converter
(mainlp/logme-nlp, commit 0046c725, project/src/tasks/sentiment/convert.py -rs 4012, run in a
separate clone because it is GPL-3.0; Slurm 10709670) into train 10,248 (md5
b0fa4865e8d442b5e2146715e7a84c10), dev 1,464 (3d396e482e100a1274ccaecbc0d14d07) and test 2,928
(88c93304b94179117fbadfc72fa5d3a9). The paper scores LogME on the train split and reports frozen and
tuned performance on dev; its gold rows are already in the table above.

Setting: the same seven hub encoders, none pretrained on the target data; budget 10,000 of the
10,248 training tweets, a seeded sample that does not depend on the encoder; sample seeds 42, 1 and
2; no training (frozen features; the tuned gold is the paper's fine-tuning); input the tweet text as
the converter leaves it, last hidden layer, [CLS] and the mean over non-padding subwords including
[CLS] and [SEP] (the authors average word vectors and skip special tokens), maximum length 256.
Extraction: 7 gpu-partition jobs (logs t4x_*.log in the folder above), every feature file recording
the split's md5; scoring Slurm 10716202 (126 new scores, all converged); evaluation 10716582.

Reproduction of the results above: the scores file grew from 504 to 630 lines and its first 504
lines hash to 52ed7805e369eb479f5ca45afcc1cb34, the 4-task file; rerun with --tasks agnews mnli qnli
rte the evaluation reproduces all four of its outputs byte for byte (summary
17d4dfd7116bd9d2a0a4642719210c8d), and the ceiling's default report reproduces too
(dc37d1cc158b39359a646340948a52ba).

Files (md5): task4_scores.jsonl 9dfe5d930681ec04d79cb52c149bbc4e, task4_eval_airline_summary.tsv
753d37b09376ed5b573a0ccfaa34d8e2, task4_eval_airline_repro.tsv c525a004ba2d7777c3f9480feeb523f7,
task4_eval_10_summary.tsv b5544568e4b1db985c0932156f8647fd, task4_ceiling_10_cells.tsv
04675b03fdfb32317e3502589c06f1fe, task4_ceiling_10_report.md 74e28a41113bddbbca88b0376824124f.

Airline, 2 poolings x 3 seeds (LogME reproduction: Spearman +1.0000 for [CLS] and +1.0000 for mean
pooling):

| estimator | gold | rho | tau-b | pairwise accuracy | top-1 | tied with best | regret mean / max | seed stability |
|---|---|---|---|---|---|---|---|---|
| hscore | frozen | +0.9464 | +0.8571 | 0.9472 | 6/6 | 6/6 | 0.00 / 0.00 | 1.0000 |
| hscore | tuned | +0.8750 | +0.7619 | 0.9349 | 6/6 | 6/6 | 0.00 / 0.00 | 1.0000 |
| hscore_shrunk | frozen | +0.9464 | +0.8571 | 0.9472 | 6/6 | 6/6 | 0.00 / 0.00 | 1.0000 |
| hscore_shrunk | tuned | +0.8750 | +0.7619 | 0.9349 | 6/6 | 6/6 | 0.00 / 0.00 | 1.0000 |
| logme | frozen | +0.9464 | +0.8571 | 0.9472 | 6/6 | 6/6 | 0.00 / 0.00 | 1.0000 |
| logme | tuned | +0.8750 | +0.7619 | 0.9349 | 6/6 | 6/6 | 0.00 / 0.00 | 1.0000 |

All five tasks run, 10 settings x 3 seeds (the 8-setting rows are the table above):

| estimator | gold | rho | tau-b | pairwise accuracy | top-1 | tied with best | regret mean / max | seed stability |
|---|---|---|---|---|---|---|---|---|
| hscore | frozen | +0.8012 | +0.6603 | 0.8733 | 22/30 | 25/30 | 0.84 / 4.06 | 0.9556 |
| hscore | tuned | +0.3429 | +0.2984 | 0.6615 | 10/30 | 10/30 | 2.70 / 6.90 | 0.9556 |
| hscore_shrunk | frozen | +0.7988 | +0.6762 | 0.8644 | 21/30 | 27/30 | 0.69 / 4.06 | 0.9429 |
| hscore_shrunk | tuned | +0.3464 | +0.3079 | 0.6825 | 8/30 | 11/30 | 3.07 / 6.90 | 0.9429 |
| logme | frozen | +0.7845 | +0.6571 | 0.8548 | 20/30 | 23/30 | 0.75 / 4.06 | 0.9619 |
| logme | tuned | +0.3405 | +0.3016 | 0.6725 | 13/30 | 16/30 | 2.46 / 5.87 | 0.9619 |

Reference rankings from the published table on the same 10 settings (analysis/task4_ceiling.py
--also_run airline):

| ranking | gold | rho | tau-b | pairwise accuracy | top-1 | tied with best | regret mean / max |
|---|---|---|---|---|---|---|---|
| published_logme | frozen | +0.8156 | +0.7173 | 0.9031 | 5/10 | 7/10 | 1.11 / 4.61 |
| loto_best_average | frozen | +0.3863 | +0.2954 | 0.6580 | 4/10 | 5/10 | 1.94 / 5.54 |
| frozen_oracle | tuned | +0.5036 | +0.4095 | 0.7457 | 4/10 | 5/10 | 1.85 / 5.87 |
| published_logme | tuned | +0.6277 | +0.5066 | 0.7971 | 4/10 | 5/10 | 2.27 / 6.16 |
| loto_best_average | tuned | +0.5137 | +0.4459 | 0.7253 | 2/10 | 4/10 | 1.37 / 3.84 |

SciERC is excluded. The authors' repository has no SciERC converter (project/src/tasks/relclass
holds only run scripts), so its entity-marked splits are not available, and the paper's LogME reads
the markers as plain text while its classifiers add them as special tokens. Under the tie rule
above, SciBERT, the domain-matched encoder, is also top-equivalent to the tuned best (BioBERT) in
both poolings, so SciERC could not show a domain-matched failure.

_Read: on Airline the domain-matched encoder is the right pick in both regimes and every estimator
makes it: Twitter-RoBERTa is the sole frozen best and the tuned best, tied with RoBERTa, and
H-score, its shrunk form and LogME pick it in all 6 rankings with zero regret and seed stability 1.
Frozen and tuned rankings agree there (frozen oracle rho +0.9643 against tuned gold in both
poolings), so adding Airline raises every estimator's rank correlation and lowers its mean regret:
H-score against tuned gold reaches rho +0.3429 and top-1 10/30 (LogME +0.3405, 13/30), and the
frozen oracle picks the tuned winner in 4/10 settings (2/8 above). The best-average rule keeps the
lowest mean tuned regret (1.37 points) and picks the winner only on the two Airline settings. The
failure review's fixed targets are defined on the 8 settings above._

_End of consolidated results._

