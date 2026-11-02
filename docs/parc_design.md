# PARC as an additional baseline (fixed before any PARC score exists)

Prof. Hazra asked on 2026-10-27 to include PARC as an additional transferability baseline. This
file fixes how it is computed and evaluated; it is committed together with the code before any
scoring job runs.

## Method (Bolya, Mittapalli and Hoffman, NeurIPS 2021, Eqs. 7 and 8)

- On a probe set of n target examples: D_f = 1 - corrcoef(features after PCA to f dimensions),
  D_y = 1 - corrcoef(one-hot labels); PARC is the Spearman correlation of the two upper
  triangles. Higher is better.
- Settings from the paper: n = 500 with at least two examples per class (Sec. 3.1), PCA to
  f = 32 (Table 4), PCA fitted on the probe. The paper's depth heuristic ("+l", Eq. 6) is a
  separate ensemble, not part of PARC, and is not used: the StudentBERT candidates share one
  architecture, and on the NLP benchmark it would reward layer count rather than the
  representation.

## Our features and probe

- StudentBERT: the same frozen, causal, final-layer features as H-score, LogME and NLEEP
  (scripts/score_task2.features_like_logme): the fine-tune's learner draw at the estimator seed,
  at most 50,000 positions, each paired with the next interaction's correctness. Candidates:
  scratch and the seven full encoders; estimator seeds 42, 1 and 2 (21 rankings).
- NLP: the saved last-layer features of Task 4 (10,000 training examples per task and seed, the
  whole training split for RTE), first-token and mean pooling, seeds 42, 1 and 2 (30 rankings),
  the version of RESULTS.md 13.1.
- Probe: 500 rows drawn with numpy default_rng(seed), topped up so each class has two; the same
  rows for every candidate at a seed.

## Evaluation (existing code, unchanged)

- StudentBERT: analysis/evaluate_estimators.py on the PARC score files, the same gold, tie rule
  and metrics as RESULTS.md 12.3; its non-estimator rows (largest source and the rules computed
  from the executions) must equal those of benchmark_final/estimators_summary.tsv.
- NLP: analysis/task4_evaluate.py on the PARC score file, against the published frozen and
  fine-tuned gold, as in RESULTS.md 13.1; its reference rows (frozen oracle, best-average rule)
  must equal those of 13.1.
- Reported whatever the outcome: PARC rows in Tables 3 and 5 and one sentence each in Sections
  4.2 and 5. PARC is a baseline; no claim of the paper depends on how it performs.
