# Paper C: the NLP selective-trust budget extended to 5 and 10 epochs (fixed before any run)

Prof. Hazra (2026-11-20, reply to the 11-17 update): in RESULTS.md 12.28 the NLP decisions never
reach the selective-trust targets, but their curves stop at epoch 3, so it cannot be told whether
NLP needs a larger early fine-tuning budget or whether selective trust never reaches the target
there. She asked to extend the existing NLP runs to 5 and 10 epochs with the same models, tasks,
seeds, decision rule and compute accounting, with no new threshold and no tuning on intermediate
results, and to show whether coverage and regret improve with budget and where the regret against
compute curve flattens or gets worse. This file fixes the runs and the analysis before any run is
submitted and before `analysis/nlp_budget_extension.py` is written. Written on 2026-11-20.

Known before: the NLP results up to epoch 3 (12.15, 12.19, 12.20, 12.27, 12.28), and that the
12.28 targets are not reached there, which motivated this extension. New: epochs 4 to 10, the
budgets 5 and 10, the flattening and worsening read-outs, and the transfer test at 5 and 10.

## Runs

- The 630 early fine-tunes of 12.27 and 12.28 (7 encoders x 5 tasks x 2 poolings x 9 estimator
  seeds: 42, 1, 2; 3, 4, 5; 6, 7, 8), rerun with `scripts/task4_early_ft.py` unchanged (md5
  9c87a401c955fc90aca772e5c17cbf08) and `--epochs 10`; every other setting is the script's default,
  as before (the estimators' 10,000-example training sample, at most 2,000 validation rows shared
  by the encoders, learning rate 5e-5 held constant, batch size 32, maximum length 256), on
  V100-SXM2 with envs/t4. No checkpoint was kept, so each run starts again from the pretrained
  encoder; with a constant learning rate and a seeded shuffle per epoch, its first 3 epochs replay
  the recorded ones up to GPU nondeterminism.
- Outputs go to new folders (/projects/algl/dai.hany/task4/early_ft10, early_ft10_fresh,
  early_ft10_p3, one per seed group); the recorded 3-epoch runs are not touched. Jobs are written
  by `slurm/generators/gen_nlp_ft10_jobs.sh`, one per encoder, task and pooling (its 3 seeds).
- Estimated compute: the 630 3-epoch runs took 22.8 GPU-hours (7.6 per seed group, 12.15 and
  later), so 10 epochs take about 76 GPU-hours.
- Protocol check first: 5 runs (bert-base-uncased, [CLS] pooling, seed 42, one per task, 10
  epochs) for timing and for how closely epochs 1 to 3 match the recorded runs
  (`tools/compare_ft_epochs.py`). The check can stop or delay the full set (a failed run, or a
  job that does not fit its time limit); it cannot change anything in this file. No output of the
  analysis is computed or read before all 630 runs are complete.

## Decisions, signal and outcome

As in 12.27 and 12.28: 90 NLP decisions (5 tasks x 2 poolings x 9 seeds), 7 candidate encoders,
the static choice is H-score's (the head-pair files of 12.27), the outcome is the published
fine-tuned performance, and the validation curve is the run's accuracy after each epoch. Compute
per decision is the frozen H-score scoring time plus the recorded time of k epochs of every
candidate (V100-SXM2), in GPU-hours, as in 12.27.

## One campaign

Every NLP budget, 1, 2, 3, 5 and 10, is computed from the new 10-epoch runs, so one table rests on
one campaign. For each of the 630 runs the report gives how its epochs 1 to 3 compare with the
recorded run (identical or not, and the largest difference in validation accuracy), and puts the
new t = 0 rows at 1, 2 and 3 epochs beside those of 12.28. Differences are reported, not used to
stop or adjust anything; 12.28 stays as recorded.

## Unchanged rules

The rules of docs/selective_trust_design.md with amendments 1 and 2, applied through the functions
of `analysis/selective_trust.py` (imported, md5 561f9d471f9df4686c35bf6f5c7129a0): the normalized
lead L, trust when L >= t, t from -1.00 to +1.00 in steps of 0.05, t = 0 as the common fixed rule,
normalized regret (the NLP normalizer depends on the outcomes only, so it is unchanged), the
targets 0.10 and 0.50, the area under the curve, ties, and intervals (95% cluster bootstrap over
the 5 tasks, 10,000 resamples, `random.Random(0)`). Budgets: k = 0 (trust all), 1, 2, 3, 5, 10.

## Reported

1. Per budget: compute per decision, and at t = 0 the coverage and the selective regret (raw and
   normalized) with intervals; the area over all thresholds; the full grid in a table file.
2. The domain-specific budget of 12.28 item 4 for NLP, now over k = 0, 1, 2, 3, 5, 10.
3. Whether more budget helps, for selective trust at t = 0 (normalized selective regret, with its
   coverage beside it), the early leader and successive halving (normalized regret over all
   decisions):
   - the budget with the lowest mean normalized regret (k*);
   - flattening: the smallest budget k after which no larger budget lowers the mean normalized
     regret by more than 0.02 (a fifth of the 0.10 target), that is, R(k) minus the minimum of
     R over the larger budgets is at most 0.02;
   - worsening: every budget larger than k* with a higher mean normalized regret, with a 95%
     cluster-bootstrap interval of R(k) minus R(k*), both computed on the same resampled tasks.
   These read-outs are descriptive.
4. References at each budget: the early leader at 1, 2, 3, 5 and 10 epochs; successive halving
   with the rungs 1, 2, 3 of 12.27 and with geometric rungs started at 1, 2 and 3 epochs (factors 2
   and 3, capped at 10), computed by `analysis/regret_compute.py` (imported, md5
   fd4288700ffd9af74fe92911371935ae); the nearest compute as in amendment 1, point 6.
5. Secondary: the leave-one-domain-out threshold transfer of 12.28 (primary, by decision, and the
   equal-domain-weight sensitivity analysis), at k = 5 and 10, which all three domains now share;
   StudentBERT and Quick-Tune decisions as in 12.28.

## Limitations, stated with the result

Five NLP tasks make the intervals wide. The outcome is the published fully fine-tuned performance,
not epoch 10 of these 10,000-example runs. Epochs 1 to 3 may differ slightly from the recorded runs
(GPU nondeterminism), which is why every budget comes from the new runs. The NLP result up to epoch
3 was known before this extension was designed.
