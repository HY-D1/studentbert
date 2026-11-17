# Selective trust, amendment 1: points the design leaves open (fixed before any code)

Amends docs/selective_trust_design.md (md5 f63bfdc6bc0df97a97fb2fc97ef0eddd, commit c8ca7dd).
Written on 2026-11-17, before `analysis/selective_trust.py` exists and before any selective-trust
number has been computed. The design's signal, thresholds, budgets, normalized regret, targets
(0.10 and 0.50) and bootstrap settings are unchanged. Six points in it can be read in more than one
way, and each reading changes a reported number, so they are fixed here. Each choice is the more
conservative or the more literal reading.

## 1. Area under the normalized selective-regret against coverage curve (Reported, item 1)

The curve for a domain and budget is the set of distinct (coverage, normalized selective regret)
points over the 41 thresholds whose coverage is above zero. Thresholds with equal coverage trust the
same decisions (the trusted set shrinks as t grows), so duplicate points are identical; thresholds
with coverage zero have no selective regret and are left out. The area is the trapezoid rule over
these points sorted by coverage, from the smallest positive coverage c_min to the largest coverage
(1 at t = -1.00), and c_min is reported beside it. No point is extrapolated.

## 2. Leave-one-domain-out: ties and the case with no qualifying threshold (item 3)

Several thresholds can give the same pooled coverage (the same pooled trusted set) and still trust
different decisions in the held-out domain. Among the qualifying thresholds with the highest pooled
coverage, the transferred threshold is the largest t (the conservative end of the tied band). The
report also gives the band and the held-out coverage and normalized selective regret at its
smallest t. A threshold qualifies only if its pooled coverage is above zero and its pooled
normalized selective regret is at most 0.10; if none qualifies, the transfer is reported as "none"
for that budget and held-out domain. The held-out domain's own best threshold follows the same rule
on the held-out decisions alone (ties there trust the same decisions, so the band does not change
its result).

## 3. Pooling in the leave-one-domain-out choice (item 3)

The two source domains are pooled by decision: every decision has weight one and its regret is
divided by its own domain's mean regret of a random candidate. Pooled coverage is the number of
trusted decisions over the number of pooled decisions, and pooled normalized selective regret is the
mean normalized regret of the trusted decisions.

## 4. The domain-specific budget (item 4)

Budget k = 0 (no fine-tuning, every decision trusted, the static choice) counts as a budget and is
checked first; then k in the design's order. The smallest qualifying budget is reported with the
threshold that has the highest coverage among those meeting both targets (largest t on ties, as in
point 2). If no budget qualifies, the result is "not reached". The share of full compute is the
mean compute per decision at that budget (frozen scoring plus k epochs of every candidate, with
Quick-Tune's scoring off the axis as in 12.27) divided by the mean compute per decision of
fine-tuning every candidate to its last epoch (20 on StudentBERT, 50 on Quick-Tune), the same ratio
of means as in the 12.27 reading.

## 5. Operating points, intervals and the bootstrap (items 5 and 6)

The reported operating points are: t = 0 at every budget of every domain (item 2); the transferred
threshold, the smallest t of its band and the held-out domain's own best threshold at k = 1, 2 and
3 (item 3); and the budget-rule point of each domain (item 4). For each, the coverage interval is
the cluster bootstrap of the 0/1 trust indicator over all decisions of the domain, and the
selective-regret interval is the cluster bootstrap over the trusted decisions only (the convention
of 12.27, which resamples the clusters that hold a trusted decision), raw and normalized, with the
domain's normalizer held fixed. Both use `boot_ci` of analysis/regret_compute.py: 10,000
resamples, a new `random.Random(0)` per interval, clusters are targets (StudentBERT) or tasks (NLP,
Quick-Tune). No interval is given where nothing is trusted.

## 6. Multi-fidelity reference points at the same or nearest compute (item 5)

The references are the rows that analysis/regret_compute.py computes on the same decisions: for each
reported operating point, the early-leader row and the row of each successive-halving family
(factor 2 and factor 3) whose compute per decision is nearest in absolute GPU-hours, ties going to
the lower compute. On Quick-Tune frozen scoring is off the axis on both sides.

## Arithmetic

The normalized lead and every rule comparison (L at least t, coverage at least 0.50, normalized
selective regret at most 0.10) are computed exactly, with rational numbers built from the stored
values and thresholds t = i/20 for i = -20 to 20, so ties at grid points and L at least 0 (12.27's
"no other candidate strictly ahead") are decided without rounding error. Displayed numbers are
rounded once.
