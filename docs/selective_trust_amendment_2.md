# Selective trust, amendment 2: equal-domain-weight sensitivity analysis for the transfer test

Amends docs/selective_trust_design.md (md5 f63bfdc6bc0df97a97fb2fc97ef0eddd, commit c8ca7dd) and
docs/selective_trust_amendment_1.md (md5 de8a55b63b442dd2d1f28bb7bb2c1c9e). Written on 2026-11-17,
before any selective-trust number has been computed. Nothing in the design or amendment 1 changes;
this adds one labelled sensitivity analysis.

## Reason

In the primary leave-one-domain-out rule (amendment 1, point 3) the two source domains are pooled by
decision. By decision count Quick-Tune and NLP (90 each) outweigh StudentBERT (21), so the pooled
threshold can be dominated by one domain. The sensitivity analysis gives each source domain equal
weight.

## Sensitivity rule

At each budget k = 1, 2, 3 and for each held-out domain, with the two other domains as sources:

- pooled coverage is the mean of the two source domains' coverages;
- pooled normalized selective regret is the mean of the two source domains' normalized selective
  regrets (each the mean over that domain's trusted decisions of regret divided by that domain's
  mean regret of a random candidate);
- a threshold qualifies only if both source domains have coverage above zero and that mean is at
  most 0.10;
- among the qualifying thresholds with the highest pooled coverage, the transferred threshold is the
  largest t; the band of tied thresholds and the held-out result at its smallest t are reported as
  well; if no threshold qualifies, the result is "none" (amendment 1, point 2).

The chosen threshold is applied to the held-out domain and compared, as in the primary analysis,
with the held-out domain's own best threshold under the same target (unchanged; it uses the
held-out domain alone). The held-out coverage and normalized selective regret at the sensitivity
threshold (and at the smallest t of its band) are operating points with intervals as in amendment
1, point 5. Comparisons are exact, as in amendment 1.

## Reporting

The sensitivity result is reported beside the primary result at k = 1, 2 and 3, in the same table,
labelled "sensitivity analysis (equal domain weight)". The primary rule remains amendment 1,
point 3; a difference between the two is reported, not used to choose between them.
