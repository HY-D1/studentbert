from __future__ import annotations

"""Option 1, step B: a fixed early fine-tuning rule, nothing learned, on the 7 StudentBERT targets.

The rule, stated before it was evaluated (option1_nlp_scoping.md): follow H-score unless, at epoch 3
of the fine-tune whose seed equals the estimator seed, H-score's pick is not the validation leader
among the 8 candidates; in that case abstain (run full fine-tuning). It has no parameters, so there
is nothing to fit or tune, and leave-one-target-out is not needed. It is retrospective evidence on
StudentBERT, because the rule was suggested by diagnostic D8 (RESULTS.md 12.8) on these targets;
on the NLP benchmark, whose early trajectories do not exist yet, it will be a baseline fixed before
the data. Scored with the recommendation-level metrics of analysis/reversal_model_v2.py.

--epoch 1 (added 2026-10-20) applies the same check after one epoch, the budget carried into
phase 2; it is retrospective evidence on StudentBERT too. The default, epoch 3, is unchanged.

  PYTHONPATH=. python analysis/fixed_rule_eval.py --pairs benchmark_final/reversal_pairs.tsv \
      --out benchmark_final/fixed_rule
"""

import argparse
import hashlib
from pathlib import Path

from analysis import reversal_model as rm
from analysis import reversal_model_v2 as v2


def rule_risk(rows: list[dict], epoch: int = 3) -> dict:
    """1.0 for every head pair of a recommendation whose pick is not the epoch-k leader, else 0."""
    col = f"rank_e{epoch}_a"
    return {rm.key(r): 0.0 if r[col] == "1" else 1.0 for r in rows if v2.is_head(r)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True, help="prefix for the _recs and _report files")
    ap.add_argument("--epoch", type=int, default=3, choices=(1, 3),
                    help="budget of the check (the pair table records ranks at epochs 1 and 3)")
    a = ap.parse_args(argv)
    rows = rm.load(a.pairs)
    head = [r for r in rows if v2.is_head(r)]
    risk = rule_risk(rows, a.epoch)
    k = a.epoch
    out = Path(a.out)
    L = ["# Fixed early fine-tuning rule on StudentBERT (retrospective evidence)", "",
         f"Rule: abstain when H-score's pick is not the validation leader at epoch {k} of the "
         "fine-tune with the estimator's seed; otherwise follow H-score. Nothing is fitted.", ""]
    allrecs = []
    for lab, title in (("label_main", "Main label"),
                       ("label_robust", "Robustness label (Algebra 2006 pair ambiguous)")):
        recs = v2.rec_rows(head, risk, lab)
        e = v2.evaluate(recs, "risk_noisy_or", None)
        s = e["fixed 0.5"]
        use = [r for r in recs if not r["ambiguous"]]
        always = sum(r["regret"] for r in use) / len(use)
        L += [f"## {title}", "",
              "| rule | followed | caught | false alarms | selective regret |",
              "|---|---|---|---|---|",
              f"| always follow H-score | {len(use)}/{len(use)} | 0/{e['reversed']} | 0 | "
              f"{always:.4f} |",
              f"| fixed epoch-{k} rule | {s['followed']}/{len(use)} | "
              f"{s['caught']}/{e['reversed']} | "
              f"{s['false_alarms']} | {s['selective_regret']:.4f} |", "",
              f"| target | seed | selected | epoch-{k} leader | abstain | reversed |",
              "|---|---|---|---|---|---|"]
        for r in recs:
            ab = int(r["risk_noisy_or"] > 0.5)
            L.append(f"| {r['target']} | {r['est_seed']} | {r['selected']} | {1 - ab} | {ab} | "
                     f"{r['reversed']}{' (ambiguous)' if r['ambiguous'] else ''} |")
            allrecs.append({"label": lab, **r, "abstain": ab})
        L.append("")
    p = out.with_name(out.name + "_recs.tsv")
    m = v2.write_tsv(p, allrecs)
    L += ["Retrospective: the rule was suggested by diagnostic D8 (RESULTS.md 12.8) on these same "
          "targets, so this is development evidence, not a held-out test.", "",
          f"Files: {p.name} {m}"]
    rp = out.with_name(out.name + "_report.md")
    rp.write_text("\n".join(L) + "\n")
    print(f"wrote {p.name} and {rp.name} md5 {hashlib.md5(rp.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
