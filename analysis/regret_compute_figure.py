from __future__ import annotations

"""Regret against compute, one panel per benchmark, from benchmark_final/regret_compute.tsv
(analysis/regret_compute.py). Methods that choose on every decision are plotted; the frontier is
joined in order of compute. Quick-Tune's static points sit at zero compute, because their frozen
scoring ran on other hardware than Quick-Tune's fine-tuning curves (docs/regret_compute_design.md).
The abstention rows are not plotted (they choose on only part of the decisions). For the record,
not a paper figure.

  python analysis/regret_compute_figure.py --tsv benchmark_final/regret_compute.tsv \
      --out benchmark_final/regret_compute_frontier.pdf
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

TSV_MD5 = "d77242c6243450571be358a5661c5ca6"
PANELS = (("studentbert", "StudentBERT", "mean regret (test AUC)"),
          ("nlp", "NLP (curves stop at epoch 3)", "mean regret (points)"),
          ("quicktune", "Quick-Tune micro (median curves)", "mean regret (top-1 points)"))
STYLE = {"static": ("s", "black"), "early leader": ("o", "tab:blue"),
         "override rule": ("^", "tab:orange"), "successive halving": ("D", "tab:green"),
         "full fine-tuning": ("*", "tab:red")}


def family(method: str) -> str:
    for k in STYLE:
        if method.startswith(k):
            return k
    return "other"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-pin", action="store_true", help="tests only: skip the md5 pin")
    a = ap.parse_args(argv)
    p = Path(a.tsv)
    got = hashlib.md5(p.read_bytes()).hexdigest()
    if not a.no_pin and got != TSV_MD5:
        sys.exit(f"ABORT: {p} md5 {got}, expected {TSV_MD5}")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in csv.DictReader(p.read_text().splitlines(), delimiter="\t")
            if r["chosen"] == r["decisions"]]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for ax, (bench, title, ylab) in zip(axes, PANELS):
        rs = [r for r in rows if r["benchmark"] == bench]
        if not rs:
            sys.exit(f"ABORT: no rows for {bench}")

        def x(r: dict) -> float:
            if bench == "quicktune" and r["method"].startswith("static"):
                return 0.0
            return float(r["compute_gpu_h"])
        seen = set()
        for r in rs:
            fam = family(r["method"])
            marker, colour = STYLE.get(fam, ("x", "grey"))
            hollow = "factor 3" in r["method"] or r["method"] == "static plain H-score"
            ax.scatter(x(r), float(r["regret"]), marker=marker, s=26,
                       facecolors="none" if hollow else colour, edgecolors=colour,
                       label=None if fam in seen else fam, zorder=3)
            seen.add(fam)
        front = sorted((r for r in rs if r["frontier"] == "True"), key=x)
        ax.plot([x(r) for r in front], [float(r["regret"]) for r in front], color="0.5",
                linewidth=0.9, linestyle="--", zorder=2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("GPU-hours per decision", fontsize=8)
        ax.set_ylabel(ylab, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={"CreationDate": None, "ModDate": None})
    print(f"wrote {out.name} md5 {hashlib.md5(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
