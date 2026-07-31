#!/usr/bin/env python3
"""
StudentBERT showcase poster (36in x 24in, vector PDF).
Every number is taken from the repo's RESULTS.md (canonical) or the project's
recorded headline analyses. Regenerate with:  python make_poster.py
"""
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# ---------------------------------------------------------------- style
RED    = "#D41B2C"   # Northeastern Red, Pantone 186 C (brand.northeastern.edu)
INK    = "#1F1F1F"
SLATE  = "#55565A"
BLUE   = "#23608F"   # correctness-driven regime
LBLUE  = "#7FA3C4"
PAPER  = "#F7F6F3"
CARD   = "#FFFFFF"
EDGE   = "#E3E0DA"
FAINT  = "#8A8B8F"

plt.rcParams.update({
    "font.family": "Liberation Sans",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})

FW, FH = 36.0, 24.0
fig = plt.figure(figsize=(FW, FH))
fig.patch.set_facecolor(PAPER)

def W(s, n):
    return "\n".join(textwrap.fill(p, n) for p in s.split("\n"))

def box(x, y, w, h, fc=CARD, ec=EDGE, lw=1.6, r=0.008, z=1):
    p = FancyBboxPatch((x, y), w, h, transform=fig.transFigure,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=z, mutation_aspect=FW/FH)
    fig.patches.append(p)
    return p

def T(x, y, s, size, color=INK, weight="normal", ha="left", va="top",
      ls=1.32, z=5, family=None, alpha=1.0):
    kw = {}
    if family:
        kw["family"] = family
    fig.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va,
             linespacing=ls, zorder=z, alpha=alpha, **kw)

CARDS = []

def mkax(rect, z=4):
    """Chart axes. Figure-level card patches sit at zorder 1 and Axes default to
    zorder 0, so an axes must be lifted above the cards or it is painted over."""
    a = fig.add_axes(rect)
    a.set_zorder(z)
    a.patch.set_visible(False)
    return a

# ---------------------------------------------------------------- geometry
MARG   = 0.013
GUT    = 0.011
NCOL   = 4
CW     = (1 - 2 * MARG - (NCOL - 1) * GUT) / NCOL
COLX   = [MARG + i * (CW + GUT) for i in range(NCOL)]
BODY_T = 0.795
GAP    = 0.012
PADX   = 0.0068

def stack(ci, heights):
    y = BODY_T
    out = []
    for h in heights:
        out.append((COLX[ci], y - h, CW, h))
        y -= h + GAP
    return out

def card(x, y, w, h, num, title):
    box(x, y, w, h)
    CARDS.append((x, y, w, h))
    T(x + PADX, y + h - 0.0065, num, 26, color=RED, weight="bold")
    tx = x + PADX + (0.0155 if num else 0.0)
    T(tx, y + h - 0.0065, title, 26, weight="bold")
    ry = y + h - 0.0325
    fig.patches.append(Rectangle((x + PADX, ry), w - 2 * PADX, 0.0012,
                                 transform=fig.transFigure, fc=RED, ec="none", zorder=3))
    return (x + PADX, y, w - 2 * PADX, ry - 0.008)  # content area (cx, cy_bottom, cw, content_top)

# ---------------------------------------------------------------- top + bottom accent rules
fig.patches.append(Rectangle((0, 0.9955), 1, 0.0045, transform=fig.transFigure, fc=RED, ec="none"))
fig.patches.append(Rectangle((0, 0), 1, 0.0045, transform=fig.transFigure, fc=RED, ec="none"))

# ---------------------------------------------------------------- header
# logo block (placeholder wordmark; swap in the official Khoury logo before print)
box(0.013, 0.9075, 0.0300, 0.0450, fc=RED, ec=RED, r=0.004)
T(0.0280, 0.9300, "N", 44, color="white", weight="bold", ha="center", va="center")
T(0.0475, 0.9445, "Northeastern University", 20.5, weight="bold", va="center")
T(0.0475, 0.9165, "Khoury College of Computer Sciences", 16.5, color=SLATE, va="center")

# title block
T(0.500, 0.9895, "S T U D E N T B E R T   ·   A  P R E T R A I N E D  M O D E L  F O R  S T U D E N T  I N T E R A C T I O N  D A T A",
  16.5, color=RED, weight="bold", ha="center")
T(0.500, 0.9700, "When Does Pretraining Help Model Student Learning?",
  58, weight="bold", ha="center")
T(0.500, 0.9265, "Whose past learning data to train on, when it actually helps, and a dataset property that goes with it. "
                 "7 datasets · 511,273 retained learner sequences · 115M interactions · 3 tasks.",
  22.5, color=SLATE, ha="center")
T(0.500, 0.9030, "Hanyu Dai   ·   Supervised by Prof. Hazra Imran", 21, ha="center")
T(0.500, 0.8830, "CS7980 Research Capstone   ·   Khoury College of Computer Sciences, Northeastern University Vancouver   ·   Summer 2026",
  17.5, color=SLATE, ha="center")

# QR placeholder (replace with a real QR code before printing)
qx, qy, qw, qh = 0.892, 0.9040, 0.0430, 0.0645
box(qx, qy, qw, qh, fc="#FBFAF8", ec=FAINT, lw=1.8, r=0.004)
T(qx + qw / 2, qy + qh * 0.62, "QR CODE", 15, color=FAINT, weight="bold", ha="center", va="center")
T(qx + qw / 2, qy + qh * 0.36, "(add before print)", 11.5, color=FAINT, ha="center", va="center")
T(qx + qw / 2, qy - 0.0075, "Code + results", 14, color=INK, weight="bold", ha="center")
T(qx + qw / 2, qy - 0.0215, "github.com/HY-D1/studentbert", 13, color=SLATE, ha="center")

# ---------------------------------------------------------------- takeaway banner
by, bh = 0.806, 0.062
box(MARG, by, 1 - 2 * MARG, bh, fc=RED, ec=RED, r=0.008)
LBLW = 0.070                      # left label strip: names the band as the summary
bw = (1 - 2 * MARG - LBLW) / 3
T(MARG + 0.014, by + bh / 2, "IN\nBRIEF", 19, color="white", weight="bold", va="center", ls=1.12)
fig.patches.append(Rectangle((MARG + LBLW - 0.004, by + 0.010), 0.0012, bh - 0.020,
                             transform=fig.transFigure, fc="white", alpha=0.45, ec="none"))
msgs = [
    "On ASSISTments next-skill prediction, gains were largest at N=25:\n+9.2 points top-1 there, at most +1.5 by 1,000.",
    "Source size mattered for knowledge tracing:\nthe 442K-student EdNet source was best in all 3 source tests.",
    "For knowledge tracing, skill-only versus correct-only preference varied\nby target, and tracked practice per skill. Reversed by a within-dataset test.",
]
for i, m in enumerate(msgs):
    cx = MARG + LBLW + i * bw
    T(cx + 0.010, by + bh / 2, str(i + 1), 34, color="white", weight="bold", va="center")
    T(cx + 0.024, by + bh / 2, m, 17.5, color="white", va="center", ls=1.35)
    if i:
        fig.patches.append(Rectangle((cx - 0.002, by + 0.010), 0.0012, bh - 0.020,
                                     transform=fig.transFigure, fc="white", alpha=0.45, ec="none"))

# ================================================================ COLUMN 1
(c1a, c1b, c1c) = stack(0, [0.160, 0.152, 0.408])

# ---- 1 Motivation
cx, cy, cw, ct = card(*c1a, "1", "Problem statement")
mot = [
    "Adaptive platforms model what each student knows from their interaction history (knowledge tracing).",
    "Those models are data-hungry, yet every new course or tool starts with only dozens of students.",
    "Cold-start learners get the least personalized support, and small classrooms may never reach big-data scale.",
    "In NLP and vision, pretrain-then-fine-tune is the standard fix. For student data there is no recipe.",
]
yy = ct
for m in mot:
    T(cx + 0.0085, yy, W(m, 62), 17.5, ls=1.28)
    fig.patches.append(Rectangle((cx + 0.0012, yy - 0.0058), 0.0035, 0.0035,
                                 transform=fig.transFigure, fc=RED, ec="none"))
    yy -= 0.0128 * (len(textwrap.wrap(m, 62))) + 0.0068

# ---- 2 Research questions
cx, cy, cw, ct = card(*c1b, "2", "Research questions")
rqs = [
    ("RQ1", "When does pretraining on student sequences help downstream tasks (knowledge tracing, next-skill, early disengagement)?"),
    ("RQ2", "Which source corpus transfers best: the one closest in skill-vocabulary size, or the biggest one?"),
    ("RQ3", "Which masked target transfers better, and what property of the target dataset goes with it?"),
]
yy = ct
for tag, q in rqs:
    T(cx + 0.0012, yy, tag, 17.5, color=RED, weight="bold")
    T(cx + 0.0185, yy, W(q, 56), 17.5, ls=1.28)
    yy -= 0.0128 * len(textwrap.wrap(q, 56)) + 0.0075

# ---- 3 Model & pipeline (diagram)
cx, cy, cw, ct = card(*c1c, "3", "Prototype and methodology")
T(cx, ct, W("StudentBERT is a BERT-style model for student learning histories: every step is a "
            "skill practiced, whether the answer was right, and a response-time bin.", 58),
  16, ls=1.24, color=SLATE)
ax = mkax([cx, cy + 0.010, cw, ct - cy - 0.052])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

def dbox(x, y, w, h, lines, fc="#FBFAF8", ec=INK, lw=1.6, sizes=None, colors=None, weights=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.018",
                                fc=fc, ec=ec, lw=lw, mutation_aspect=(cw*FW)/((ct-cy-0.024)*FH)))
    n = len(lines)
    for i, ln in enumerate(lines):
        fy = y + h * (n - i - 0.5) / n
        ax.text(x + w / 2, fy, ln, ha="center", va="center",
                fontsize=(sizes[i] if sizes else 15),
                color=(colors[i] if colors else INK),
                weight=(weights[i] if weights else "normal"))

def darrow(x1, y1, x2, y2, lbl=None, lx=0.02):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=26, lw=2.2, color=SLATE))
    if lbl:
        ax.text(max(x1, x2) + lx, (y1 + y2) / 2, lbl, fontsize=15.5, color=SLATE, va="center")

ax.text(0.5, 0.985, "Input: one student's interaction sequence", fontsize=16, ha="center", va="top", color=INK, weight="bold")
tok = [("skill 12", "right · fast"), ("skill 7", "wrong · slow"), ("MASKED", ""), ("skill 3", "right · med")]
tw, th = 0.215, 0.085
for i, (a, b) in enumerate(tok):
    x0 = 0.012 + i * (tw + 0.022)
    if a == "MASKED":
        dbox(x0, 0.865, tw, th, ["MASKED"], fc=INK, ec=INK, sizes=[15.5], colors=["white"], weights=["bold"])
    else:
        dbox(x0, 0.865, tw, th, [a, b], sizes=[15.5, 13.5], colors=[INK, SLATE], weights=["bold", "normal"])
ax.text(0.975, 0.9075, "...", fontsize=18, va="center")
darrow(0.5, 0.862, 0.5, 0.800, "mask skill and correctness\nat 15% of steps")
dbox(0.15, 0.685, 0.70, 0.112, ["StudentBERT", "transformer encoder"], fc="#FDECEC", ec=RED, lw=2.2,
     sizes=[19, 14.5], colors=[RED, SLATE], weights=["bold", "normal"])
darrow(0.5, 0.682, 0.5, 0.628)
dbox(0.06, 0.535, 0.88, 0.090, ["Pretraining: predict the masked skill AND correctness"],
     sizes=[15.5], weights=["bold"])
for i, ob in enumerate(["full", "skill-only", "correct-only"]):
    dbox(0.10 + i * 0.24, 0.448, 0.21, 0.062, [ob], fc="#F1F0EC", ec=SLATE, lw=1.3, sizes=[14])
ax.text(0.83, 0.479, "objective\nablation", fontsize=13.5, color=SLATE, va="center")
darrow(0.5, 0.442, 0.5, 0.372, "fine-tune on the target\n(25 students to the full split)")
tb = [("Knowledge tracing", "next answer right?"), ("Next-skill", "recommendation"), ("Early", "disengagement")]
for i, (a, b) in enumerate(tb):
    x0 = 0.018 + i * 0.330
    dbox(x0, 0.235, 0.305, 0.115, [a, b], sizes=[15, 13.5], weights=["bold", "normal"], colors=[INK, SLATE])
    ax.add_patch(FancyArrowPatch((0.5, 0.372), (x0 + 0.152, 0.355), arrowstyle="-|>",
                                 mutation_scale=20, lw=1.8, color=SLATE))
ax.text(0.5, 0.155, W("Scale: 9 pretrained encoders, 7 target datasets, hundreds of controlled "
                      "fine-tuning runs. Sources pretrain at full scale; targets are "
                      "subsampled to fixed budgets N.", 54),
        fontsize=16, ha="center", va="top", color=SLATE, linespacing=1.3)

# ================================================================ COLUMN 2
(c2a, c2b, c2c, c2d) = stack(1, [0.264, 0.194, 0.094, 0.163])

# ---- 4 Data table
cx, cy, cw, ct = card(*c2a, "4", "Methodology: 7 datasets, one schema")
rows = [
    ("ASSISTments 2017", "US math",       "1,708",   "102",   "4.32",  "skill"),
    ("EdNet KT1",        "TOEIC English", "441,997", "142",   "0.211", "correctness"),
    ("Junyi Academy",    "K-12 math",     "61,442",  "1,326", "0.066", "correctness"),
    ("Algebra 2005",     "Algebra tutor", "567",     "109",   "5.33",  "skill"),
    ("Bridge 2006",      "Algebra tutor", "1,130",   "492",   "2.79",  "skill"),
    ("ASSISTments 2009", "US math",       "3,119",   "123",   "0.325", "skill"),
    ("Algebra 2006-07",  "Algebra tutor", "1,310",   "484",   "2.41",  "skill"),
]
colx = [0.000, 0.300, 0.505, 0.660, 0.770, 0.880]
hdr  = ["Dataset", "Domain", "Students", "Skills", "PPS", "Regime"]
ha_  = ["left", "left", "right", "right", "right", "left"]
toff = [0, 0, 0.135, 0.095, 0.095, 0.005]
rh = 0.0214
yy = ct - 0.004
for j, h in enumerate(hdr):
    T(cx + colx[j] * cw + toff[j] * cw, yy, h, 15.5, weight="bold", ha=ha_[j])
yy -= 0.019
for i, r in enumerate(rows):
    if i % 2 == 0:
        fig.patches.append(Rectangle((cx - 0.002, yy - rh + 0.0055), cw + 0.004, rh,
                                     transform=fig.transFigure, fc="#F4F2EE", ec="none", zorder=1))
    for j, v in enumerate(r[:5]):
        T(cx + colx[j] * cw + toff[j] * cw, yy, v, 15, ha=ha_[j],
          weight=("bold" if j == 0 else "normal"))
    reg = r[5]
    col = RED if reg == "skill" else BLUE
    chw = 0.115 if reg == "skill" else 0.138
    box(cx + colx[5] * cw, yy - 0.0148, chw * cw, 0.0155,
        fc=("#FDECEC" if reg == "skill" else "#E8F0F7"), ec=col, lw=1.2, r=0.004, z=2)
    T(cx + (colx[5] + chw / 2) * cw, yy - 0.0068, reg, 12, color=col, weight="bold",
      ha="center", va="center", z=6)
    yy -= rh
T(cx, yy - 0.002, W("PPS proxy = median raw sequence length / skills, before the 512-step cap. "
                    "Regime = which objective transfers better (R3).", 62),
  16, color=SLATE, ls=1.25)
T(cx, yy - 0.030, W("One schema for all 7; split 80/10/10 by student; min 10 interactions; "
                    "511,273 retained sequences, 115.2M interactions.", 62),
  16, color=SLATE, ls=1.25)

# ---- 5 Evaluation protocol
cx, cy, cw, ct = card(*c2b, "5", "Evaluation plan")
ev = [
    "Metrics: knowledge-tracing AUC (0.5 = chance, 1.0 = perfect), next-skill top-1, early-disengagement AUC (bottom quartile of activity; prefix censoring keeps length out of the input).",
    "Budget-matched: every condition fine-tunes on the same number of target students (N).",
    "Seeds vary fine-tuning only, on one fixed checkpoint per source and objective; fine-tuning picks the best checkpoint by validation, pretraining the lowest training MLM loss. 3 seeds descriptive, 6 to 8 with paired bootstrap for confirmatory claims.",
    "Predict-before-test: predictions were written into the scripts before the runs. Algebra 2006-07 held, at +0.0186.",
]
yy = ct
for m in ev:
    T(cx + 0.0085, yy, W(m, 64), 16, ls=1.26)
    fig.patches.append(Rectangle((cx + 0.0012, yy - 0.0055), 0.0034, 0.0034,
                                 transform=fig.transFigure, fc=RED, ec="none"))
    yy -= 0.0117 * len(textwrap.wrap(m, 64)) + 0.0052

# ---- Reality check
cx, cy, cw, ct = card(*c2c, "6", "Baselines: how StudentBERT compares")
base = [
    "Beats or ties from-scratch training on all 7 datasets (ASSIST 2017 0.693 vs 0.670 AUC).",
    "A 2015 DKT still wins the 4 datasets added later, so architecture alone does not explain the improvement.",
]
yy = ct
for m in base:
    T(cx + 0.0085, yy, W(m, 62), 16, ls=1.22)
    fig.patches.append(Rectangle((cx + 0.0012, yy - 0.0055), 0.0034, 0.0034,
                                 transform=fig.transFigure, fc=RED, ec="none"))
    yy -= 0.0117 * len(textwrap.wrap(m, 62)) + 0.0040

# ---- References
cx, cy, cw, ct = card(*c2d, "", "References")
refs = [
    "Devlin J., Chang M.-W., Lee K., Toutanova K. (2019). BERT: pre-training of deep bidirectional transformers for language understanding. NAACL-HLT.",
    "Piech C. et al. (2015). Deep knowledge tracing. NeurIPS.",
    "Ghosh A., Heffernan N., Lan A. (2020). Context-aware attentive knowledge tracing. ACM SIGKDD.",
    "Choi Y. et al. (2020). Towards an appropriate query, key, and value computation for knowledge tracing (SAINT). ACM Learning at Scale.",
    "Choi Y. et al. (2020). EdNet: a large-scale hierarchical dataset in education. AIED.",
    "Stamper J. et al. (2010). KDD Cup 2010 educational data mining challenge data sets. PSLC DataShop.",
    "Feng M., Heffernan N., Koedinger K. (2009). Addressing the assessment challenge with an online system that tutors as it assesses. UMUAI.",
    "Patikorn T., Baker R., Heffernan N. (2020). ASSISTments longitudinal data mining competition special issue: a preface. Journal of Educational Data Mining 12(2).",
    "Shin D., Shim Y., Yu H., Lee S., Kim B., Choi Y. (2021). SAINT+: integrating temporal features for EdNet correctness prediction. LAK.",
    "Junyi Academy online learning activity dataset (2020). Kaggle.",
]
yy = ct - 0.001
for i, r in enumerate(refs):
    lines = textwrap.wrap(f"[{i+1}] {r}", 94)
    T(cx, yy, "\n".join(lines), 12.5, color=SLATE, ls=1.16)
    yy -= 0.0080 * len(lines) + 0.0014

# ================================================================ COLUMN 3
(c3a, c3b) = stack(2, [0.404, 0.328])

# ---- R1 low-data break-even
cx, cy, cw, ct = card(*c3a, "R1", "Pretraining pays off when data is scarce")
T(cx, ct, W("Next-skill prediction on ASSISTments 2017: which skill does a student practice next? "
            "Top-1 = the first guess is right. Fine-tuned on N = 25 to 1,000 students "
            "(4 conditions x 6 budgets x 3 seeds). In-domain = pretrained on more data "
            "from the same platform.", 62),
  16, ls=1.26, color=SLATE)
axh = 0.202
ax1 = mkax([cx + 0.0225, cy + 0.104, cw - 0.032, axh])
N = np.array([25, 50, 100, 200, 500, 1000])
series = {
    "scratch":            ([0.6577, 0.7511, 0.7664, 0.7717, 0.7773, 0.7855], "#6E6F72", "--", "o"),
    "in-domain pretrain": ([0.7498, 0.7651, 0.7727, 0.7784, 0.7887, 0.7975], RED, "-", "o"),
    "Junyi pretrain":     ([0.7435, 0.7627, 0.7729, 0.7794, 0.7910, 0.8000], BLUE, "-", "s"),
    "EdNet pretrain":     ([0.6917, 0.7465, 0.7649, 0.7743, 0.7862, 0.7975], LBLUE, "-", "^"),
}
for lab, (v, c, lsty, mk) in series.items():
    ax1.plot(N, np.array(v) * 100, lsty, color=c, lw=3.2, marker=mk, ms=9, label=lab, zorder=3)
ax1.set_xscale("log")
ax1.minorticks_off()
ax1.set_xticks(N); ax1.set_xticklabels([str(n) for n in N], fontsize=15)
ax1.tick_params(axis="y", labelsize=15)
ax1.set_ylabel("top-1 accuracy (%), axis truncated", fontsize=15)
ax1.set_xlabel("number of target students (log scale)", fontsize=16)
ax1.set_ylim(63, 82)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)
ax1.grid(axis="y", color="#E8E6E1", lw=1)
for xv in (50, 200):
    ax1.axvline(xv, color="#C9C6BF", lw=1.4, ls=":")
ax1.text(34, 81.3, "largest gains", fontsize=14.5, color=RED, ha="center", weight="bold")
ax1.text(100, 81.3, "modest", fontsize=14.5, color=SLATE, ha="center")
ax1.text(470, 81.3, "gap under 1.5 pts", fontsize=14.5, color=SLATE, ha="center")
ax1.annotate("", xy=(25, 74.98), xytext=(25, 65.77),
             arrowprops=dict(arrowstyle="<->", color=INK, lw=2))
ax1.text(28, 70.3, "+9.2 pts", fontsize=16, weight="bold", color=INK)
ax1.text(25, 64.0, "EdNet at N=25 is high\nvariance (std 4.6 pts)", fontsize=12.5, color=SLATE)
ax1.legend(fontsize=14, loc="lower right", frameon=False)

dg = [
    ("N = 25", "largest observed gain, +0.092 top-1 from in-domain, 3/3 seeds. The 442K EdNet source is least stable here"),
    ("N = 50 - 200", "+0.006 to +0.014, 3/3 seeds for in-domain and Junyi. EdNet is below scratch here"),
    ("N = 500 - 1,000", "best top-1 gap narrows to +0.011 to +0.015; scratch closes most of it"),
]
bw3 = (cw - 0.012) / 3
for i, (h, b) in enumerate(dg):
    bx = cx + i * (bw3 + 0.006)
    box(bx, cy + 0.004, bw3, 0.076, fc="#F4F2EE", ec=EDGE, r=0.005)
    T(bx + 0.004, cy + 0.0725, h, 15.5, weight="bold", color=RED)
    T(bx + 0.004, cy + 0.0570, W(b, 24), 13.5, ls=1.18)

# ---- R2 source choice
cx, cy, cw, ct = card(*c3b, "R2", "The largest source transferred best")
T(cx, ct, W("A different question after R1: the task switches to knowledge tracing, the budget is "
            "fixed, and the pretraining source is what varies. AUC gain over scratch on the "
            "ASSISTments 2017 target; budget cap 3,000, this target has 1,366 training students, "
            "so every condition uses its full split.", 60),
  16, ls=1.26, color=SLATE)
ax2 = mkax([cx + 0.054, cy + 0.146, cw - 0.076, 0.080])
src = ["EdNet source\n(442K students)", "In-domain ASSIST\n(1.7K students)", "Junyi source\n(61K students)"]
val = [0.0269, 0.0232, 0.0189]
cols = [RED, "#8A8B8F", BLUE]
yp = np.arange(len(src))[::-1]
ax2.barh(yp, val, height=0.62, color=cols)
for y0, v in zip(yp, val):
    ax2.text(v + 0.0006, y0, f"+{v:.4f}", fontsize=15.5, va="center", weight="bold")
ax2.set_yticks(yp); ax2.set_yticklabels(src, fontsize=15)
ax2.set_xlim(0, 0.0335)
ax2.set_xticks([0, 0.01, 0.02, 0.03]); ax2.tick_params(axis="x", labelsize=14)
ax2.set_xlabel("AUC gain vs scratch", fontsize=15)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
notes = [
    "The 442K-student EdNet source was best in all 3 source tests. On the EdNet target the closest skill-vocabulary match (ASSISTments) transfers worst, below scratch.",
    "Cross-dataset loading drops the skill embeddings: this compares corpus scale and granularity, not semantic similarity.",
    "Gains shrink toward 0 as target data grows; only in-domain EdNet survives (+0.0069 AUC, 6/6 seeds). ASSISTments 2017's full split is still low-resource.",
]
yy = cy + 0.108
for m in notes:
    T(cx + 0.0085, yy, W(m, 60), 16, ls=1.24)
    fig.patches.append(Rectangle((cx + 0.0012, yy - 0.0055), 0.0034, 0.0034,
                                 transform=fig.transFigure, fc=RED, ec="none"))
    yy -= 0.0117 * len(textwrap.wrap(m, 60)) + 0.0048

# ================================================================ COLUMN 4
(c4a, c4b, c4c) = stack(3, [0.388, 0.146, 0.194])

# ---- R3 objective reversal + causal flip
cx, cy, cw, ct = card(*c4a, "R3", "Skill-only or correct-only, by dataset")
T(cx, ct, W("EdNet is the fixed pretraining source; three objectives fine-tuned into every target, "
            "6 seeds per cell. Above 0: skill-only beats correct-only. The full objective has the "
            "highest mean on 5 of 7.", 60),
  16, ls=1.26, color=SLATE)

axs = mkax([cx + 0.036, cy + 0.154, cw - 0.052, 0.140])
pts = [  # (pps, skill_only - correct_only AUC, label, regime, dx, dy)
    (0.066, -0.0122, "Junyi",       "c", -0.012,  0.0028),
    (0.211, -0.0068, "EdNet",       "c", -0.045, -0.0037),
    (0.325,  0.0030, "ASSIST 2009", "s",  0.060, -0.0012),
    (2.414,  0.0186, "Algebra 06-07","s", -1.35,  0.0042),
    (2.790,  0.0106, "Bridge 2006", "s",  0.55,  -0.0030),
    (4.324,  0.0240, "ASSIST 2017", "s", -1.55,   0.0032),
    (5.330,  0.0228, "Algebra 2005","s",  0.40,  -0.0052),
]
axs.axhline(0, color="#B9B6AF", lw=1.6)
axs.text(8.9, 0.0022, "above 0: skill-only better", fontsize=11.5, color=SLATE, ha="right")
axs.text(8.9, -0.0060, "below 0: correct-only better", fontsize=11.5, color=SLATE, ha="right")
axs.axvspan(0.211, 0.325, color="#DEDBD4", alpha=0.7, zorder=0, gid="band")
for p, g, lab, reg, dx, dy in pts:
    c = RED if reg == "s" else BLUE
    axs.scatter([p], [g], s=340, color=c, zorder=4, edgecolor="white", lw=1.5)
    axs.annotate(lab, (p, g), xytext=(p + dx if abs(dx) > 0.2 else p * (1 + dx * 8), g + dy),
                 fontsize=13.5, color=INK, weight="bold")
axs.set_xscale("log")
axs.minorticks_off()
axs.set_xlim(0.045, 9.5); axs.set_ylim(-0.020, 0.031)
axs.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5])
axs.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "5"], fontsize=13.5)
axs.tick_params(axis="y", labelsize=13.5)
axs.set_xlabel("practice per skill (log scale)", fontsize=15)
axs.set_ylabel("skill-only minus correct-only\n(transfer AUC gap)", fontsize=14)
for s in ("top", "right"):
    axs.spines[s].set_visible(False)
axs.text(0.26, 0.0285, "observed regime gap;\nno dataset sampled here", fontsize=12.5,
         color=SLATE, ha="center", gid="band_caption")
axs.text(0.046, 0.024, "skill-driven", fontsize=13, color=RED, weight="bold")
axs.text(0.046, -0.018, "correctness-driven", fontsize=13, color=BLUE, weight="bold")

# causal flip mini-panel
axf = mkax([cx + 0.032, cy + 0.024, 0.082, 0.090])
fv  = [-0.0121, 0.0265]
err = np.array([[0.0091, 0.0012], [0.0088, 0.0015]])  # lower, upper rows
axf.bar([0, 1], fv, width=0.58, color=[BLUE, RED],
        yerr=err, capsize=6, error_kw=dict(lw=1.8, ecolor=INK))
axf.axhline(0, color="#B9B6AF", lw=1.4)
axf.set_xticks([0, 1])
axf.set_xticklabels(["K = 10\n(short)", "K = 512\n(model cap)"], fontsize=13)
axf.tick_params(axis="y", labelsize=12.5)
axf.set_ylim(-0.026, 0.033)
axf.set_ylabel("skill minus correct", fontsize=12)
for s in ("top", "right"):
    axf.spines[s].set_visible(False)

T(cx + 0.120, cy + 0.128, "Within-dataset truncation test", 15.5, weight="bold", color=RED)
caus = [
    "Keep each learner's most recent K interactions, skills (102) and students (1,708) held fixed: the preference reverses, skill-only at K=512, correct-only at K=10; CIs exclude 0 at both ends. A scratch control run to K=320 changes with K without the reversal.",
    "Prospective test passed: Algebra 2006-07, predicted skill-driven from pps 2.41, confirmed at +0.0186, CI [+0.0149, +0.0228], 6/6 seeds.",
]
yy = cy + 0.116
for m in caus:
    T(cx + 0.120, yy, W(m, 36), 13.5, ls=1.24)
    yy -= 0.0096 * len(textwrap.wrap(m, 36)) + 0.0036

# ---- R4 probe
cx, cy, cw, ct = card(*c4b, "R4", "Mechanism: what pretraining adds")
axp = mkax([cx + 0.044, cy + 0.027, 0.092, 0.076])
pl = ["ASSIST 2009", "Bridge 2006", "Algebra 2005", "EdNet", "Algebra 06-07", "ASSIST 2017", "Junyi"]
pv = [0.0389, 0.0314, 0.0282, 0.0143, 0.0111, 0.0051, 0.0044]
yp = np.arange(len(pl))[::-1]
axp.barh(yp, pv, height=0.62, color=RED)
for y0, v in zip(yp, pv):
    axp.text(v + 0.0007, y0, f"+{v:.4f}", fontsize=12.5, va="center")
axp.set_yticks(yp); axp.set_yticklabels(pl, fontsize=12.5)
axp.set_xlim(0, 0.056); axp.set_xticks([0, 0.02, 0.04])
axp.tick_params(axis="x", labelsize=12)
axp.set_xlabel("probe gain: pretrained minus scratch", fontsize=12.5)
for s in ("top", "right"):
    axp.spines[s].set_visible(False)
T(cx + 0.146, ct, W("Blank out every skill ID, then ask the frozen model to name them. The probe "
                    "beats scratch on all 7 targets (EdNet-full encoder). On the original 3, probe "
                    "scores rank sources the way transfer does (mean within-target rank correlation "
                    "0.83 vs 0.50 for LogME). Descriptive, and out of distribution.", 28), 13.5, ls=1.26)

# ---- Limitations + next
cx, cy, cw, ct = card(*c4c, "7", "Limitations and future work")
lim = [
    "Practice per skill orders all 7 datasets, but leave-one-dataset-out prediction holds for only 3 of 6 held-out sets: associated, not established. Next step: prospective tests on new mid-density datasets.",
    "PPS is a proxy: skills are defined differently per dataset, and it is measured before the 512-step cap, which lowers Bridge 2006 to 1.04 and Algebra 2006-07 to 1.06; the unsampled interval is 0.33 to 1.04.",
    "The truncation test keeps each learner's most recent K interactions, changing amount, horizon, skill mix and density together; the crossover is not localized. Size, compute and density stay confounded at n = 7.",
    "Scarce-data effect; embedding geometry failed a vocabulary control; disengagement mixed. 10.1% of next-skill labels are a placeholder, but per-skill averaging roughly doubles the N=25 gain (+0.18 macro top-1): not a frequent-skill artifact.",
]
nxt = []
yy = ct
for m in lim:
    T(cx + 0.0085, yy, W(m, 70), 14, ls=1.20)
    fig.patches.append(Rectangle((cx + 0.0012, yy - 0.0052), 0.0032, 0.0032,
                                 transform=fig.transFigure, fc=SLATE, ec="none"))
    yy -= 0.0096 * len(textwrap.wrap(m, 70)) + 0.0034
for m in nxt:
    T(cx + 0.0085, yy, W(m, 70), 14, ls=1.20)
    fig.patches.append(Rectangle((cx + 0.0012, yy - 0.0052), 0.0032, 0.0032,
                                 transform=fig.transFigure, fc=RED, ec="none"))
    yy -= 0.0096 * len(textwrap.wrap(m, 70)) + 0.0034

# ---------------------------------------------------------------- footer
box(MARG, 0.010, 1 - 2 * MARG, 0.033, fc="#EFEDE8", ec=EDGE, r=0.005)
T(MARG + 0.006, 0.0265, "Hanyu Dai  ·  dai.hany@northeastern.edu  ·  github.com/HY-D1/studentbert",
  15, va="center", weight="bold")
T(0.5, 0.0265, "Khoury College of Computer Sciences, Northeastern University Vancouver  ·  August 2026 Showcase",
  14.5, va="center", ha="center", color=SLATE)
T(1 - MARG - 0.006, 0.0265, "Computations: Northeastern Explorer HPC  ·  Tracking: Weights & Biases",
  14, va="center", ha="right", color=SLATE)

# ---------------------------------------------------------------- QA pass
# Catch text or chart artists that spill outside the card they belong to.
fig.canvas.draw()
rend = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()

def fbox(artist, tight=False):
    bb = artist.get_tightbbox(rend) if tight else artist.get_window_extent(rend)
    (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    return x0, y0, x1, y1

def inside(b, r, pad=0.0016):
    return (b[0] >= r[0] - pad and b[2] <= r[0] + r[2] + pad
            and b[1] >= r[1] - pad and b[3] <= r[1] + r[3] + pad)

violations = []
for t in fig.texts:
    b = fbox(t)
    if b[1] > 0.802 or b[3] < 0.044:      # header/banner above, footer strip below
        continue
    if not any(inside(b, r) for r in CARDS):
        violations.append(("text", repr(t.get_text()[:46]), tuple(round(v, 4) for v in b)))
for a in fig.axes:
    b = fbox(a, tight=True)
    if not any(inside(b, r, pad=0.003) for r in CARDS):
        violations.append(("axes", str(a.get_position()), tuple(round(v, 4) for v in b)))

# pairwise collisions: text over text, and text over a chart
def olap(a, b, tol=0.0012):
    return (min(a[2], b[2]) - max(a[0], b[0]) > tol) and (min(a[3], b[3]) - max(a[1], b[1]) > tol)

body_txt = [(t, fbox(t)) for t in fig.texts if fbox(t)[1] <= 0.802 and fbox(t)[3] >= 0.044]
for i in range(len(body_txt)):
    for j in range(i + 1, len(body_txt)):
        if olap(body_txt[i][1], body_txt[j][1]):
            violations.append(("text/text", repr(body_txt[i][0].get_text()[:28]),
                               repr(body_txt[j][0].get_text()[:28])))
for a in fig.axes:
    ab = fbox(a, tight=True)
    for t, tb in body_txt:
        if olap(ab, tb, tol=0.0008):
            violations.append(("text/chart", repr(t.get_text()[:34]),
                               tuple(round(v, 4) for v in ab)))

# ---- ink check: prove every chart region actually contains drawn pixels ------
import os
if os.environ.get("QA_DUMP"):
    import numpy as _np
    buf = _np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(int)
    Hpx, Wpx = buf.shape[:2]
    print("\n--- INK CHECK (nonwhite pixel share inside each chart rect) ---")
    for nm, a in zip(["pipeline", "R1 break-even", "R2 sources", "R3 scatter",
                      "R3 flip", "R4 probe"], fig.axes):
        p = a.get_position()
        x0, x1 = int(p.x0 * Wpx), int(p.x1 * Wpx)
        y0, y1 = int((1 - p.y1) * Hpx), int((1 - p.y0) * Hpx)
        reg = buf[y0:y1, x0:x1]
        nonwhite = (reg.sum(axis=2) < 720).mean()
        print(f"  {nm:14s} {100*nonwhite:5.1f}%  {'DRAWN' if nonwhite > 0.01 else 'EMPTY -- BUG'}")
    print("\n--- ZORDER (cards are zorder 1; axes must exceed it) ---")
    print("  axes zorders:", sorted({a.get_zorder() for a in fig.axes}))
    print("\n--- TEXT INVENTORY BY CARD ---")
    named = {(round(r[0],4), round(r[1],4)): i for i, r in enumerate(CARDS)}
    for i, r in enumerate(CARDS):
        items = []
        for t in fig.texts:
            b = fbox(t)
            if inside(b, r):
                items.append((-b[3], b[0], t.get_text()))
        items.sort()
        print(f"\n=== CARD {i} at x={r[0]:.3f} y={r[1]:.3f} ===")
        for _, _, txt in items:
            print("   |", txt.replace("\n", " / "))
    print("\n--- CHART DATA LABELS (axes-internal text) ---")
    for nm, a in zip(["pipeline", "R1", "R2", "R3 scatter", "R3 flip", "R4 probe"], fig.axes):
        tl = [t.get_text() for t in a.texts if t.get_text().strip()]
        if tl:
            print(f"  [{nm}]", " | ".join(tl[:24]))
    print("\n--- HEADER / BANNER / FOOTER TEXT ---")
    for t in fig.texts:
        b = fbox(t)
        if b[3] > 0.800 or b[1] < 0.050:
            print("   |", t.get_text().replace("\n", " / "))

if violations:
    print(f"QA: {len(violations)} artist(s) outside their card")
    for v in violations:
        print("   ", *v)
else:
    print("QA: no overflow, all text and charts inside their cards")

# ---------------------------------------------------------------- export for
# editable formats: geometry JSON (native shapes/text) + chart crops as PNG
if os.environ.get("EXPORT"):
    import json
    from matplotlib.colors import to_hex

    def hx(c):
        return to_hex(c).lstrip("#").upper() if c is not None else None

    def opaque(a):
        return None if (isinstance(a, tuple) and len(a) == 4 and a[3] == 0) else a

    shapes = []
    for p in fig.patches:
        bb = p.get_window_extent(rend)
        (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        shapes.append(dict(
            kind="round" if isinstance(p, FancyBboxPatch) else "rect",
            x=x0, y=y1, w=x1 - x0, h=y1 - y0,
            fc=hx(opaque(p.get_facecolor())), ec=hx(opaque(p.get_edgecolor())),
            lw=float(p.get_linewidth()), z=float(p.get_zorder()),
            r=float(getattr(getattr(p, "get_boxstyle", lambda: None)() or object(),
                            "rounding_size", 0.0) or 0.0),
            alpha=(p.get_alpha() if p.get_alpha() is not None else 1.0)))

    texts = []
    for t in fig.texts:
        x0, y0, x1, y1 = fbox(t)
        texts.append(dict(text=t.get_text(), x=x0, y=y1, w=x1 - x0, h=y1 - y0,
                          size=float(t.get_fontsize()), color=hx(t.get_color()),
                          bold=(str(t.get_fontweight()) == "bold"),
                          ha=t.get_ha(), ls=float(t._linespacing)))

    charts, names = [], ["pipeline", "r1_break_even", "r2_sources",
                         "r3_regime_scatter", "r3_truncation_flip", "r4_probe"]
    for nm, a in zip(names, fig.axes):
        bb = a.get_tightbbox(rend)
        (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        pad = 0.0015
        charts.append(dict(name=nm, x=x0 - pad, y=y1 + pad,
                           w=(x1 - x0) + 2 * pad, h=(y1 - y0) + 2 * pad))

    json.dump(dict(fig=[FW, FH], paper=hx(PAPER), shapes=shapes, texts=texts,
                   charts=charts), open("poster_layout.json", "w"), indent=1)

    # 300 dpi crops of each chart region, transparent so cards show through
    DPI = 300
    fig.savefig("_full.png", dpi=DPI, transparent=True)
    from PIL import Image
    im = Image.open("_full.png")
    Wpx, Hpx = im.size
    for c in charts:
        px = (int(c["x"] * Wpx), int((1 - c["y"]) * Hpx),
              int((c["x"] + c["w"]) * Wpx), int((1 - c["y"] + c["h"]) * Hpx))
        im.crop(px).save(f"chart_{c['name']}.png")
    print(f"exported poster_layout.json + {len(charts)} chart PNGs at {DPI} dpi")

# ---------------------------------------------------------------- save
fig.savefig("poster.svg", facecolor=PAPER)   # fully editable vector (Illustrator/Inkscape)
fig.savefig("poster.pdf", facecolor=PAPER)
fig.savefig("poster_preview.png", dpi=68, facecolor=PAPER)
print("saved poster.pdf + poster_preview.png")
