#!/usr/bin/env python3
"""Deep layout audit for the poster. Catches what the in-build QA cannot see:
chart-vs-chart overlap, chart-vs-shape overlap, and collisions between labels
and data *inside* each chart."""
import runpy
import numpy as np

ns = runpy.run_path("make_poster.py")
fig, CARDS = ns["fig"], ns["CARDS"]
fig.canvas.draw()
rend = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
NAMES = ["pipeline", "R1 break-even", "R2 sources", "R3 scatter", "R3 flip", "R4 probe"]


def fb(artist, tight=False):
    bb = artist.get_tightbbox(rend) if tight else artist.get_window_extent(rend)
    if bb is None:
        return None
    (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    return x0, y0, x1, y1


def olap(a, b, tol=0.0):
    if a is None or b is None:
        return 0.0
    ox = min(a[2], b[2]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[1], b[1])
    return min(ox, oy) if (ox > tol and oy > tol) else 0.0


def IN(x):   # figure fraction -> inches (for readable reporting)
    return x * 36


issues = []

# ---- 1. chart bounding boxes must not overlap each other -------------------
boxes = [(n, fb(a, True)) for n, a in zip(NAMES, fig.axes)]
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        o = olap(boxes[i][1], boxes[j][1], 0.0008)
        if o:
            issues.append(f"CHART/CHART  {boxes[i][0]} overlaps {boxes[j][0]} by {IN(o):.2f} in")

# ---- 2. charts must clear their card edges by a visible margin -------------
for n, b in boxes:
    home = None
    for c in CARDS:
        inside_x = b[0] >= c[0] - 0.004 and b[2] <= c[0] + c[2] + 0.004
        inside_y = b[1] >= c[1] - 0.004 and b[3] <= c[1] + c[3] + 0.004
        if inside_x and inside_y:
            home = c
            gaps = {"left": b[0] - c[0], "right": c[0] + c[2] - b[2],
                    "bottom": b[1] - c[1], "top": c[1] + c[3] - b[3]}
            tight = {k: IN(v) for k, v in gaps.items() if v < 0.0045}
            if tight:
                issues.append(f"TIGHT MARGIN {n}: " +
                              ", ".join(f"{k} {v:.2f} in" for k, v in tight.items()))
            break
    if home is None:
        issues.append(f"CHART ADRIFT {n} is not fully inside any card")

# ---- 3. charts must not sit under a non-card shape ------------------------
cardset = {(round(c[0], 5), round(c[1], 5)) for c in CARDS}
for p in fig.patches:
    pb = fb(p)
    if pb is None:
        continue
    if (round(pb[0], 5), round(pb[1], 5)) in cardset:
        continue                                     # this is a card
    if (pb[2] - pb[0]) > 0.5:
        continue                                     # banner / footer strip
    for n, b in boxes:
        o = olap(pb, b, 0.0008)
        if o:
            issues.append(f"CHART/SHAPE  {n} overlaps a {IN(pb[2]-pb[0]):.1f}x"
                          f"{IN(pb[3]-pb[1])*24/36:.1f} in shape by {IN(o):.2f} in")

# ---- 4. inside each chart: label vs label ---------------------------------
for n, a in zip(NAMES, fig.axes):
    items = [(t.get_text(), fb(t)) for t in a.texts if t.get_text().strip()]
    items += [(f"xtick {t.get_text()}", fb(t)) for t in a.get_xticklabels() if t.get_text()]
    items += [(f"ytick {t.get_text()}", fb(t)) for t in a.get_yticklabels() if t.get_text()]
    for lbl, art in (("xlabel", a.xaxis.label), ("ylabel", a.yaxis.label), ("title", a.title)):
        if art.get_text().strip():
            items.append((lbl + " " + art.get_text()[:22], fb(art)))
    leg = a.get_legend()
    if leg is not None:
        items.append(("legend", fb(leg)))
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            o = olap(items[i][1], items[j][1], 0.0006)
            if o:
                issues.append(f"LABEL/LABEL  [{n}] {items[i][0][:26]!r} vs "
                              f"{items[j][0][:26]!r} by {IN(o):.2f} in")

# ---- 5. inside each chart: label vs plotted data --------------------------
for n, a in zip(NAMES, fig.axes):
    if n == "pipeline":
        continue                                     # the diagram is all shapes by design
    pts = []
    for ln in a.lines:
        d = ln.get_transform().transform(ln.get_xydata())
        pts.append(d)
    for col in a.collections:                        # scatter points
        off = col.get_offsets()
        if len(off):
            pts.append(col.get_offset_transform().transform(off))
    bars = [(p.get_gid(), fb(p)) for p in a.patches]
    P = np.vstack(pts) if pts else np.empty((0, 2))
    if len(P):
        P = inv.transform(P)
    for t in a.texts:
        if not t.get_text().strip():
            continue
        b = fb(t)
        hit = ((P[:, 0] > b[0]) & (P[:, 0] < b[2]) &
               (P[:, 1] > b[1]) & (P[:, 1] < b[3])).sum() if len(P) else 0
        if hit:
            issues.append(f"LABEL/DATA   [{n}] {t.get_text()[:30]!r} sits on {hit} plotted point(s)")
        for gid, bb in bars:
            if gid == "band" and t.get_gid() == "band_caption":
                continue                             # the caption labels the band on purpose
            o = olap(b, bb, 0.0010)
            if o:
                what = "the shaded band" if gid == "band" else "a bar"
                issues.append(f"LABEL/BAR    [{n}] {t.get_text()[:30]!r} overlaps {what} by {IN(o):.2f} in")

# ---- 6. figure text must be fully inside a shape or fully outside it -------
small = []
for p in fig.patches:
    pb = fb(p)
    if pb is None or (pb[2] - pb[0]) > 0.5:
        continue
    if (round(pb[0], 5), round(pb[1], 5)) in cardset:
        continue
    small.append(pb)
bodytxt = []
for t in fig.texts:
    b = fb(t)
    if b is None or not t.get_text().strip():
        continue
    if b[1] > 0.802 or b[3] < 0.044:
        continue
    bodytxt.append((t.get_text(), b))
for txt, b in bodytxt:
    for pb in small:
        o = olap(b, pb, 0.0006)
        if not o:
            continue
        inside = (b[0] >= pb[0] - 0.0012 and b[2] <= pb[2] + 0.0012 and
                  b[1] >= pb[1] - 0.0012 and b[3] <= pb[3] + 0.0012)
        if not inside:
            issues.append(f"TEXT/SHAPE   {txt[:30]!r} straddles a shape edge by {IN(o):.2f} in")

# ---- 7. adjacent text blocks should not be cramped ------------------------
for i in range(len(bodytxt)):
    for j in range(i + 1, len(bodytxt)):
        a_, b_ = bodytxt[i][1], bodytxt[j][1]
        if min(a_[2], b_[2]) - max(a_[0], b_[0]) <= 0:
            continue                                  # different columns
        gap = max(a_[1] - b_[3], b_[1] - a_[3])
        if 0 <= gap < 0.0007:
            issues.append(f"CRAMPED      {bodytxt[i][0][:22]!r} / {bodytxt[j][0][:22]!r} "
                          f"gap {gap*24:.03f} in")

print(f"\n=== DEEP LAYOUT AUDIT: {len(issues)} issue(s) ===")
for s in issues:
    print("  *", s)
if not issues:
    print("  clean")
