// Build an editable 36in x 24in PowerPoint poster from poster_layout.json.
// Card shapes and every text block become native PowerPoint objects; the six
// charts are placed as 300 dpi images (re-run make_poster.py to regenerate them).
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const L = JSON.parse(fs.readFileSync("poster_layout.json", "utf8"));
const [FW, FH] = L.fig;

const X = f => f * FW;              // figure fraction -> inches from left
const Y = f => (1 - f) * FH;        // figure fraction (top edge) -> inches from top

const pres = new pptxgen();
pres.defineLayout({ name: "POSTER", width: FW, height: FH });
pres.layout = "POSTER";             // must be set before addSlide
pres.author = "Hanyu Dai";
pres.title = "StudentBERT poster";

const slide = pres.addSlide();
slide.background = { color: L.paper };

// ---- 1. shapes, in zorder then insertion order (PowerPoint paints in order) --
const shapes = L.shapes.map((s, i) => ({ ...s, i }))
                       .sort((a, b) => a.z - b.z || a.i - b.i);
for (const s of shapes) {
  const w = s.w * FW, h = s.h * FH;
  const opts = { x: X(s.x), y: Y(s.y), w, h };
  if (s.fc) opts.fill = { color: s.fc };
  else opts.fill = { type: "none" };
  if (s.ec && s.lw > 0) {
    opts.line = { color: s.ec, width: s.lw * 0.75 };          // px -> pt
  } else {
    // width:0 alone leaves the theme outline in place, which drew a dark rule
    // across every table row; an explicit fully transparent line kills it.
    opts.line = { color: s.fc || L.paper, width: 0.25, transparency: 100 };
  }
  if (s.alpha < 1) opts.transparency = Math.round((1 - s.alpha) * 100);
  if (s.r > 0) {
    // OOXML roundRect adj is a fraction of half the shorter side
    const radIn = s.r * FW;
    opts.rectRadius = Math.min(0.5, (2 * radIn) / Math.min(w, h));
    slide.addShape(pres.ShapeType.roundRect, opts);
  } else {
    slide.addShape(pres.ShapeType.rect, opts);
  }
}

// ---- 2. chart images -------------------------------------------------------
for (const c of L.charts) {
  slide.addImage({
    path: `chart_${c.name}.png`,
    x: X(c.x), y: Y(c.y), w: c.w * FW, h: c.h * FH,
  });
}

// ---- 3. text, using each block's measured box so nothing re-wraps ----------
for (const t of L.texts) {
  if (!t.text.trim()) continue;          // skip the empty card-number placeholders
  const w = t.w * FW, h = t.h * FH;
  const padW = Math.max(0.06, w * 0.04);      // slack for Arial vs Liberation Sans
  const padH = Math.max(0.10, h * 0.22);
  // the slack must grow away from the anchored edge, or the text visibly shifts
  const dx = t.ha === "left" ? 0 : t.ha === "right" ? -padW : -padW / 2;
  slide.addText(t.text, {
    x: X(t.x) + dx,
    y: Y(t.y) - padH / 2,
    w: w + padW,
    h: h + padH,
    fontFace: "Arial",                        // metric-compatible with Liberation Sans
    fontSize: t.size,
    color: t.color,
    bold: t.bold,
    align: t.ha,
    valign: "middle",
    margin: 0,
    wrap: false,                              // lines are already broken as rendered
    lineSpacingMultiple: Math.round((t.ls / 1.2) * 100) / 100,
    isTextBox: true,
  });
}

pres.writeFile({ fileName: "studentbert_poster_editable.pptx" }).then(f => {
  console.log(`wrote ${f}: ${shapes.length} shapes, ${L.charts.length} chart images, ${L.texts.length} text boxes`);
});
