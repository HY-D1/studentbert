from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Why this exists: analysis/similarity_baseline.py rounded its report display with float
# formatting, which rounds the binary value half to even (0.00015 prints as 0.0001). The project
# rule is Decimal, rounded once from the full-precision value, half away from zero. Only the
# _report.md display changes; the _cells.tsv and _summary.tsv files keep full precision and are
# untouched. Adds a test that pins the tie case. Open item from the 2026-09-28 hand-off; Harry,
# 2026-09-29.

SCRIPT = Path("analysis/similarity_baseline.py")
TESTS = Path("tests/test_similarity_baseline.py")

EDITS = {
    SCRIPT: [
        ("from collections import defaultdict\nfrom pathlib import Path\n",
         "from collections import defaultdict\nfrom decimal import ROUND_HALF_UP, Decimal\n"
         "from pathlib import Path\n",
         "import Decimal"),
        ('def fmt(x) -> str:\n    return f"{x:.4f}" if isinstance(x, float) else str(x)\n',
         "def fmt(x) -> str:\n"
         "    # Decimal from the shortest repr, half away from zero, rounded once; float\n"
         "    # formatting would round the binary value half to even (0.00015 -> 0.0001).\n"
         "    if isinstance(x, float):\n"
         '        d = Decimal(repr(x)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)\n'
         "        return str(d)\n"
         "    return str(x)\n",
         "fmt rounds in Decimal, half away from zero"),
    ],
    TESTS: [
        ("\n\ndef main() -> int:\n",
         "\n\ndef test_report_rounding_is_decimal_half_up():\n"
         '    assert sb.fmt(0.00015) == "0.0002"\n'
         '    assert sb.fmt(-0.00015) == "-0.0002"\n'
         '    assert sb.fmt(0.1234) == "0.1234"\n'
         '    assert sb.fmt(7) == "7"\n'
         "\n\ndef main() -> int:\n",
         "test pins the half-up tie"),
    ],
}


def apply(text: str, old: str, new: str, why: str) -> str:
    # Insertion-style edits keep their anchor, so "old in text" stays true after patching;
    # deciding by containment direction keeps re-runs as no-ops.
    if new in text and (old not in text or old in new):
        print(f"skip (already applied): {why}")
        return text
    n = text.count(old)
    if n != 1:
        sys.exit(f"ABORT ({why}): anchor matched {n} times, expected 1; nothing written")
    print(f"ok: {why}")
    return text.replace(old, new, 1)


def main() -> None:
    staged = {}
    for path, edits in EDITS.items():
        src = path.read_text(encoding="utf-8")
        out = src
        for old, new, why in edits:
            out = apply(out, old, new, f"{path}: {why}")
        staged[path] = (src, out)
    for path, (src, out) in staged.items():
        if out != src:
            path.write_text(out, encoding="utf-8")
        print("md5", hashlib.md5(path.read_bytes()).hexdigest(), path)


if __name__ == "__main__":
    main()
