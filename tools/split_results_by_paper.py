from __future__ import annotations

"""Split RESULTS.md by paper, verbatim, without changing it (Prof. Hazra, Nov 2026).

RESULTS.md stays the single record: its sections are cited by md5 and pinned by the paper asset
scripts, so they are never moved or relabelled. This script copies each section, unchanged, into
one of four derived files, by the explicit table below, and writes an index of what went where:

  PAPER_B_RESULTS.md  original reliable-transfer results, estimator comparisons, reversal
                      pipeline, NLP cross-check, fresh-seed validation, warning signal, PARC
                      (12.3 to 12.21, 13)
  PAPER_C_RESULTS.md  trajectories, compute, Quick-Tune, multi-fidelity and selective trust
                      (12.22 on)
  SHARED_RESULTS.md   methods, datasets and the transfer benchmark both papers rest on
                      (0, 0.1, 12, 12.1)
  OTHER_RESULTS.md    sections of the other papers (Paper A, LAK, the TLT paper; 1 to 11, 12.2)

It stops if any section is unassigned or assigned twice, or if the sections do not reassemble
RESULTS.md byte for byte, so a new section must be given a paper before the files can be rebuilt.

  python tools/split_results_by_paper.py --results RESULTS.md --out .
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

FILES = {"B": "PAPER_B_RESULTS.md", "C": "PAPER_C_RESULTS.md", "S": "SHARED_RESULTS.md",
         "O": "OTHER_RESULTS.md"}
TITLES = {"B": "Paper B: reliable transfer guidance (estimators, warning signal, fresh-seed "
               "validation)",
          "C": "Paper C: trajectories, compute, Quick-Tune, multi-fidelity and selective trust",
          "S": "Shared by Papers B and C: methods, datasets and the transfer benchmark",
          "O": "Other papers (Paper A, LAK, TLT): sections outside Papers B and C"}
# Section number (from its heading) -> paper. "head" is the text before the first heading.
ASSIGN = {"head": "S", "0": "S", "0.1": "S",
          "1": "O", "1.1": "O", "2": "O", "2.1": "O", "2.2": "O", "3": "O", "3.1": "O",
          "3.2": "O", "3.3": "O", "4": "O", "5": "O", "6": "O", "6.1": "O", "6.2": "O",
          "7": "O", "8": "O", "8.1": "O", "8.2": "O", "9": "O", "10": "O", "11": "O",
          "11.1": "O", "12": "S", "12.1": "S", "12.2": "O",
          **{f"12.{i}": "B" for i in range(3, 22)},
          **{f"12.{i}": "C" for i in range(22, 29)},
          "13": "B", "13.1": "B"}
HEADING = re.compile(r"^(#{2,3}) (\d+(?:\.\d+)?)[. ]", re.M)


def md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def sections(text: str) -> list[tuple[str, str]]:
    starts = [(m.start(), m.group(2)) for m in HEADING.finditer(text)]
    out = [("head", text[:starts[0][0]])] if starts and starts[0][0] > 0 else []
    for i, (pos, num) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        out.append((num, text[pos:end]))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    src = Path(a.results)
    raw = src.read_bytes()
    text = raw.decode("utf-8")
    secs = sections(text)
    nums = [n for n, _ in secs]
    dup = sorted({n for n in nums if nums.count(n) > 1})
    unassigned = [n for n in nums if n not in ASSIGN]
    unused = sorted(set(ASSIGN) - set(nums))
    if dup or unassigned or unused:
        sys.exit(f"ABORT: duplicate sections {dup}, unassigned {unassigned}, "
                 f"assigned but absent {unused}; nothing written")
    if "".join(body for _, body in secs) != text:
        sys.exit("ABORT: the sections do not reassemble RESULTS.md; nothing written")
    src_md5 = md5(raw)
    out_dir = Path(a.out)
    index = []
    for key, name in FILES.items():
        part = [(n, b) for n, b in secs if ASSIGN[n] == key]
        hdr = [f"# {TITLES[key]}", "",
               f"Generated from RESULTS.md (md5 {src_md5}) by tools/split_results_by_paper.py. "
               "Do not edit: RESULTS.md stays the record, and every section below is copied from "
               "it unchanged. Sections, with the md5 of their text:", ""]
        hdr += [f"- {n}: {md5(b.encode('utf-8'))}" for n, b in part]
        body = "\n".join(hdr) + "\n\n---\n\n" + "".join(b for _, b in part)
        path = out_dir / name
        path.write_text(body, encoding="utf-8")
        index.append(f"{name}: {len(part)} sections, md5 {md5(path.read_bytes())}")
        print(f"{name}: sections {', '.join(n for n, _ in part)}")
    print(f"source RESULTS.md md5 {src_md5}")
    print("\n".join(index))


if __name__ == "__main__":
    main()
