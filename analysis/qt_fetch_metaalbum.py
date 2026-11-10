from __future__ import annotations

"""Download the Meta-Album micro datasets behind the Quick-Tune micro tasks (stdlib only).

The task list comes from the Quick-Tune meta-dataset (the keys of eval_top1.json, such as
mtlbm/micro/set0/BCT). Each task maps to the OpenML dataset Meta_Album_<code>_Micro, whose id comes
from the OpenML web interface; its image archive, <code>_Micro.zip, is listed and downloaded from
OpenML's file store (data.openml.org) with urllib, which uses the cluster proxy from the
environment (the openml package's own download does not, and timed out). Each archive is saved,
hashed and extracted under --root. Nothing is scored here. The report gives, per task, the OpenML
id, the archive size and md5, the label file found and its rows and classes, and the image count.

  python analysis/qt_fetch_metaalbum.py --quicktune /projects/algl/dai.hany/quicktune \
      --root /projects/algl/dai.hany/quicktune/metaalbum_micro \
      --out /projects/algl/dai.hany/quicktune/metaalbum_micro_report
"""

import argparse
import collections
import csv
import http.client
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

API = "https://www.openml.org/api/v1/json/data/list/data_name/{name}"
STORE = "https://data.openml.org/datasets"
S3 = "{http://s3.amazonaws.com/doc/2006-03-01/}"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def get(url: str, timeout: int = 120, tries: int = 6) -> bytes:
    # The cluster proxy sometimes refuses a tunnel with 503 or drops a transfer partway
    # (IncompleteRead); retry with growing waits. A file is written only after a full read.
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError,
                TimeoutError) as e:
            if i == tries - 1:
                sys.exit(f"ABORT: {url} failed {tries} times, last error: {e}")
            wait = 15 * 2 ** i
            print(f"retry {i + 1} of {tries - 1} in {wait} s after: {e}", flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


def openml_id(name: str) -> int:
    data = json.loads(get(API.format(name=name)))
    ids = [d["did"] for d in data["data"]["dataset"] if d["status"] == "active"]
    if len(ids) != 1:
        sys.exit(f"ABORT: {name} matches {len(ids)} active OpenML datasets")
    return int(ids[0])


def archive_key(did: int, code: str) -> tuple[str, int]:
    prefix = f"{did // 10000:04d}/{did}"
    xml = ET.fromstring(get(f"{STORE}?list-type=2&max-keys=100&prefix={prefix}"))
    keys = {c.find(f"{S3}Key").text: int(c.find(f"{S3}Size").text)
            for c in xml.iter(f"{S3}Contents")}
    want = f"{prefix}/{code}_Micro.zip"
    if want not in keys:
        sys.exit(f"ABORT: no {want} in the OpenML store; found {sorted(keys)}")
    return want, keys[want]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quicktune", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="fetch only the first N tasks")
    a = ap.parse_args(argv)
    tasks = sorted(json.load((Path(a.quicktune) / "micro" / "micro" / "eval_top1.json").open()))
    if len(tasks) != 30:
        sys.exit(f"ABORT: {len(tasks)} micro tasks, expected 30")
    if a.limit:
        tasks = tasks[:a.limit]
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in tasks:
        code = t.rsplit("/", 1)[1]
        did = openml_id(f"Meta_Album_{code}_Micro")
        key, size = archive_key(did, code)
        zpath = root / f"{code}_Micro.zip"
        if not zpath.is_file() or zpath.stat().st_size != size:
            zpath.write_bytes(get(f"{STORE}/{key}", timeout=600))
        if zpath.stat().st_size != size:
            sys.exit(f"ABORT: {zpath} has {zpath.stat().st_size} bytes, the store lists {size}")
        dest = root / code
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest)
        images = sorted(p for p in dest.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted(dest.rglob("labels.csv"))
        n_rows, classes, per_class = 0, 0, set()
        if len(labels) == 1:
            lab = list(csv.DictReader(labels[0].open()))
            n_rows = len(lab)
            col = "CATEGORY" if lab and "CATEGORY" in lab[0] else ""
            counts = collections.Counter(r[col] for r in lab) if col else collections.Counter()
            classes, per_class = len(counts), set(counts.values())
        rows.append({"task": t, "openml_id": did, "archive": zpath.name, "bytes": size,
                     "md5": hashlib.md5(zpath.read_bytes()).hexdigest(),
                     "labels_file": labels[0].relative_to(root).as_posix() if labels else "",
                     "label_rows": n_rows, "classes": classes,
                     "images_per_class": sorted(per_class), "images": len(images)})
        print(f"{t}: id {did}, {size} bytes, labels {rows[-1]['labels_file'] or 'NOT FOUND'} "
              f"({n_rows} rows, {classes} classes), images {len(images)}", flush=True)
    out = Path(a.out).with_suffix(".json")
    out.write_text(json.dumps(rows, indent=1))
    # Micro is 40 images per class; most tasks have 20 classes, a few releases 19 (DOG, PNU).
    bad = [r["task"] for r in rows if r["images"] != r["label_rows"]
           or r["images_per_class"] != [40] or r["classes"] < 19]
    print("classes per task: " + ", ".join(f"{r['task'].rsplit('/', 1)[1]} {r['classes']}"
                                           for r in rows))
    print(f"tasks failing images = labels, 40 per class, at least 19 classes: {bad or 'none'}")
    print(f"wrote {out.name} md5 {hashlib.md5(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
