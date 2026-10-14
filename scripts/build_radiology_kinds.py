#!/usr/bin/env python
"""The kinds of radiology words, from the RadLex® ontology, for the reader's base vocabulary (14 October 2026).

Why: the reader's vocabulary of word kinds comes from NCIt, which does not know many words of radiology as
written ("CT" is a gene, a protein and a procedure at once; "radiograph" and "view" have no usable kind), so
"CT brain" is read as a plain structure. RadLex is the radiology lexicon of the RSNA: CT and MRI are imaging
modalities, "radiograph" a procedure, "opacity" an imaging observation.

How: the whole RadLex (BioPortal CSV export, ``BIOPORTAL_API_KEY``), each class with its labels, synonyms and
parents. The kind of a class is read from the branch it sits under (``BRANCHES``). A word or a name gets a
kind only when every RadLex class that has it as label or synonym gives the same kind (an exact
correspondence; otherwise it is left out and listed as a conflict).

What the reader does with it (``chunk_lattice.BlockMemory.load``): a RadLex kind fills a word NCIt has no kind
for, or replaces an NCIt "conceptual" (too wide to act on); a name of several words not remembered from NCIt is
added. It is one more piece of evidence in the lattice and the reader, which still integrate everything.

Licence: the RadLex License (RSNA) allows derivative material with notice; this table is derived and says so
("based on the RadLex® Ontology developed at the Radiological Society of North America"); it is not RadLex
and is not called RadLex.

    BIOPORTAL_API_KEY=... python scripts/build_radiology_kinds.py --out data/linking/radiology_kinds.json
    python scripts/build_radiology_kinds.py --csv RADLEX.csv.gz --out data/linking/radiology_kinds.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

DOWNLOAD = "https://data.bioontology.org/ontologies/RADLEX/download?download_format=csv"
ATTRIBUTION = ("This table is based on the RadLex® Ontology developed at the Radiological Society of North "
               "America (RSNA), used under the RadLex License; it is derived material, not RadLex itself.")

# RadLex branches (labels of ancestor classes, lower case) -> the kinds of the reader; first match wins.
BRANCHES = (
    ("device", "device"), ("implant", "device"),
    ("imaging modality", "procedure"), ("procedure", "procedure"), ("imaging technique", "procedure"),
    ("imaging observation", "disease"), ("clinical finding", "disease"), ("pathophysiologic", "disease"),
    ("anatomical entity", "anatomy"),
    ("property", "property"),
)
_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def _fold(text: str) -> str:
    return " ".join(_TOKEN.findall(str(text or "").lower()))


def fetch_csv(key: str, attempts: int = 4) -> bytes:
    last = ""
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(DOWNLOAD, headers={"Authorization": f"apikey token={key}"})
            with urllib.request.urlopen(req, timeout=600) as r:  # noqa: S310 (fixed https URL)
                return r.read()
        except OSError as error:
            last = f"{type(error).__name__}: {error}"
        print(f"attempt {attempt}/{attempts}: {last}", file=sys.stderr)
        time.sleep(20 * attempt)
    raise SystemExit(f"RadLex not downloaded: {last}")


def read_classes(data: bytes) -> dict[str, dict]:
    """Class id -> labels (label and synonyms, folded), parents, obsolete."""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    text = data.decode("utf-8-sig", "replace")
    reader = csv.DictReader(io.StringIO(text))
    cols = {c.lower(): c for c in reader.fieldnames or []}

    def col(*names):
        return next((cols[n] for n in names if n in cols), None)

    cid, label = col("class id", "id"), col("preferred label", "label", "preferred name")
    syn, parents, obsolete = col("synonyms", "synonym"), col("parents", "parent"), col("obsolete")
    if not cid or not label:
        raise SystemExit(f"unexpected RadLex CSV columns: {reader.fieldnames}")
    classes = {}
    for row in reader:
        if obsolete and str(row.get(obsolete, "")).strip().lower() in ("true", "1"):
            continue
        names = [row.get(label, "")] + [s for s in str(row.get(syn, "") or "").split("|") if s]
        classes[row[cid]] = {
            "label": row.get(label, ""),
            "names": sorted({_fold(n) for n in names if _fold(n)}),
            "parents": [p for p in str(row.get(parents, "") or "").split("|") if p],
        }
    return classes


def kind_of(cid: str, classes: dict, memo: dict) -> str:
    """The kind from the branch above the class: the first branch label met going up."""
    if cid in memo:
        return memo[cid]
    memo[cid] = ""  # guards cycles
    seen, stack, labels = set(), [cid], []
    while stack:
        c = stack.pop()
        if c in seen or c not in classes:
            continue
        seen.add(c)
        labels.append(_fold(classes[c]["label"]))
        stack.extend(classes[c]["parents"])
    for needle, kind in BRANCHES:
        if any(needle in lab for lab in labels[1:] + labels[:1]):
            memo[cid] = kind
            break
    return memo[cid]


def build(classes: dict) -> dict:
    memo: dict[str, str] = {}
    by_name: dict[str, Counter] = defaultdict(Counter)
    for cid, c in classes.items():
        kind = kind_of(cid, classes, memo)
        if not kind:
            continue
        for name in c["names"]:
            by_name[name][kind] += 1
    heads, names, conflicts = {}, {}, []
    for name, kinds in sorted(by_name.items()):
        if len(kinds) > 1:
            conflicts.append(name)
            continue
        kind = next(iter(kinds))
        if " " in name:
            names[name] = kind
        else:
            heads[name] = [kind, 1.0, 0]
    return {"heads": heads, "names": names, "conflicts": conflicts}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", type=Path, help="a RadLex CSV export already downloaded (else BioPortal)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.csv:
        data = args.csv.read_bytes()
    else:
        key = os.environ.get("BIOPORTAL_API_KEY")
        if not key:
            raise SystemExit("set BIOPORTAL_API_KEY or pass --csv")
        data = fetch_csv(key)
    classes = read_classes(data)
    table = build(classes)
    table["meta"] = {
        "built": time.strftime("%Y-%m-%d"), "source": "RadLex via BioPortal (CSV export)",
        "sha256": hashlib.sha256(data).hexdigest(), "classes": len(classes),
        "heads": len(table["heads"]), "names": len(table["names"]), "conflicts": len(table["conflicts"]),
        "kinds": dict(Counter(v[0] for v in table["heads"].values())), "attribution": ATTRIBUTION,
    }
    args.out.write_text(json.dumps(table, ensure_ascii=False, indent=0), "utf-8")
    print(json.dumps(table["meta"], indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
