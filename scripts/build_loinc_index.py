#!/usr/bin/env python
"""Build the LOINC index the laboratory mapper reads (``data/loinc/index.json``) from the LOINC release.

Download LOINC from https://loinc.org/downloads (free account; accept the LOINC licence). The release zip holds
``LoincTable/Loinc.csv`` and, under ``AccessoryFiles/LinguisticVariants``, the Italian variant
(``itIT*LinguisticVariant.csv``). Only active, point-in-time laboratory terms are kept (CLASSTYPE 1).

    python scripts/build_loinc_index.py --loinc Loinc_2.82.zip --out data/loinc/index.json

This material contains content from LOINC (http://loinc.org). LOINC is copyright Regenstrief Institute, Inc. and
the Logical Observation Identifiers Names and Codes (LOINC) Committee and is available at no cost under the
license at http://loinc.org/license.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from melampo.data.loinc import norm  # noqa: E402


def _rows(z: zipfile.ZipFile, suffix: str):
    name = next((n for n in z.namelist() if n.lower().endswith(suffix.lower())), None)
    if name is None:
        return []
    with z.open(name) as fh:
        return list(csv.DictReader(io.TextIOWrapper(fh, "utf-8-sig")))


def build(zip_path: Path) -> dict:
    z = zipfile.ZipFile(zip_path)
    table = _rows(z, "LoincTable/Loinc.csv")
    italian = {}
    for n in z.namelist():
        if "linguisticvariant" in n.lower() and "itit" in n.lower().replace("_", "").replace("-", ""):
            with z.open(n) as fh:
                for r in csv.DictReader(io.TextIOWrapper(fh, "utf-8-sig")):
                    italian[r.get("LOINC_NUM")] = r
    terms = []
    for r in table:
        if r.get("STATUS") != "ACTIVE" or r.get("CLASSTYPE") != "1":
            continue
        component = r["COMPONENT"].split("^")[0]
        names = {norm(component)}
        names |= {norm(s) for s in (r.get("RELATEDNAMES2") or "").split(";") if 2 <= len(s.strip()) <= 40}
        it = italian.get(r["LOINC_NUM"])
        if it and it.get("COMPONENT"):
            names.add(norm(it["COMPONENT"].split("^")[0]))
        terms.append({"code": r["LOINC_NUM"], "component": component, "property": r["PROPERTY"],
                      "time": r["TIME_ASPCT"], "system": r["SYSTEM"], "scale": r["SCALE_TYP"],
                      "method": r.get("METHOD_TYP", ""), "names": sorted(n for n in names if n),
                      "long_name": r.get("LONG_COMMON_NAME", "")})
    return {"source": zip_path.name, "terms": terms,
            "notice": "Contains content from LOINC (http://loinc.org), Regenstrief Institute, licence at loinc.org/license"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--loinc", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    index = build(args.loinc)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(index, ensure_ascii=False), "utf-8")
    print(f"{len(index['terms'])} laboratory terms indexed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
