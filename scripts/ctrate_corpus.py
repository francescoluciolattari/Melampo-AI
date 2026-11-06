#!/usr/bin/env python
"""CT-RATE's chest CT reports in PARROT's form (6 November 2026), for the owner probe's sampler and context.

    python scripts/ctrate_corpus.py --csv validation_reports.csv --out ctrate.jsonl

CT-RATE (Hamamci et al., CC BY-NC-SA 4.0, gated on Hugging Face: the terms are accepted with an account) gives
each report in parts: clinical information, technique, findings, impressions (English). They are joined with
their headings, so that the section reader (``memory/report_sections``) reads them as a radiologist does: the
clinical information is the indication, not a finding. Several reconstructions of one scan share a report: each
report is kept once. Like PARROT's, the texts are never committed: only the sample's labels are.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# the parts of a report and the headings they are given; a column is found by its name (English version first)
PARTS = (("clinical", "Clinical information"), ("technique", "Technique"), ("finding", "Findings"),
         ("impression", "Impressions"))


def columns(header: list[str]) -> dict[str, str]:
    found = {}
    for key, _ in PARTS:
        named = [h for h in header if key in h.lower()]
        english = [h for h in named if h.lower().endswith("_en")]
        if english or named:
            found[key] = (english or named)[0]
    return found


def convert(rows: list[dict], header: list[str]) -> list[dict]:
    cols = columns(header)
    if "finding" not in cols:
        raise SystemExit(f"no findings column among {header}")
    name = next((h for h in header if "volume" in h.lower()), header[0])
    seen: set[str] = set()
    out = []
    for r in rows:
        parts = []
        for key, heading in PARTS:
            text = " ".join(str(r.get(cols.get(key, ""), "") or "").split())
            if text and text.lower() not in ("nan", "none"):
                parts.append(f"{heading}: {text}" + ("" if text.endswith(".") else "."))
        report = "\n".join(parts)
        if not report or report in seen:
            continue                    # another reconstruction of the same scan
        seen.add(report)
        volume = str(r.get(name, ""))
        out.append({"no": f"ctrate-{len(out) + 1}", "language": "English", "modality": "CT", "area": "chest",
                    "report": report, "translation": report, "source": "CT-RATE", "volume": volume,
                    "patient": re.sub(r"_[a-z]_\d+\.nii(\.gz)?$", "", volume)})
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with open(a.csv, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        header = reader.fieldnames or []
    out = convert(rows, header)
    Path(a.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", "utf-8")
    print(f"{len(out)} reports from {len(rows)} rows ({', '.join(columns(header).values())}) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
