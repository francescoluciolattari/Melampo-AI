#!/usr/bin/env python
"""Assemble the development ("silver") gold set from the items and the labeller's files (12 October 2026).

Inputs: the folder written by ``build_silver_gold.py`` (``items.jsonl``, ``sentences.jsonl``,
``reports.jsonl``) and the label files in ``data/gold/silver_2026-10-12/labels``:

* span items (E3C): ``item_id|structure|relation|role|span_text|note``
* sentence items (PARROT, IU X-ray): ``report_id|mention|structure|relation|role|span_text|note``; the
  mention is located in the sentence the builder chose for that report (first occurrence).

Short codes: structure ``N`` = NONE_IN_CLASSES, ``X`` = NOT_ANATOMY, ``A`` = AMBIGUOUS; relation ``e`` equal,
``po`` part_of, ``co`` contour_of, ``ap`` approx; role ``s`` structure, ``p`` procedure_site, ``d``
device_site, ``o`` source_of, ``il`` inherent_location, ``in`` inside_a_name, ``nb`` not_a_body_site,
``ot`` other. Writes ``gold.jsonl`` for ``scripts/gold_set.py evaluate`` and fails on any label it cannot
place (a mention not in its sentence, an unknown class).

    python scripts/assemble_silver_gold.py --items silver/ --labels data/gold/silver_2026-10-12/labels \
        --out silver/gold.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRUCT = {"N": "NONE_IN_CLASSES", "X": "NOT_ANATOMY", "A": "AMBIGUOUS"}
REL = {"e": "equal", "po": "part_of", "co": "contour_of", "ap": "approx", "": ""}
ROLE = {"s": "structure", "p": "procedure_site", "d": "device_site", "o": "source_of", "il": "inherent_location",
        "in": "inside_a_name", "nb": "not_a_body_site", "ot": "other", "": ""}


def _jsonl(path: Path):
    return [json.loads(x) for x in path.read_text("utf-8").split("\n") if x.strip()]


def decode(structure: str, relation: str, role: str, classes: set[str]) -> tuple[str, str, str]:
    s = STRUCT.get(structure, structure)
    if s not in STRUCT.values() and s not in classes:
        raise ValueError(f"unknown class {structure!r}")
    rel = REL[relation]
    if s in ("NOT_ANATOMY", "AMBIGUOUS"):
        rel = ""
    elif not rel:
        rel = "equal"
    return s, rel, ROLE[role]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    classes = set(json.loads((ROOT / "data/linking/anatomy_lexicon.json").read_text("utf-8"))["classes"])
    items = {x["item_id"]: x for x in _jsonl(args.items / "items.jsonl")}
    sentences = {x["report_id"]: x for x in _jsonl(args.items / "sentences.jsonl")}
    gold, problems = [], []
    for path in sorted(args.labels.glob("*.txt")):
        for n, line in enumerate(path.read_text("utf-8").split("\n"), 1):
            if not line.strip():
                continue
            parts = line.split("|")
            try:
                if parts[0] in items:
                    item_id, structure, relation, role, span, note = (parts + [""] * 6)[:6]
                    it = items[item_id]
                    s, rel, rl = decode(structure, relation, role, classes)
                    row = {**it, "structure": s, "relation": rel, "role": rl, "span_text": span.strip(),
                           "note": note.strip(), "labelled_by": "Claude (single labeller)"}
                else:
                    rid, mention, structure, relation, role, span, note = (parts + [""] * 7)[:7]
                    sent = sentences[rid]
                    at = sent["sentence"].find(mention)
                    if at < 0:
                        raise ValueError(f"mention {mention!r} not in the sentence of {rid}")
                    s, rel, rl = decode(structure, relation, role, classes)
                    row = {"item_id": f"{rid}#{at}", "report_id": rid, "language": sent["language"],
                           "sentence": sent["sentence"], "mention": mention, "start": sent["offset"] + at,
                           "end": sent["offset"] + at + len(mention), "stratum": sent["stratum"],
                           "structure": s, "relation": rel, "role": rl, "span_text": span.strip(),
                           "note": note.strip(), "span_by": "Claude", "labelled_by": "Claude (single labeller)"}
                gold.append(row)
            except (KeyError, ValueError) as exc:
                problems.append(f"{path.name}:{n}: {exc}")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    args.out.write_text("\n".join(json.dumps(g, ensure_ascii=False) for g in gold) + "\n", "utf-8")
    print(f"{len(gold)} gold rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
