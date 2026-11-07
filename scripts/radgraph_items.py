#!/usr/bin/env python
"""RadGraph and RadGraph-XL annotations as owner-probe items and labels (6 November 2026), for a protected run.

    python scripts/radgraph_items.py --radgraph test.json --out-items items.jsonl --out-labels labels.txt \\
        [--labeler labeler_1]

RadGraph (Jain et al. 2021) and RadGraph-XL (Delbrouck et al. 2024) annotate radiology reports with entities,
``ANAT-DP`` (an anatomy) and ``OBS-DP`` / ``OBS-DA`` / ``OBS-U`` (an observation definitely present, definitely
absent, uncertain), and relations ``located_at`` (an observation at an anatomy), ``modify`` and
``suggestive_of``. The labels are the radiologists', not ours: this is the measure without the single labeller.

How an annotation becomes the label of an anatomy mention (fixed before any data is seen; to verify on the first
files, and to change only before a run):

* an anatomy that only modifies another anatomy ("right" of "right lung") is not an item: its head is;
* an observation present ``located_at`` the anatomy, or at an anatomy that modifies it: ``observed``;
  uncertain only: ``uncertain``; absent only: ``absent``; none: ``nothing``.

RadGraph says where an observation is and whether it is present, not whether it is abnormal: "the heart is
normal" is an observation present at the heart. So these labels are not our claims: they measure two things,
the site (what RadGraph locates at an anatomy must not be set aside by us, nor attached elsewhere) and the
assertion (present, absent, uncertain). RadGraph has no "only placed" relation either, so criterion 1 is not
measured on it. The table that compares our readings with these labels is fixed before the first run.

The items keep the report's text for the run's context, so they are PhysioNet data: they are written only to the
protected environment, never committed, and the probe is run there with models of our own (``check_endpoint``
refuses an external service without ``--public-data``, which is never given for these data).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _entities(record: dict, labeler: str) -> dict:
    if "entities" in record:
        return record["entities"]
    return (record.get(labeler) or {}).get("entities", {})


def items_of(report_id: str, record: dict, labeler: str = "labeler_1") -> list[tuple[dict, str]]:
    """The anatomy mentions of a report, each with its label."""
    text = record.get("text", "")
    tokens = text.split()
    ents = _entities(record, labeler)
    modifies: dict[str, str] = {}          # anatomy -> the anatomy it modifies
    at: dict[str, list[str]] = {}          # anatomy -> labels of the observations located at it
    for eid, e in ents.items():
        for rel, target in e.get("relations", []):
            target = str(target)
            if rel == "modify" and e.get("label", "").startswith("ANAT") and ents.get(target, {}).get("label", "").startswith("ANAT"):
                modifies[eid] = target
            if rel == "located_at" and e.get("label", "").startswith("OBS"):
                at.setdefault(target, []).append(e["label"])
    for child, head in list(modifies.items()):    # what is located at a modifier is located at its head
        at.setdefault(head, []).extend(at.get(child, []))
    out = []
    for eid, e in sorted(ents.items(), key=lambda kv: kv[1].get("start_ix", 0)):
        if not e.get("label", "").startswith("ANAT") or eid in modifies:
            continue
        found = at.get(eid, [])
        if "OBS-DP" in found:
            kind = "observed"
        elif "OBS-U" in found:
            kind = "uncertain"
        elif found:
            kind = "absent"
        else:
            kind = "nothing"
        s, t = int(e.get("start_ix", 0)), int(e.get("end_ix", 0))
        clause = _sentence(tokens, s)
        mention = " ".join(tokens[s:t + 1])
        out.append(({"item_id": f"radgraph-{report_id}-{eid}", "pair": f"radgraph-{report_id}",
                     "report": f"radgraph-{report_id}", "language": "en", "mention": mention,
                     "clause": clause, "radgraph_entity": eid}, kind))
    return out


def _sentence(tokens: list[str], at: int) -> str:
    start = at
    while start > 0 and not re.search(r"[.?!]$", tokens[start - 1]):
        start -= 1
    end = at
    while end < len(tokens) - 1 and not re.search(r"[.?!]$", tokens[end]):
        end += 1
    return " ".join(tokens[start:end + 1])


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--radgraph", required=True)
    ap.add_argument("--out-items", required=True)
    ap.add_argument("--out-labels", required=True)
    ap.add_argument("--labeler", default="labeler_1", help="for the test set, annotated by two radiologists")
    a = ap.parse_args(argv)
    data = json.loads(Path(a.radgraph).read_text("utf-8"))
    items, labels = [], []
    for rid, record in data.items():
        for item, kind in items_of(rid, record, a.labeler):
            items.append(item)
            labels.append(f"{item['item_id']}|{kind}|RadGraph {a.labeler if 'entities' not in record else ''}".rstrip())
    Path(a.out_items).write_text("\n".join(json.dumps(i) for i in items) + "\n", "utf-8")
    Path(a.out_labels).write_text("# RadGraph labels (radiologists'), PhysioNet data: never committed\n"
                                  + "\n".join(labels) + "\n", "utf-8")
    print(f"{len(items)} anatomy mentions from {len(data)} reports -> {a.out_items}, {a.out_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
