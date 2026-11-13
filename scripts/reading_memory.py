#!/usr/bin/env python
"""The memory of verified examples (11 November 2026): add the rows of a claims probe that two sources of
different mechanism agree on, list it, take an example out.

    python scripts/reading_memory.py add --run owner_probe.json --set ctrate5    # labels of data/gold/ctrate5
    python scripts/reading_memory.py list
    python scripts/reading_memory.py remove --item ctrate-215-en-7 --reason "..."

An example enters only if: the label has no doubt; the reading attached what the label says (no doubt, nothing
to a person); every model's own reading agrees with the label on what the sentence states and on its
uncertainty (src/melampo/memory/reading_memory.py, ``verified``); in a sample labelled by two labellers, only
the items both labelled (``double_labelled.txt``). Only samples labelled with the convention in
force enter (``CONVENTION``): an example labelled with a superseded convention would teach it. The memory holds
no text of the corpus: the item, its sample, the claims, how it was verified.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from melampo.memory import reading_memory as rm  # noqa: E402

# the samples labelled with the convention in force (11 November 2026: the uncertainty with its level, as the
# literature on hedging tells it apart); earlier samples read "consistent with" as certain
CONVENTION = {"ctrate5": "2026-11-11", "ctrate6": "2026-11-11", "ctrate7": "2026-11-13", "ctrate8": "2026-11-13", "ctrate9": "2026-11-13"}
REMOVED = ROOT / "data/reading/removed.jsonl"


def _load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()] if path.exists() else []


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def cmd_add(args: argparse.Namespace) -> int:
    import owner_probe as op

    if args.set not in CONVENTION:
        raise SystemExit(f"{args.set} was not labelled with the convention in force: its examples do not enter")
    labels = op.claims_labels(args.set)
    run = json.loads(Path(args.run).read_text("utf-8"))
    memory = _load(Path(args.memory))
    removed = {r["item_id"] for r in _load(REMOVED)}
    # a sample labelled by two labellers gives only the items both labelled (13 November 2026): an item with one
    # label has no check against an error of labelling but the readers', and the readers are what is measured
    double = op.CLAIMS_FILES[args.set].parent / "double_labelled.txt"
    both = {x.strip() for x in double.read_text("utf-8").splitlines() if x.strip() and not x.startswith("#")} \
        if double.exists() else None
    have = {r["item_id"] for r in memory}
    added = skipped = 0
    for row in run.get("rows", []):
        if row.get("set") != args.set or row["item_id"] not in labels or (both is not None and row["item_id"] not in both):
            continue
        claims, doubtful = labels[row["item_id"]]
        row = {**row, "gold_claims": sorted(claims), "gold_doubt": doubtful}   # the labels as they are now
        how = rm.verified(row, op.outcome_of_claims, op.attached_kinds)
        if not how or row["item_id"] in have or row["item_id"] in removed:
            skipped += 1
            continue
        note = next((ln.split("|", 2)[2] for ln in op.CLAIMS_FILES[args.set].read_text("utf-8").splitlines()
                     if ln.startswith(row["item_id"] + "|")), "")
        memory.append({"item_id": row["item_id"], "sample": args.set, "structure": row.get("structure", ""),
                       "claims": sorted(claims), "note": note, "verified_by": how,
                       "convention": CONVENTION[args.set], "run": args.name or Path(args.run).name,
                       "added": dt.date.today().isoformat()})
        have.add(row["item_id"])
        added += 1
    _write(Path(args.memory), memory)
    print(f"{added} examples added, {skipped} not verified or already there; {len(memory)} in the memory")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    memory = _load(Path(args.memory))
    from collections import Counter

    print(f"{len(memory)} examples; by sample {dict(Counter(r['sample'] for r in memory))}; "
          f"by claims {dict(Counter('+'.join(r['claims']) for r in memory))}")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    memory = _load(Path(args.memory))
    kept = [r for r in memory if r["item_id"] != args.item]
    if len(kept) == len(memory):
        raise SystemExit(f"{args.item} is not in the memory")
    removed = _load(REMOVED) + [{"item_id": args.item, "reason": args.reason,
                                 "removed": dt.date.today().isoformat()}]
    _write(Path(args.memory), kept)
    _write(REMOVED, removed)
    print(f"{args.item} removed: {args.reason}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--memory", default=str(rm.MEMORY))
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--run", required=True)
    a.add_argument("--set", required=True)
    a.add_argument("--name", default="", help="the run's name (e.g. owner-probe_20)")
    sub.add_parser("list")
    r = sub.add_parser("remove")
    r.add_argument("--item", required=True)
    r.add_argument("--reason", required=True)
    args = ap.parse_args(argv)
    return {"add": cmd_add, "list": cmd_list, "remove": cmd_remove}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
