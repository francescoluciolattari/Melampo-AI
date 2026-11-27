#!/usr/bin/env python
"""The archive of the reading confirmations (27 November 2026): from the cases reviewed in the review mask to the
rows of the plan of predetermined changes. Everything stays on this computer (``data/private``, never committed).

    python scripts/reading_confirmations.py collect [--cases data/private/cases] [--archive ARCHIVE] [--check]
    python scripts/reading_confirmations.py rows --part train|measure [--archive ARCHIVE] --out rows.jsonl

``collect`` adds to the archive every answer a second person confirmed (and a third adjudicated, when it is none
of the interpreters' readings), with its sentence de-identified (``data/deidentify.py``); with ``--check`` the two
local interpreters (``MELAMPO_LOCAL_CHAT_ENDPOINT``, ``MELAMPO_LOCAL_CHAT_MODELS``) read each de-identified sentence,
and only a sentence read to the confirmed card may train. ``rows`` writes the rows of one part: "train" (only what
passed the check) or "measure" (one case in five, never trained). It trains nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melampo.data.case_review import CASES, CaseFolder
from melampo.evaluation.review_feedback import ConfirmationStore, check_deidentified

ARCHIVE = ROOT / "data" / "private" / "reading_confirmations.jsonl"


def _key(r) -> str:
    return json.dumps([r.case, r.document, r.doubt, r.sentence, r.answer], ensure_ascii=False)


def collect(cases: Path, archive: Path, check: bool, read=None) -> dict[str, int]:
    """New confirmations of the cases into the archive (one already there is not added again)."""
    store = ConfirmationStore(archive)
    known = {_key(r) for r in store.records()}
    if check and read is None:
        from melampo.data.report_reading import one_procedure_interpreters
        from melampo.reading.reader import read_text

        chats, notes = one_procedure_interpreters()
        if not chats:
            raise SystemExit(f"--check needs the local interpreters: {', '.join(notes)}")

        def read(sentence: str) -> list[dict]:
            return read_text([("", sentence)], "case", chats)["cards"]

    counts = {"added": 0, "known": 0, "checked": 0, "passed": 0}
    for case in CaseFolder.all(cases):
        for record in case.confirmations():
            if _key(record) in known:
                counts["known"] += 1
                continue
            if read is not None:
                counts["checked"] += 1
                counts["passed"] += bool(check_deidentified(record, read))
            store.add(record)
            known.add(_key(record))
            counts["added"] += 1
    return counts


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--cases", type=Path, default=CASES)
    c.add_argument("--archive", type=Path, default=ARCHIVE)
    c.add_argument("--check", action="store_true")
    r = sub.add_parser("rows")
    r.add_argument("--archive", type=Path, default=ARCHIVE)
    r.add_argument("--part", choices=("train", "measure"), required=True)
    r.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.command == "collect":
        print(json.dumps(collect(args.cases, args.archive, args.check)))
        return 0
    rows = ConfirmationStore(args.archive).rows(args.part)
    args.out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    print(f"{len(rows)} rows ({args.part}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
