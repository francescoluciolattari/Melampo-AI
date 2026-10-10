#!/usr/bin/env python
"""Freeze the MedMentions test split, or check that the frozen list is intact.

  freeze_split.py create --medmentions <dir> [--rows external_check.rows.json] [--out data/linking/frozen_test_split.json]
  freeze_split.py verify [--manifest data/linking/frozen_test_split.json]

``create`` reads the corpus's own test list (``full/data/corpus_pubtator_pmids_test.txt``) and writes
the manifest with the SHA-256 of the sorted list and, if the external-check rows are given, how many
judged links of these documents were seen before the freeze. It refuses to overwrite a manifest.
"""

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melampo.evaluation.frozen_split import DEFAULT_PATH, FrozenSplit, FrozenSplitError  # noqa: E402


def create(args) -> int:
    out = Path(args.out)
    if out.exists():
        print(f"{out} exists: a frozen split is never rewritten", file=sys.stderr)
        return 2
    source = Path(args.medmentions) / "full" / "data" / "corpus_pubtator_pmids_test.txt"
    ids = source.read_text("utf-8").split()
    seen: dict = {}
    if args.rows:
        frozen_ids = set(ids)
        rows = json.loads(Path(args.rows).read_text("utf-8"))
        judged = [r for r in rows if r.get("corpus") == "medmentions" and str(r.get("doc")) in frozen_ids
                  and r.get("by_project_rule")]
        seen = {"external_check_judged_links": len(judged),
                "by_project_rule": dict(Counter(r["by_project_rule"] for r in judged)),
                "note": "these links were judged and their errors read before the freeze; "
                        "the split is frozen from now on, it is not unseen"}
    split = FrozenSplit.from_ids("medmentions", ids, args.date or dt.date.today().isoformat(),
                                 "corpus_pubtator_pmids_test.txt (MedMentions official split)", seen)
    split.save(out)
    print(f"frozen {len(split.ids)} documents, sha256 {split.sha256}")
    return 0


def verify(args) -> int:
    try:
        split = FrozenSplit.load(args.manifest)
    except FrozenSplitError as err:
        print(err, file=sys.stderr)
        return 1
    print(f"intact: {len(split.ids)} {split.corpus} documents, sha256 {split.sha256}, frozen {split.frozen_on}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--medmentions", required=True)
    c.add_argument("--rows")
    c.add_argument("--out", default=str(DEFAULT_PATH))
    c.add_argument("--date")
    c.set_defaults(fn=create)
    v = sub.add_parser("verify")
    v.add_argument("--manifest", default=str(DEFAULT_PATH))
    v.set_defaults(fn=verify)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
