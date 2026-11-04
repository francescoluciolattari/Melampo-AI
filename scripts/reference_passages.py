#!/usr/bin/env python
"""The local copy of the paragraphs of the reference sources (4 November 2026). Not committed: the texts belong
to their authors (Radiopaedia: CC BY-NC-SA 3.0 modified; open-access articles: their licence), as the texts of
PARROT. The table (``data/reference/values.yml``) holds only the facts with their citation.

    python scripts/reference_passages.py add --url URL --section "gross anatomy" --title "Portal vein" \\
        --licence "CC BY-NC-SA 3.0 (modified)" --text-file paragraph.txt [--links id1,id2] --out passages.jsonl
    python scripts/reference_passages.py check --out passages.jsonl

A paragraph is added by a person who has read the page (copied from it, with its section), never a whole
chapter: ``add`` refuses a text of more than MAX_WORDS words. ``links`` names the paragraphs of other sections
the paragraph refers to ("see the 3-6-9 rule"). Automated downloading is not done here: the terms of each site
decide whether it is allowed, and they are checked first.

The runtime reads the copy from ``MELAMPO_REFERENCE_PASSAGES``; for patients' reports it is a local file, never
fetched during the reading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

MAX_WORDS = 250


def _id(url: str, section: str, text: str) -> str:
    return "p-" + hashlib.sha1(f"{url}|{section}|{text}".encode()).hexdigest()[:10]


def add(args: argparse.Namespace) -> int:
    text = " ".join(Path(args.text_file).read_text("utf-8").split())
    if len(text.split()) > MAX_WORDS:
        print(f"a paragraph, not a chapter: {len(text.split())} words, at most {MAX_WORDS}", file=sys.stderr)
        return 2
    out = Path(args.out)
    rows = [json.loads(x) for x in out.read_text("utf-8").splitlines() if x.strip()] if out.exists() else []
    row = {"id": _id(args.url, args.section, text), "url": args.url, "section": args.section, "title": args.title,
           "licence": args.licence, "text": text, "links": [x for x in (args.links or "").split(",") if x]}
    if any(r["id"] == row["id"] for r in rows):
        print(f"already there: {row['id']}")
        return 0
    rows.append(row)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", "utf-8")
    print(f"added {row['id']} ({len(text.split())} words)")
    return 0


def check(args: argparse.Namespace) -> int:
    rows = [json.loads(x) for x in Path(args.out).read_text("utf-8").splitlines() if x.strip()]
    ids = {r["id"] for r in rows}
    problems = [f"{r['id']}: {len(r['text'].split())} words" for r in rows if len(r["text"].split()) > MAX_WORDS]
    problems += [f"{r['id']}: links to {link}, not in the copy" for r in rows for link in r.get("links", [])
                 if link not in ids]
    problems += [f"{r['id']}: no licence" for r in rows if not r.get("licence")]
    for p in problems:
        print(p)
    print(f"{len(rows)} paragraphs, {len(problems)} problem(s)")
    return 1 if problems else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    for name in ("--url", "--section", "--title", "--licence", "--text-file", "--out"):
        a.add_argument(name, required=True)
    a.add_argument("--links", default="")
    c = sub.add_parser("check")
    c.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return add(args) if args.cmd == "add" else check(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
