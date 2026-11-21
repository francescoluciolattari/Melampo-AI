#!/usr/bin/env python
"""The index of the sources from the command line (8 November 2026): list it, add to it, let the project search,
check the rows of the reference library against their pages.

    python scripts/sources_index.py list [--status to_register]
    python scripts/sources_index.py add --id ... --kind journal --title ... --url ... --access open --status available
    python scripts/sources_index.py search [--limit 5]      # the weekly search: candidates from Europe PMC
    python scripts/sources_index.py check-quotes            # the words of each row of values.yml on its page
    python scripts/sources_index.py check                   # the index is well formed

The search and the check need the network (the workflow sources-index runs them every week on GitHub); the
mask (scripts/sources_mask.py) adds and edits by hand. Only public pages are fetched; nothing of a patient.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melampo.memory import sources_index as si
from melampo.memory.reference_values import VALUES, ReferenceValues

AGENT = "Melampo-sources-index/1.0 (research; https://github.com/francescoluciolattari/Melampo-AI)"


PMC_ARTICLE = re.compile(r"^https?://(?:pmc\.ncbi\.nlm\.nih\.gov|www\.ncbi\.nlm\.nih\.gov/pmc)/articles/(PMC\d+)", re.I)


def _get(url: str, timeout: int = 60) -> tuple[bytes, str]:
    # An article of PubMed Central is read from Europe PMC, the same open text (21 November 2026): PMC answers the
    # runners of GitHub with a reCAPTCHA page, and the row of the inferior vena cava could not be checked.
    pmc = PMC_ARTICLE.match(url)
    if pmc:
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc.group(1)}/fullTextXML"
    req = Request(url, headers={"User-Agent": AGENT, "Accept": "text/html,application/xml,application/pdf,application/json"})
    with urlopen(req, timeout=timeout) as r:
        raw, ctype = r.read(), r.headers.get("Content-Type", "")
    return raw, ("text/html" if pmc else ctype)          # the article's XML read by the HTML parser, word by word


def _json(url: str):
    raw, _ = _get(url)
    return json.loads(raw.decode("utf-8"))


def cmd_list(args: argparse.Namespace) -> int:
    index = si.SourcesIndex.load(args.index)
    for s in index.sources:
        if args.status and s.status != args.status:
            continue
        print(f"{s.id:38} {s.kind:15} {s.access:18} {s.status:12} {s.title}")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    index = si.SourcesIndex.load(args.index)
    source = si.Source(id=args.id, kind=args.kind, title=args.title, url=args.url, access=args.access,
                       status=args.status, publisher=args.publisher, licence=args.licence,
                       fields=[f for f in args.fields.split(",") if f], how_to_reach=args.how_to_reach,
                       note=args.note, of=args.of, identifier=args.identifier, added_by="manual")
    problems = index.put(source)
    for p in problems:
        print(p, file=sys.stderr)
    print("saved" if not problems else "not saved")
    return 1 if problems else 0


def cmd_search(args: argparse.Namespace) -> int:
    index = si.SourcesIndex.load(args.index)
    rows = ReferenceValues.load().rows
    added = 0
    for q in si.search_queries(rows):
        try:
            found = si.candidates_from_europe_pmc(q, _json, limit=args.limit)
        except Exception as e:  # noqa: BLE001 - one query that fails does not stop the others
            print(f"- {q}: {type(e).__name__}")
            continue
        n = si.add_candidates(index, found)
        added += n
        print(f"- {q}: {len(found)} found, {n} new")
        time.sleep(0.2)
    print(f"{added} candidate documents added")
    return 0


def cmd_check_quotes(args: argparse.Namespace) -> int:
    import yaml

    data = yaml.safe_load(Path(args.values).read_text("utf-8")) or {}
    pages: dict[str, tuple[str, str]] = {}
    results = {}
    for row in data.get("rows", []):
        url, quoted = (row.get("source") or {}).get("url", ""), row.get("quoted", "")
        if not url or not quoted:
            continue
        if url not in pages:
            try:
                raw, ctype = _get(url)
                pages[url] = (si.page_text(raw, ctype), "")
            except Exception as e:  # noqa: BLE001 - the row says the page could not be read
                pages[url] = ("", type(e).__name__)
            time.sleep(1.0)
        text, error = pages[url]
        res = si.check_quote(quoted, text) if text else {"found": False, "error": error or "no text"}
        results[row["id"]] = res
        if args.pages_out and not res.get("found"):
            # the words the check read, for a row it did not find (20 November 2026): what the check saw can be
            # compared with the page; never committed (the pages are the authors' work)
            out = Path(args.pages_out)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{row['id']}.txt").write_text(f"{url}\n\n{text}", "utf-8")
        print(f"- {row['id']}: {'found' if res.get('found') else 'not found'} "
              f"{res.get('error', '') or res.get('missing', '')}")
        for piece, near in (res.get("nearest") or {}).items():
            print(f"    the page says: {near}")
    changed = si.mark_rows(Path(args.values), results)
    print(f"{sum(r.get('found', False) for r in results.values())} of {len(results)} rows found on their pages; "
          f"{changed} rows written")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    problems = si.SourcesIndex.load(args.index).problems()
    for p in problems:
        print(p)
    print(f"{len(problems)} problem(s)")
    return 1 if problems else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--index", default=str(si.SOURCES))
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list")
    s.add_argument("--status", default="")
    a = sub.add_parser("add")
    for name in ("--id", "--kind", "--title", "--url", "--access", "--status"):
        a.add_argument(name, required=True)
    for name in ("--publisher", "--licence", "--fields", "--how-to-reach", "--note", "--of", "--identifier"):
        a.add_argument(name, default="")
    se = sub.add_parser("search")
    se.add_argument("--limit", type=int, default=5)
    q = sub.add_parser("check-quotes")
    q.add_argument("--values", default=str(VALUES))
    q.add_argument("--pages-out", default="", help="folder for the words read of the pages of rows not found")
    sub.add_parser("check")
    args = ap.parse_args(argv)
    return {"list": cmd_list, "add": cmd_add, "search": cmd_search, "check-quotes": cmd_check_quotes,
            "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
