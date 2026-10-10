#!/usr/bin/env python
"""Count collocations (ordered word pairs) over PubMed baseline files, for ``melampo.memory.collocations``.

Reads PubMed XML (``pubmedNNnXXXX.xml.gz``, local files or https URLs) title and abstract text, counts
unigrams and ordered bigrams inside runs (see ``collocations.runs``) with lossy counting, drops every
PMID listed in ``--exclude-pmids`` (the MedMentions documents: they are PubMed abstracts and some are
evaluation documents, so their text must never be in the counts), and writes a pruned ``.json.gz``.

    python scripts/build_collocations.py --files pubmed26n0001.xml.gz ... \
        --exclude-pmids ext/mm/full/data/corpus_pubtator_pmids_*.txt --out pubmed_collocations.json.gz

Licence: the PubMed baseline is distributed by the U.S. National Library of Medicine under its terms
and conditions; abstracts may be under the copyright of their publishers. This writes *counts* of word
pairs, not text; whether the counts may ship inside the device is a question for whoever signs the
technical file, and is written down in the documentation.
"""

from __future__ import annotations

import argparse
import gzip
import io
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melampo.memory.collocations import Collocations  # noqa: E402


def abstracts(stream):
    """(pmid, title + abstract) of every article in a PubMed XML stream."""
    for _, elem in ET.iterparse(stream, events=("end",)):
        if elem.tag != "PubmedArticle":
            continue
        pmid = elem.findtext("MedlineCitation/PMID") or ""
        article = elem.find("MedlineCitation/Article")
        parts = []
        if article is not None:
            parts.append("".join(article.find("ArticleTitle").itertext()) if article.find("ArticleTitle") is not None else "")
            for node in article.findall("Abstract/AbstractText"):
                parts.append("".join(node.itertext()))
        elem.clear()
        text = " ".join(p for p in parts if p)
        if text:
            yield pmid, text


def open_source(source: str):
    if source.startswith("https://"):
        data = urllib.request.urlopen(source, timeout=300).read()
        return gzip.open(io.BytesIO(data))
    return gzip.open(source)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--files", nargs="+", required=True, help="PubMed XML .gz files or https URLs")
    ap.add_argument("--exclude-pmids", nargs="*", default=[], help="files with PMIDs (whitespace separated) to skip")
    ap.add_argument("--epsilon", type=float, default=2e-7, help="lossy counting error bound (share of all pairs)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    start = time.monotonic()
    excluded: set[str] = set()
    for path in args.exclude_pmids:
        excluded.update(Path(path).read_text("utf-8").split())
    memory = Collocations(epsilon=args.epsilon)
    n = skipped = 0
    for source in args.files:
        with open_source(source) as stream:
            for pmid, text in abstracts(stream):
                if pmid in excluded:
                    skipped += 1
                    continue
                memory.add_text(text)
                n += 1
        print(f"[{time.monotonic() - start:7.1f}s] {source}: {n} abstracts, {memory.tokens:,} tokens, "
              f"{len(memory.bigrams):,} pairs kept, {skipped} excluded", file=sys.stderr, flush=True)
    memory.prune().save(Path(args.out))
    print(f"wrote {args.out}: {n} abstracts, {memory.tokens:,} tokens, {len(memory.bigrams):,} pairs "
          f"(seen at least 3 times), {skipped} excluded PMIDs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
