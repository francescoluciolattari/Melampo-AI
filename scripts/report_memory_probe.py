#!/usr/bin/env python
"""Learning from reports: how the reader's memory grows as it reads radiology reports (10 October 2026).

Kintsch: where knowledge is missing there is no long-term working memory. The parts of the reader's
memory that come from plain text, with no label, are the collocations (pairs and triples of words) and the
frames of procedure heads; they can be learned by reading reports, as a resident learns by reading them.
This script simulates that learning and measures it on reports the memory has not read:

- each corpus (``reports.jsonl``: one ``{"text": ...}`` per line, e.g. from ``fetch_public_reports.py``)
  is split 90/10 by report, with a fixed seed;
- the memory reads 0 %, 10 %, 25 %, 50 % and 100 % of the 90 % (on top of an optional base memory, e.g.
  the counts of abstracts and definitions);
- on the held-out 10 % it counts **coverage**: the share of adjacent word pairs, and of triples, that the
  memory knows (seen at least ``MIN_COUNT`` times), and the share of **familiar units**: runs of two to
  five words in which every adjacent pair is known and above the threshold of ``chunk_lattice``.

No label is involved, so this says how much of what a report says the memory has met before, not whether
the reader is right: that needs the gold set.

    python scripts/report_memory_probe.py --reports iuxray.jsonl ct_rate.jsonl \
        [--base colloc.json.gz] --out report_memory.json --markdown report_memory.md --save reports_colloc.json.gz
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melampo.memory.chunk_lattice import COLLOC_THRESHOLD  # noqa: E402
from melampo.memory.collocations import MIN_COUNT, Collocations, runs  # noqa: E402

FRACTIONS = (0.0, 0.1, 0.25, 0.5, 1.0)


def read_reports(path: Path) -> list[str]:
    out = []
    for line in path.read_text("utf-8").split("\n"):
        if line.strip():
            text = json.loads(line).get("text", "")
            if text:
                out.append(text)
    return out


def merged(base: Collocations | None, more: Collocations) -> Collocations:
    if base is None:
        return more
    m = Collocations()
    m.unigrams = base.unigrams + more.unigrams
    m.bigrams = base.bigrams + more.bigrams
    m.trigrams = base.trigrams + more.trigrams
    m.tokens, m.pairs = base.tokens + more.tokens, base.pairs + more.pairs
    return m


def coverage(memory: Collocations, texts: list[str]) -> dict:
    pairs = known_pairs = triples = known_triples = 0
    familiar_words = words_total = 0
    for text in texts:
        for run in runs(text):
            words_total += len(run)
            ok = []
            for a, b in zip(run, run[1:]):
                pairs += 1
                n = memory.count(a, b) >= MIN_COUNT
                known_pairs += n
                v = memory.npmi(a, b) if n else None
                ok.append(v is not None and v > COLLOC_THRESHOLD)
            for t in zip(run, run[1:], run[2:]):
                triples += 1
                known_triples += memory.trigrams.get(t, 0) >= MIN_COUNT
            # words inside a familiar unit (a stretch of 2-5 words whose pairs are all familiar)
            i = 0
            while i < len(ok):
                if ok[i]:
                    j = i
                    while j < len(ok) and ok[j] and j - i < 4:
                        j += 1
                    familiar_words += j - i + 1
                    i = j + 1
                else:
                    i += 1
    return {
        "pairs": pairs,
        "pairs_known": round(known_pairs / pairs, 4) if pairs else 0.0,
        "triples_known": round(known_triples / triples, 4) if triples else 0.0,
        "words_in_familiar_units": round(familiar_words / words_total, 4) if words_total else 0.0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reports", nargs="+", type=Path, required=True)
    ap.add_argument("--base", type=Path, help="counts read before the reports (json.gz)")
    ap.add_argument("--out", default="report_memory.json")
    ap.add_argument("--markdown", default="report_memory.md")
    ap.add_argument("--save", type=Path, help="write the counts of all the training reports here (json.gz)")
    args = ap.parse_args(argv)
    base = Collocations.load(args.base) if args.base else None
    report: dict = {"base_tokens": base.tokens if base else 0, "corpora": {}}
    every = Collocations()
    for path in args.reports:
        texts = read_reports(path)
        random.Random(5).shuffle(texts)
        cut = max(1, len(texts) // 10)
        held, train = texts[:cut], texts[cut:]
        curve = {}
        for fraction in FRACTIONS:
            read = Collocations().add_texts(train[: int(len(train) * fraction)])
            memory = merged(base, read)
            curve[f"{int(fraction * 100)}%"] = {"reports_read": int(len(train) * fraction),
                                                "tokens_read": read.tokens, **coverage(memory, held)}
            print(f"{path.name} {fraction:.0%}: {curve[f'{int(fraction * 100)}%']}", file=sys.stderr, flush=True)
        report["corpora"][path.name] = {"reports": len(texts), "held_out": len(held), "curve": curve}
        every.add_texts(train)
    if args.save:
        every.prune().save(args.save)
    Path(args.out).write_text(json.dumps(report, indent=1), "utf-8")
    lines = ["# Learning from reports: what the memory knows of reports it has not read", "",
             f"Base memory before the reports: {report['base_tokens']:,} tokens.", "",
             "| corpus | reports read | tokens read | pairs known | triples known | words in familiar units |",
             "|---|---|---|---|---|---|"]
    for name, c in report["corpora"].items():
        for _, r in c["curve"].items():
            lines.append(f"| {name} | {r['reports_read']:,} | {r['tokens_read']:,} | {r['pairs_known']} "
                         f"| {r['triples_known']} | {r['words_in_familiar_units']} |")
    Path(args.markdown).write_text("\n".join(lines) + "\n", "utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
