#!/usr/bin/env python
"""Do collocation counts find the units that annotators marked? (Frank's point 3, 10 October 2026)

A MedMentions mention span is a unit a person marked: its words go together. Two adjacent words of a
run (no punctuation between them) are either *inside* one span or *across* a span edge (one in a span and
the other out, or in two different spans); pairs with both words outside every span are not judged. The
question: does the association of the pair, learned from plain text with no label, tell the two apart?

Fixed before the run (no tuning on the evaluation documents):

- primary score: normalised PMI (``Collocations.npmi``), pairs seen fewer than ``MIN_COUNT`` times have
  no value. The decision rule "join when npmi > threshold": the first run used 0 (chance) and joined
  almost every pair (in running text adjacent words are nearly always above chance); the threshold is now
  fitted on training documents only (counts from one half, best F1 on the other) and then applied;
- the counts never contain the documents they are evaluated on: MedMentions *dev* documents are read
  with counts from training documents (+ CRAFT, + NCIt definitions); the frozen test documents only with
  ``--final``;
- reported for each corpus size, so that the curve says what more text would buy.

Numbers: AUC (inside vs across) on the pairs that have a value; coverage (share of judged pairs that
have a value); with the rule npmi > 0, precision and recall of "inside" and the share of multi-word spans
read as one unit (every inner pair joined and both edges cut, where an edge pair exists).

    python scripts/collocation_probe.py --medmentions ext/mm --craft ext/craft --ncit ncit.obo \
        --out collocation_probe.json --markdown collocation_probe.md [--save collocations.json.gz]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import external_check as ec  # noqa: E402
import phrase_knowledge as pk  # noqa: E402
from head_probe import auc  # noqa: E402

from melampo.evaluation.frozen_split import DEFAULT_PATH as FROZEN_PATH  # noqa: E402
from melampo.evaluation.frozen_split import FrozenSplit  # noqa: E402
from melampo.memory.chunk_lattice import _BREAK, _TOKEN  # noqa: E402
from melampo.memory.collocations import Collocations  # noqa: E402
from melampo.memory.word_senses import _PHRASE_STOP, _fold  # noqa: E402


def say(start: float, text: str) -> None:
    print(f"[{time.monotonic() - start:7.1f}s] {text}", file=sys.stderr, flush=True)


def judged_pairs(text: str, spans: list[tuple[int, int]]):
    """(a, b, inside, span index of a, span index of b) for adjacent words of a run, in the text as written."""
    spans = sorted(spans)

    def where(start: int, end: int):
        for k, (s, e) in enumerate(spans):
            if s <= start and end <= e:
                return k
            if s > end:
                break
        return None

    toks = list(_TOKEN.finditer(text))
    for x, y in zip(toks, toks[1:]):
        if _BREAK.search(text[x.end() : y.start()]):
            continue
        sx, sy = where(x.start(), x.end()), where(y.start(), y.end())
        if sx is None and sy is None:
            continue
        yield _fold(x.group(0)), _fold(y.group(0)), (sx is not None and sx == sy), sx, sy, x.start(), y.start()


def _content(word: str) -> bool:
    return word not in _PHRASE_STOP and not word.isdigit()


def evaluate(memory: Collocations, docs, threshold: float = 0.0) -> dict:
    """The numbers of the docstring; ``threshold`` is the npmi above which two words are joined."""
    scores, labels, content_scores, content_labels = [], [], [], []
    base_scores, base_labels = [], []
    judged = covered = 0
    tp = fp = fn = 0
    span_pairs: dict = {}
    for doc, text, labs in docs:
        spans = [(s, e) for s, e, _ in labs]
        for a, b, inside, sx, sy, _xs, _ys in judged_pairs(text, spans):
            judged += 1
            both = _content(a) and _content(b)
            base_scores.append(1.0 if both else 0.0)
            base_labels.append(1 if inside else 0)
            v = memory.npmi(a, b)
            join = v is not None and v > threshold
            if v is not None:
                covered += 1
                scores.append(v)
                labels.append(1 if inside else 0)
                if both:
                    content_scores.append(v)
                    content_labels.append(1 if inside else 0)
            tp += join and inside
            fp += join and not inside
            fn += (not join) and inside
            for k in {sx, sy} - {None}:
                rec = span_pairs.setdefault((doc, k), {"inner": [], "edge": []})
                (rec["inner"] if inside else rec["edge"]).append(join if inside else not join)
    multi = [r for r in span_pairs.values() if r["inner"]]
    whole = sum(all(r["inner"]) and all(r["edge"]) for r in multi)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None

    def area(x, y):
        return round(auc(np.array(x), np.array(y)), 4) if y and 0 < sum(y) < len(y) else None

    return {
        "judged_pairs": judged,
        "threshold": threshold,
        "coverage": round(covered / judged, 4) if judged else 0.0,
        "auc_npmi": area(scores, labels),
        "auc_npmi_content_pairs": area(content_scores, content_labels),
        "auc_baseline_both_content_words": area(base_scores, base_labels),
        "inside_share": round(sum(labels) / len(labels), 4) if labels else None,
        "join_precision": None if precision is None else round(precision, 4),
        "join_recall": None if recall is None else round(recall, 4),
        "join_f1": None if f1 is None else round(f1, 4),
        "multiword_spans": len(multi),
        "spans_read_as_one_unit": round(whole / len(multi), 4) if multi else None,
    }


def fit_threshold(texts_with_labels, rng) -> float:
    """The npmi threshold that best separates inside from across on training documents only: counts from
    one half, the threshold that maximises F1 of "inside" on the other half (a grid of 0.05)."""
    docs = texts_with_labels[:]
    rng.shuffle(docs)
    half = len(docs) // 2
    memory = Collocations().add_texts(t for _, t, _ in docs[:half])
    best, best_f1 = 0.0, -1.0
    for step in range(0, 19):
        t = round(step * 0.05, 2)
        f1 = evaluate(memory, docs[half:], t)["join_f1"] or 0.0
        if f1 > best_f1:
            best, best_f1 = t, f1
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--medmentions", required=True)
    ap.add_argument("--craft")
    ap.add_argument("--ncit")
    ap.add_argument("--extra", nargs="*", default=[], help="collocation files (json.gz) to add, e.g. PubMed counts")
    ap.add_argument("--out", default="collocation_probe.json")
    ap.add_argument("--markdown", default="collocation_probe.md")
    ap.add_argument("--save", help="write the largest memory (training + CRAFT + NCIt) here")
    ap.add_argument("--final", action="store_true", help="evaluate on the frozen test documents")
    args = ap.parse_args(argv)
    start = time.monotonic()
    mm = list(ec.medmentions_documents(Path(args.medmentions)))
    splits = pk.medmentions_splits(Path(args.medmentions))
    frozen = FrozenSplit.load(FROZEN_PATH) if FROZEN_PATH.exists() else None
    train_docs = [(d, t, labs) for d, t, labs in mm if splits.get(d) == "trng"]
    train = [t for _, t, _ in train_docs]
    target = "test" if args.final else "dev"
    evaluation = [(d, t, labs) for d, t, labs in mm if splits.get(d) == target]
    if not args.final and frozen is not None:
        evaluation = [x for x in evaluation if not frozen.is_frozen(x[0], "medmentions")]
    say(start, f"MedMentions: {len(train)} training texts, {len(evaluation)} {target} texts")
    craft = [t for _, t, _ in ec.craft_documents(Path(args.craft))] if args.craft else []
    ncit = list(pk._ncit_definitions(Path(args.ncit)).values()) if args.ncit else []
    say(start, f"CRAFT {len(craft)} articles, NCIt {len(ncit)} definitions")
    rng = random.Random(13)
    shuffled = train[:]
    rng.shuffle(shuffled)
    configs = [
        ("MedMentions training 25%", shuffled[: len(shuffled) // 4]),
        ("MedMentions training 50%", shuffled[: len(shuffled) // 2]),
        ("MedMentions training 100%", shuffled),
        ("+ CRAFT", shuffled + craft),
        ("+ CRAFT + NCIt definitions", shuffled + craft + ncit),
    ]
    threshold = fit_threshold(train_docs, random.Random(7))
    say(start, f"threshold fitted on training documents only: npmi > {threshold}")
    report = {"target": target, "final": args.final, "threshold": threshold, "configs": {}}
    memory = None
    for name, texts in configs:
        memory = Collocations().add_texts(texts)
        result = evaluate(memory, evaluation, threshold)
        result["tokens"] = memory.tokens
        report["configs"][name] = result
        say(start, f"{name}: {result}")
    for path in args.extra:
        extra = Collocations.load(Path(path))
        merged = Collocations()
        merged.unigrams = memory.unigrams + extra.unigrams
        merged.bigrams = memory.bigrams + extra.bigrams
        merged.tokens, merged.pairs = memory.tokens + extra.tokens, memory.pairs + extra.pairs
        result = evaluate(merged, evaluation, threshold)
        result["tokens"] = merged.tokens
        report["configs"][f"+ {Path(path).name}"] = result
        say(start, f"+ {path}: {result}")
        memory = merged
    if args.save and memory is not None:
        memory.prune().save(Path(args.save))
        say(start, f"saved {args.save}: {len(memory.bigrams)} pairs")
    Path(args.out).write_text(json.dumps(report, indent=1), "utf-8")
    lines = [f"# Collocations against MedMentions spans ({target} documents)", "",
             f"Threshold fitted on training documents only: npmi > {threshold}. Baseline AUC (both words are content "
             f"words, no counts): {next(iter(report['configs'].values()))['auc_baseline_both_content_words']}.", "",
             "| counts from | tokens | coverage | AUC npmi | AUC npmi, content pairs | join precision | join recall | F1 | spans read as one unit |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, r in report["configs"].items():
        lines.append(f"| {name} | {r['tokens']:,} | {r['coverage']} | {r['auc_npmi']} | {r['auc_npmi_content_pairs']} "
                     f"| {r['join_precision']} | {r['join_recall']} | {r['join_f1']} | {r['spans_read_as_one_unit']} ({r['multiword_spans']}) |")
    Path(args.markdown).write_text("\n".join(lines) + "\n", "utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
