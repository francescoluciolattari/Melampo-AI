#!/usr/bin/env python
"""Steps 2 and 3 (10 October 2026): the pair in its longer context, and all the cues of a reader together.

Same question as ``collocation_probe.py``: two adjacent words of a run, inside one MedMentions span or
across a span edge? Each pair gets the cues a reader who knows the domain has:

- **sequence statistics** (step 2): npmi of the pair; P(y | w x) and P(x | y z) from ordered triples;
- **memory of names** (point 1): the pair lies inside a remembered multi-word name (structures: UBERON and
  the lexicon; other things: NCIt names in the block memory and ``longer_names``), or a name ends or starts
  between the two words;
- **the head and its kind**: the NCIt kind of each word (the block memory's heads, at the memory's bar);
- **valency** (point 2): the second word is a procedure head whose frame has a site or device slot and the
  first word is a structure or a device;
- **grammar**: function words, numbers, verb forms (``Grammar.for_language("en")``).

Step 3 combines them with a logistic regression (scikit-learn defaults, standardised inputs, nothing
tuned). To keep the evaluation honest, MedMentions training documents are split in two halves: the
counts are made from half A (+ CRAFT + NCIt definitions), the regression is fitted on the spans of half B,
and the numbers are read on the *dev* documents (the frozen test documents only with ``--final``), with
the counts of half A. Every cue alone and the combination without the sequence statistics are reported
too, so that what each part adds is visible.

    python scripts/chunk_probe.py --medmentions ext/mm --craft ext/craft --ncit ncit.obo \
        --out chunk_probe.json --markdown chunk_probe.md
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import external_check as ec  # noqa: E402
import phrase_knowledge as pk  # noqa: E402
from head_probe import auc  # noqa: E402

from melampo.evaluation.frozen_split import DEFAULT_PATH as FROZEN_PATH  # noqa: E402
from melampo.evaluation.frozen_split import FrozenSplit  # noqa: E402
from melampo.memory import chunk_lattice as cl  # noqa: E402
from melampo.memory.chunk_lattice import _BREAK, _TOKEN  # noqa: E402
from melampo.memory.collocations import Collocations  # noqa: E402
from melampo.memory.frames import DEFAULT_PATH as FRAMES_PATH  # noqa: E402
from melampo.memory.frames import ProcedureFrames  # noqa: E402
from melampo.memory.grammar import Grammar  # noqa: E402
from melampo.memory.longer_names import LongerNames  # noqa: E402
from melampo.memory.word_senses import _PHRASE_STOP, _fold  # noqa: E402

KINDS = ("anatomy", "procedure", "device", "disease", "chemical", "protein", "gene", "property", "process",
         "activity", "organism", "food", "conceptual")
FAMILIES = {
    "sequence statistics": ("npmi", "has_npmi", "forward3", "has_forward3", "backward3", "has_backward3"),
    "memory of names": ("inside_name", "name_edge"),
    "kinds of the words": tuple(f"x_{k}" for k in KINDS) + tuple(f"y_{k}" for k in KINDS),
    "valency (frames)": ("y_site_frame", "y_device_frame"),
    "grammar": ("x_function", "y_function", "x_number", "y_number", "x_verbal", "y_verbal"),
}
FEATURES = tuple(f for fam in FAMILIES.values() for f in fam)
MAX_NAME = 7


def say(start: float, text: str) -> None:
    print(f"[{time.monotonic() - start:7.1f}s] {text}", file=sys.stderr, flush=True)


def run_tokens(text: str):
    """Runs of (folded word, start, end) as written (no punctuation between the words of a run)."""
    out, current, prev = [], [], None
    for m in _TOKEN.finditer(text):
        if prev is not None and _BREAK.search(text[prev : m.start()]):
            if current:
                out.append(current)
            current = []
        current.append((_fold(m.group(0)), m.start(), m.end()))
        prev = m.end()
    if current:
        out.append(current)
    return out


class Cues:
    def __init__(self, colloc: Collocations, names: set[str], memory, frames: ProcedureFrames, grammar: Grammar):
        self.colloc, self.names, self.memory, self.frames, self.grammar = colloc, names, memory, frames, grammar

    def kind(self, word: str) -> str:
        got = self.memory.head(word)
        return got[0] if got and got[1] >= cl.MIN_SHARE else ""

    def name_spans(self, words: list[str]) -> list[tuple[int, int]]:
        out = []
        for i in range(len(words)):
            for j in range(i + 2, min(len(words), i + MAX_NAME) + 1):
                if " ".join(words[i:j]) in self.names:
                    out.append((i, j))
        return out

    def pairs(self, run):
        words = [w for w, _, _ in run]
        spans = self.name_spans(words)
        for i in range(len(words) - 1):
            x, y = words[i], words[i + 1]
            v = self.colloc.npmi(x, y)
            f3 = self.colloc.forward3(words[i - 1], x, y) if i > 0 else None
            b3 = self.colloc.backward3(x, y, words[i + 2]) if i + 2 < len(words) else None
            inside = any(s <= i and i + 1 < e for s, e in spans)
            edge = any(e == i + 1 for s, e in spans) or any(s == i + 1 for s, e in spans)
            kx, ky = self.kind(x), self.kind(y)
            slots = self.frames.slots(cl.singular(y)) if ky == "procedure" else set()
            feats = {
                "npmi": v or 0.0, "has_npmi": float(v is not None),
                "forward3": f3 or 0.0, "has_forward3": float(f3 is not None),
                "backward3": b3 or 0.0, "has_backward3": float(b3 is not None),
                "inside_name": float(inside), "name_edge": float(edge and not inside),
                "y_site_frame": float("site" in slots and kx in ("anatomy", "device")),
                "y_device_frame": float("device" in slots and kx == "device"),
                "x_function": float(x in _PHRASE_STOP), "y_function": float(y in _PHRASE_STOP),
                "x_number": float(x.isdigit()), "y_number": float(y.isdigit()),
                "x_verbal": float(self.grammar.verbal(x)), "y_verbal": float(self.grammar.verbal(y)),
            }
            for k in KINDS:
                feats[f"x_{k}"] = float(kx == k)
                feats[f"y_{k}"] = float(ky == k)
            yield i, feats


def labelled(cues: Cues, docs):
    """(features, inside, doc, span of x, span of y) for every judged pair (at least one word in a span)."""
    X, y, where = [], [], []
    for doc, text, labs in docs:
        spans = sorted((s, e) for s, e, _ in labs)

        def span_of(a: int, b: int):
            for k, (s, e) in enumerate(spans):
                if s <= a and b <= e:
                    return k
                if s > b:
                    break
            return None

        for run in run_tokens(text):
            for i, feats in cues.pairs(run):
                sx, sy = span_of(run[i][1], run[i][2]), span_of(run[i + 1][1], run[i + 1][2])
                if sx is None and sy is None:
                    continue
                X.append([feats[f] for f in FEATURES])
                y.append(int(sx is not None and sx == sy))
                where.append((doc, sx, sy))
    return np.array(X, dtype=np.float64), np.array(y), where


def unit_share(pred: np.ndarray, labels: np.ndarray, where) -> tuple[int, float]:
    """Share of multi-word spans read as one unit (all inner pairs joined, all edge pairs cut)."""
    recs: dict = {}
    for p, lab, (doc, sx, sy) in zip(pred, labels, where, strict=True):
        for k in {sx, sy} - {None}:
            r = recs.setdefault((doc, k), {"inner": [], "edge": []})
            (r["inner"] if lab else r["edge"]).append(bool(p) if lab else not p)
    multi = [r for r in recs.values() if r["inner"]]
    return len(multi), (sum(all(r["inner"]) and all(r["edge"]) for r in multi) / len(multi) if multi else 0.0)


def fit_and_read(Xa, ya, Xb, yb, where_b, columns):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    idx = [FEATURES.index(c) for c in columns]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(Xa[:, idx], ya)
    p = model.predict_proba(Xb[:, idx])[:, 1]
    pred = p > 0.5
    tp = int((pred & (yb == 1)).sum())
    fp = int((pred & (yb == 0)).sum())
    fn = int((~pred & (yb == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    n_multi, share = unit_share(pred, yb, where_b)
    return {
        "auc": round(auc(p, yb), 4),
        "precision_inside": round(precision, 4),
        "recall_inside": round(recall, 4),
        "f1_inside": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "spans_read_as_one_unit": round(share, 4),
        "multiword_spans": n_multi,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--medmentions", required=True)
    ap.add_argument("--craft")
    ap.add_argument("--ncit")
    ap.add_argument("--extra", nargs="*", default=[], help="more counts (json.gz), e.g. PubMed or reports")
    ap.add_argument("--frames", default=str(FRAMES_PATH))
    ap.add_argument("--out", default="chunk_probe.json")
    ap.add_argument("--markdown", default="chunk_probe.md")
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args(argv)
    start = time.monotonic()
    mm = list(ec.medmentions_documents(Path(args.medmentions)))
    splits = pk.medmentions_splits(Path(args.medmentions))
    frozen = FrozenSplit.load(FROZEN_PATH) if FROZEN_PATH.exists() else None
    train = [x for x in mm if splits.get(x[0]) == "trng"]
    random.Random(11).shuffle(train)
    half_a, half_b = train[: len(train) // 2], train[len(train) // 2 :]
    target = "test" if args.final else "dev"
    evaluation = [x for x in mm if splits.get(x[0]) == target]
    if not args.final and frozen is not None:
        evaluation = [x for x in evaluation if not frozen.is_frozen(x[0], "medmentions")]
    colloc = Collocations().add_texts(t for _, t, _ in half_a)
    if args.craft:
        colloc.add_texts(t for _, t, _ in ec.craft_documents(Path(args.craft)))
    if args.ncit:
        colloc.add_texts(pk._ncit_definitions(Path(args.ncit)).values())
    for path in args.extra:
        extra = Collocations.load(Path(path))
        colloc.unigrams += extra.unigrams
        colloc.bigrams += extra.bigrams
        colloc.trigrams += extra.trigrams
        colloc.tokens += extra.tokens
        colloc.pairs += extra.pairs
    say(start, f"counts: {colloc.tokens:,} tokens; fit on {len(half_b)} training texts, read on {len(evaluation)} {target}")
    memory = cl.BlockMemory.load(cl.DEFAULT_PATH, LongerNames.load().names, cl.load_anatomy_names())
    names = {n for n in memory.names if " " in n}
    frames = ProcedureFrames.load(Path(args.frames)) if Path(args.frames).exists() else ProcedureFrames()
    cues = Cues(colloc, names, memory, frames, Grammar.for_language("en"))
    Xa, ya, _ = labelled(cues, half_b)
    Xb, yb, where_b = labelled(cues, evaluation)
    say(start, f"pairs: {len(ya)} to fit, {len(yb)} to read ({yb.mean():.3f} inside)")
    report: dict = {"target": target, "tokens": colloc.tokens, "pairs_fit": len(ya), "pairs_read": len(yb),
                    "single_cues": {}, "families": {}, "combined": {}}
    for f in FEATURES:
        col = Xb[:, FEATURES.index(f)]
        if col.min() != col.max():
            report["single_cues"][f] = round(auc(col, yb), 4)
    for fam, cols in FAMILIES.items():
        report["families"][fam] = fit_and_read(Xa, ya, Xb, yb, where_b, cols)
    without = [c for c in FEATURES if c not in FAMILIES["sequence statistics"]]
    report["combined"]["all cues"] = fit_and_read(Xa, ya, Xb, yb, where_b, FEATURES)
    report["combined"]["all cues but sequence statistics"] = fit_and_read(Xa, ya, Xb, yb, where_b, without)
    only_pairs = [c for c in FEATURES if c not in ("forward3", "has_forward3", "backward3", "has_backward3")]
    report["combined"]["all cues, pairs but no triples"] = fit_and_read(Xa, ya, Xb, yb, where_b, only_pairs)
    Path(args.out).write_text(json.dumps(report, indent=1), "utf-8")
    lines = [f"# Cues of a reader against MedMentions spans ({target} documents)", "",
             f"Counts: {colloc.tokens:,} tokens (half of the training abstracts, CRAFT, NCIt definitions"
             f"{', ' + ', '.join(Path(p).name for p in args.extra) if args.extra else ''}). "
             f"Fitted on {len(ya):,} pairs of the other half, read on {len(yb):,} pairs.", "",
             "| cues | AUC | precision | recall | F1 | spans read as one unit |", "|---|---|---|---|---|---|"]
    for group in ("families", "combined"):
        for name, r in report[group].items():
            lines.append(f"| {name} | {r['auc']} | {r['precision_inside']} | {r['recall_inside']} | {r['f1_inside']} "
                         f"| {r['spans_read_as_one_unit']} ({r['multiword_spans']}) |")
    lines += ["", "Single cues (AUC): " + ", ".join(f"{k} {v}" for k, v in sorted(
        report["single_cues"].items(), key=lambda kv: -abs(kv[1] - 0.5)))]
    Path(args.markdown).write_text("\n".join(lines) + "\n", "utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
