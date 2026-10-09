#!/usr/bin/env python
"""Experiment F1: find the head of the phrase a mention stands in, and say whether it names another thing.

The linker accepts "liver" in "liver fatty acid binding protein expression" and "brain" in "gut-brain
axis" because it looks at the word next to the mention. A reader looks at the whole noun phrase,
finds its head, and asks what the head *is*: a molecule, a scale or a pathway (the mention is a piece of
a name) or a lesion, a tissue or a sample (the mention is a site). This script measures three ways of
finding the head and three ways of saying what it is, on the cases the external check judged:

* **phrase**: the words to the right of the mention up to a stop word or punctuation, head = last word
  (the rule in ``word_senses.noun_phrase_after``; no model, no dependency);
* **parser**: the head of the noun chunk in a dependency parse (spaCy; ``--spacy en_core_web_sm``);
* **attention**: the word that the mention's tokens attend to most in a biomedical encoder (middle layers),
  with ``--encoder`` (a Hugging Face model id). Attention is a proximity signal, not an explanation
  (Jain and Wallace 2019, "Attention is not Explanation"), so it is scored like any other feature, not
  trusted.

And three signals of what the head or the mention is:

* **list**: the head is in the non-site list of ``data/linking/word_senses.json``;
* **shift**: how much the encoder's reading of the mention changes when the sentence is added (cosine
  distance between the mention alone and the mention in its sentence): a piece of a compound changes more;
* **meaning**: the head word's embedding in its sentence against two prototypes, the last words of
  ontology names (UBERON: lobe, artery, wall, cortex) and the listed non-site heads (protein, score, axis),
  leaving the head itself out of its prototype.

Cases: every link the external check accepted and judged (agrees = 0, error = 1), from CRAFT and MedMentions,
plus the project's held-out sets as controls (links known to be right, English and Italian). Each signal
is reported as AUC for error against agrees, and as operating points: how many errors it catches for
how many right links it loses. It decides nothing; the report says which signal is worth keeping.

    python scripts/head_probe.py --craft ext/craft --medmentions ext/mm \
        --spacy en_core_web_sm --encoder microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import external_check as ec

from melampo.memory import anatomy_linker as al
from melampo.memory.word_senses import (
    SenseInventory,
    _fold,
    locate,
    noun_phrase_after,
    phrase_head,
)

DATA = ROOT / "data" / "linking"
WORD = re.compile(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9'’\-]*")


# -- cases ----------------------------------------------------------------------------------------


def collect_cases(args) -> list[dict]:
    obo = Path(args.uberon)
    lexicon, parts, graph, linker = ec.build_linker(obo)
    umls = ec.umls_of_nodes(obo)
    rows = []
    if args.craft:
        rows += ec.run_corpus(
            "craft",
            ec.craft_documents(Path(args.craft)),
            lexicon,
            parts,
            graph,
            linker,
            umls,
            args.limit,
            args.workers,
        )
    if args.medmentions:
        rows += ec.run_corpus(
            "medmentions",
            ec.medmentions_documents(Path(args.medmentions)),
            lexicon,
            parts,
            graph,
            linker,
            umls,
            args.limit,
            args.workers,
        )
    cases = [
        {
            "source": r["corpus"],
            "lang": "en",
            "mention": r["mention"],
            "sentence": r["sentence"],
            "at": r.get("at"),
            "label": 1 if r["by_project_rule"] == "error" else 0,
        }
        for r in rows
        if r["status"] == "accepted" and r.get("by_project_rule") in ("agrees", "error")
    ]
    for name, lang in (
        ("heldout_en", "en"),
        ("heldout2_en", "en"),
        ("heldout_it", "it"),
        ("heldout2_it", "it"),
    ):
        path = DATA / f"{name}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text("utf-8").splitlines():
            item = json.loads(line)
            result = linker.link(item["mention"], item["sentence"], at=item.get("start"))
            if result.status == al.ACCEPTED and result.cid == item["target"]:
                cases.append(
                    {
                        "source": f"control_{lang}",
                        "lang": lang,
                        "mention": item["mention"],
                        "sentence": item["sentence"],
                        "label": 0,
                    }
                )
    return cases


# -- signals --------------------------------------------------------------------------------------


def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Mann-Whitney AUC of a score for label 1 against label 0 (ties count half)."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    values = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(values))
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2 + 1
        i = j + 1
    inverse = np.empty(len(order), dtype=int)
    inverse[order] = np.arange(len(order))
    pos_rank = ranks[inverse[: len(pos)]].sum()
    return float((pos_rank - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def operating_points(scores: np.ndarray, labels: np.ndarray, lost=(0, 5, 20, 50)):
    """For each allowed number of right links lost: the threshold that catches most errors."""
    out = []
    right = scores[labels == 0]
    wrong = scores[labels == 1]
    for allowed in lost:
        best = None
        for t in sorted(set(scores.tolist()), reverse=True):
            lost_n = int((right >= t).sum())
            if lost_n > allowed:
                break
            caught = int((wrong >= t).sum())
            # most errors caught; among equal, the fewest right links lost
            if best is None or (caught, -lost_n) > (best[1], -best[2]):
                best = (t, caught, lost_n)
        out.append({"allowed_lost": allowed, "best": best})
    return out


class Syntax:
    def __init__(self, models: dict[str, str], heads: frozenset[str]):
        import spacy

        self.nlp = {lang: spacy.load(name, disable=["ner"]) for lang, name in models.items()}
        self.heads = heads

    def features(self, case: dict) -> dict:
        nlp = self.nlp.get(case["lang"])
        if nlp is None:
            return {}
        doc = nlp(case["sentence"])
        found = locate(case["mention"], case["sentence"], case.get("at"))
        if not found:
            return {}
        ids = {
            t.i
            for t in doc
            if t.idx >= found.start() and t.idx + len(t.text) <= found.end()
        }
        if not ids:
            return {}
        chunk = next(
            (c for c in doc.noun_chunks if min(ids) >= c.start and max(ids) < c.end), None
        )
        if chunk is None:
            return {"parser_head": "", "parser_is_head": 1.0, "parser_nonsite": 0.0}
        root = chunk.root
        words = [_fold(t.text) for t in chunk if t.i > max(ids)]
        head = phrase_head(words, self.heads)
        is_head = root.i in ids
        return {
            "parser_head": _fold(root.text) if not is_head else "",
            "parser_is_head": 1.0 if is_head else 0.0,
            "parser_nonsite": 1.0 if head else 0.0,
        }


class Encoder:
    """Hidden states and attentions of a Hugging Face encoder; the rest of this class is numpy."""

    def __init__(self, name: str, revision: str, layers: tuple[int, ...]):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModel.from_pretrained(
            name, revision=revision, attn_implementation="eager", output_attentions=True,
            output_hidden_states=True,
        ).eval()
        self.layers = layers
        try:
            from huggingface_hub import model_info

            self.sha = model_info(name, revision=revision).sha
        except Exception:  # noqa: BLE001
            self.sha = revision

    def encode(self, text: str):
        batch = self.tok(
            text, return_tensors="pt", return_offsets_mapping=True, truncation=True,
            max_length=256,
        )
        offsets = batch.pop("offset_mapping")[0].numpy()
        with self.torch.no_grad():
            out = self.model(**batch)
        hidden = np.stack([h[0].numpy() for h in out.hidden_states])  # L+1, T, D
        attn = np.stack([a[0].numpy() for a in out.attentions])  # L, H, T, T
        return offsets, hidden, attn


def token_ids(offsets: np.ndarray, start: int, end: int) -> list[int]:
    return [
        i
        for i, (a, b) in enumerate(offsets)
        if b > a >= start and b <= end
    ]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def pool(hidden: np.ndarray, ids: list[int], layers: tuple[int, ...]) -> np.ndarray:
    return hidden[list(layers)][:, ids].mean(axis=(0, 1))


def attention_head(offsets, attn, sentence, ids, layers):
    """The word the mention's tokens attend to most (outside the mention), and its share."""
    special = {i for i, (a, b) in enumerate(offsets) if b == a}
    mass = attn[list(layers)][:, :, ids, :].mean(axis=(0, 1, 2))  # T
    for i in list(ids) + sorted(special):
        mass[i] = 0.0
    words = [(m.start(), m.end()) for m in WORD.finditer(sentence)]
    per_word = defaultdict(float)
    for i, (a, b) in enumerate(offsets):
        if b <= a or mass[i] == 0.0:
            continue
        for k, (ws, we) in enumerate(words):
            if a >= ws and b <= we:
                per_word[k] += mass[i]
                break
    if not per_word:
        return "", 0.0, 0.0
    total = sum(per_word.values())
    k = max(per_word, key=per_word.get)
    ws, we = words[k]
    first_right = min((j for j, (a, _) in enumerate(words) if a >= offsets[ids[-1]][1]), default=None)
    right = sum(v for j, v in per_word.items() if first_right is not None and j >= first_right)
    return _fold(sentence[ws:we]), per_word[k] / total, right / total


def word_vector(enc: Encoder, text: str, word: str, layers):
    offsets, hidden, _ = enc.encode(text)
    found = re.search(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", text, re.IGNORECASE)
    if not found:
        return None
    ids = token_ids(offsets, found.start(), found.end())
    return pool(hidden, ids, layers) if ids else None


def prototypes(enc: Encoder, heads: list[str], obo: Path, layers, seed=7):
    """Mean vectors of the last words of ontology names (site) and of listed non-site heads."""
    names = []
    for line in obo.read_text("utf-8").splitlines():
        if line.startswith("name: "):
            words = line[6:].split()
            if 1 <= len(words) <= 3 and words[-1].isalpha():
                names.append(words[-1].lower())
    random.Random(seed).shuffle(names)
    site_words = list(dict.fromkeys(names))[:300]
    vectors = {}
    for word in dict.fromkeys(site_words + heads):
        vec = word_vector(enc, f"The {word}.", word, layers)
        if vec is not None:
            vectors[word] = vec
    site = {w: vectors[w] for w in site_words if w in vectors}
    non = {w: vectors[w] for w in heads if w in vectors}
    return site, non


def meaning(vec, word, site, non):
    s = np.mean([v for w, v in site.items() if w != word], axis=0)
    n = np.mean([v for w, v in non.items() if w != word], axis=0)
    return cosine(vec, n) - cosine(vec, s)


# -- run ------------------------------------------------------------------------------------------


def run(args) -> dict:
    inventory = SenseInventory.load()
    heads = inventory.non_site_heads
    cases = collect_cases(args)
    syntax = None
    if args.spacy:
        models = {"en": args.spacy}
        if args.spacy_it:
            models["it"] = args.spacy_it
        syntax = Syntax(models, heads)
    enc = None
    site = non = None
    layers_mid = tuple(int(x) for x in args.attention_layers.split(","))
    layers_last = (-4, -3, -2, -1)
    if args.encoder:
        enc = Encoder(args.encoder, args.revision, layers_mid)
        site, non = prototypes(enc, sorted(h for h in heads if h.isalpha()), Path(args.uberon), layers_last)
    for case in cases:
        sentence, mention = case["sentence"], case["mention"]
        found = locate(mention, sentence)
        phrase = noun_phrase_after(sentence, found.end()) if found else []
        head = phrase_head(phrase, heads)
        case["phrase_head"] = phrase[-1] if phrase else ""
        case["phrase_nonsite"] = 1.0 if head else 0.0
        case["phrase_has_words"] = 1.0 if phrase else 0.0
        if syntax:
            case.update(syntax.features(case))
        if enc and found and case["lang"] == "en":
            offsets, hidden, attn = enc.encode(sentence)
            ids = token_ids(offsets, found.start(), found.end())
            if ids:
                alone_off, alone_hidden, _ = enc.encode(mention)
                alone = [i for i, (a, b) in enumerate(alone_off) if b > a]
                for tag, layers in (("last", (-1,)), ("l4", layers_last), ("mid", (6, 7, 8))):
                    case[f"shift_{tag}"] = 1.0 - cosine(
                        pool(hidden, ids, layers), pool(alone_hidden, alone, layers)
                    )
                word, share, right = attention_head(offsets, attn, sentence, ids, layers_mid)
                case["attn_head"], case["attn_share"], case["attn_right"] = word, share, right
                for tag in ("phrase", "parser", "attn"):
                    w = case.get(f"{tag}_head")
                    if w and w.isalpha():
                        vec = word_vector(enc, sentence, w, layers_last)
                        if vec is not None:
                            case[f"meaning_{tag}"] = meaning(vec, w, site, non)
    return summarise(cases, heads, enc)


def summarise(cases: list[dict], heads, enc) -> dict:
    labels = np.array([c["label"] for c in cases])
    features = sorted(
        {k for c in cases for k, v in c.items() if isinstance(v, float)}
    )
    report = {
        "n": len(cases),
        "errors": int(labels.sum()),
        "by_source": {
            s: {"n": sum(c["source"] == s for c in cases), "errors": sum(c["label"] for c in cases if c["source"] == s)}
            for s in sorted({c["source"] for c in cases})
        },
        "encoder": getattr(enc, "sha", None),
        "features": {},
    }
    for name in features:
        have = [c for c in cases if name in c]
        scores = np.array([c[name] for c in have])
        y = np.array([c["label"] for c in have])
        report["features"][name] = {
            "n": len(have),
            "auc": auc(scores, y),
            "auc_craft": auc(
                np.array([c[name] for c in have if c["source"] == "craft"]),
                np.array([c["label"] for c in have if c["source"] == "craft"]),
            ),
            "auc_medmentions": auc(
                np.array([c[name] for c in have if c["source"] == "medmentions"]),
                np.array([c["label"] for c in have if c["source"] == "medmentions"]),
            ),
            "points": operating_points(scores, y),
        }
    # does the stop-list phrase find the parser's head?
    both = [c for c in cases if "parser_head" in c and c.get("phrase_head") is not None]
    if both:
        report["phrase_vs_parser"] = {
            "n": len(both),
            "same_head": sum(
                1 for c in both if c["parser_head"] and c["parser_head"] == c["phrase_head"]
            ),
            "both_say_nonsite": sum(
                1 for c in both if c["phrase_nonsite"] and c.get("parser_nonsite")
            ),
            "only_phrase": sum(
                1 for c in both if c["phrase_nonsite"] and not c.get("parser_nonsite")
            ),
            "only_parser": sum(
                1 for c in both if c.get("parser_nonsite") and not c["phrase_nonsite"]
            ),
        }
    report["examples"] = {
        "errors": [
            {k: c.get(k) for k in ("source", "mention", "phrase_head", "parser_head", "attn_head")}
            | {"sentence": c["sentence"][:160]}
            for c in cases
            if c["label"] == 1
        ][:80],
        "right_links_with_a_nonsite_head": [
            {k: c.get(k) for k in ("source", "mention", "phrase_head", "parser_head", "attn_head")}
            | {"sentence": c["sentence"][:160]}
            for c in cases
            if c["label"] == 0 and (c.get("phrase_nonsite") or c.get("parser_nonsite"))
        ][:80],
    }
    report["cases"] = cases
    return report


def markdown(report: dict) -> str:
    lines = [
        "# Experiment F1: head of the phrase",
        "",
        f"{report['n']} judged links ({report['errors']} errors). Sources: "
        + ", ".join(f"{k} {v['n']} ({v['errors']} errors)" for k, v in report["by_source"].items())
        + f". Encoder: {report['encoder']}.",
        "",
        "| signal | n | AUC | AUC CRAFT | AUC MedMentions | errors caught / right lost at 0, 5, 20, 50 lost |",
        "|---|---|---|---|---|---|",
    ]

    def f(x):
        return "-" if x is None else f"{x:.3f}"

    for name, v in sorted(report["features"].items(), key=lambda kv: -(kv[1]["auc"] or 0)):
        points = "; ".join(
            "-" if p["best"] is None else f"{p['best'][1]}/{p['best'][2]}" for p in v["points"]
        )
        lines.append(
            f"| {name} | {v['n']} | {f(v['auc'])} | {f(v['auc_craft'])} | {f(v['auc_medmentions'])} | {points} |"
        )
    if "phrase_vs_parser" in report:
        lines += ["", "Phrase scan against parser: " + json.dumps(report["phrase_vs_parser"]), ""]
    lines += ["", "## Right links a non-site head would stop", ""]
    for e in report["examples"]["right_links_with_a_nonsite_head"][:40]:
        lines.append(f"- {e['source']} `{e['mention']}` head `{e.get('phrase_head') or e.get('parser_head')}`: {e['sentence']}")
    lines += ["", "## Errors", ""]
    for e in report["examples"]["errors"][:80]:
        lines.append(
            f"- {e['source']} `{e['mention']}` phrase `{e.get('phrase_head')}` parser `{e.get('parser_head')}` attn `{e.get('attn_head')}`: {e['sentence']}"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--craft")
    ap.add_argument("--medmentions")
    ap.add_argument("--uberon", default=str(DATA / "uberon-basic.obo"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--spacy", help="English model, e.g. en_core_web_sm")
    ap.add_argument("--spacy-it", help="Italian model, e.g. it_core_news_sm")
    ap.add_argument("--encoder", help="Hugging Face model id")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--attention-layers", default="5,6,7,8")
    ap.add_argument("--out", default="head_probe.json")
    ap.add_argument("--markdown", default="head_probe.md")
    args = ap.parse_args(argv)
    report = run(args)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    Path(args.markdown).write_text(markdown(report), "utf-8")
    print(markdown(report)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
