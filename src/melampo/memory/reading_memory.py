"""The reading manual and the memory of verified examples (11 November 2026).

The models that read the reports are frozen; what they are given is not. Two things are given with every
question of the claims reading:

1. **The manual** (``data/reading/manual.md``): what the sources say about what a sentence states (hedges,
   speculation, factuality, negation, time, experiencer, the scales of certainty of radiology), with their own
   words and examples. It is versioned by its fingerprint, recorded with every answer.
2. **The examples**: the sentences most like this one that were already read and verified, with what they state.
   The memory grows with every verified sample; nothing is retrained to use it.

An example enters the memory only when two sources of different mechanism agree on it (the rule that makes a
finding certain): the label written before any reading and the reading of every model, on what the sentence
states and on its uncertainty; a label with a doubt never enters. What enters is listed with how it was verified,
and can be taken out. The same memory is the training set of the readers' adapters (every two or three months,
``scripts/reader_training.py``), which enter only if they win against the manual and the memory alone on a new
sample, by a rule fixed before.

The memory holds no text of a corpus that may not be copied (CT-RATE, PARROT): only the item, its sample, the
claims and the verification; the sentence is read from the sample's items at run time. A sample being measured
never gives examples to itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANUAL = ROOT / "data/reading/manual.md"
MEMORY = ROOT / "data/reading/memory.jsonl"

# what each claim of a label says, in the words the examples show the models
_SAYS = {
    "finding": "a finding of it", "involved": "a finding of another structure that touches it",
    "normal": "it is normal, or something is absent from it", "measure": "a measure of it",
    "history": "a fact of its past", "device": "a device in it", "not_assessable": "it cannot be assessed",
    "aside": "nothing of it: it only places something else", "indication": "why the exam was asked",
    "procedure": "a procedure on it", "nothing": "nothing of it",
    "uncertain": "the writer does not state it as certain (an uncertainty below high)",
    "uncertain_high": "stated with an uncertainty of level high (\"consistent with\")",
}


def manual_text(path: Path | None = None) -> str:
    p = path or MANUAL
    return p.read_text("utf-8") if p.exists() else ""


def manual_version(path: Path | None = None) -> str:
    text = manual_text(path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12] if text else ""


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@dataclass
class Example:
    item_id: str
    sample: str
    structure: str
    claims: list[str]
    note: str = ""
    verified_by: list[str] = field(default_factory=list)
    added: str = ""
    sentence: str = ""          # read from the sample's items at run time, never stored for a closed corpus
    mention: str = ""

    def line(self) -> str:
        says = "; ".join(_SAYS.get(c, c) for c in self.claims)
        return f"- Sentence: {self.sentence}\n  Marked words: \"{self.mention}\" ({self.structure.replace('_', ' ')})" \
               f"\n  It states: {says}" + (f" ({self.note})" if self.note else "")

    def as_dict(self) -> dict:
        return {"item_id": self.item_id, "sample": self.sample, "structure": self.structure, "claims": self.claims,
                "note": self.note, "verified_by": self.verified_by, "added": self.added}


class ReadingMemory:
    """The verified examples, and the most similar ones to a sentence (TF-IDF cosine over its words, the same
    structure first on a tie). Deterministic: the same memory and sentence give the same examples."""

    def __init__(self, examples: list[Example], exclude_samples: tuple[str, ...] = ()):
        self.examples = [e for e in examples if e.sentence and e.sample not in exclude_samples]
        self.exclude_samples = exclude_samples
        docs = [_tokens(e.sentence) for e in self.examples]
        df = Counter(t for d in docs for t in set(d))
        n = len(docs) or 1
        self._idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self._vectors = [self._vector(d) for d in docs]

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        v = {t: c * self._idf.get(t, math.log(len(self.examples) + 2) + 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def nearest(self, sentence: str, structure: str = "", k: int = 4) -> list[Example]:
        if not self.examples or k <= 0:
            return []
        q = self._vector(_tokens(sentence))
        scored = []
        for e, v in zip(self.examples, self._vectors):
            if e.sentence.strip() == sentence.strip():
                continue        # the sentence itself is never its own example
            score = sum(q[t] * v.get(t, 0.0) for t in q)
            scored.append((-score, e.structure != structure, e.item_id, e))
        scored.sort(key=lambda x: x[:3])
        return [e for *_, e in scored[:k]]

    @staticmethod
    def load(path: Path | None = None, texts: dict[str, dict] | None = None,
             exclude_samples: tuple[str, ...] = ()) -> "ReadingMemory":
        """``texts``: item_id -> the item (clause, mention) of the samples at hand; an example whose sentence is
        not at hand is not used."""
        p = path or MEMORY
        examples = []
        if p.exists():
            for line in p.read_text("utf-8").splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                item = (texts or {}).get(d["item_id"]) or {}
                examples.append(Example(item_id=d["item_id"], sample=d.get("sample", ""),
                                        structure=d.get("structure", ""), claims=list(d.get("claims", [])),
                                        note=d.get("note", ""), verified_by=list(d.get("verified_by", [])),
                                        added=d.get("added", ""), sentence=item.get("clause", ""),
                                        mention=item.get("mention", "")))
        return ReadingMemory(examples, exclude_samples)


# -- what a run gives the claims reading ----------------------------------------------------------------------
_ACTIVE: dict[str, object] = {}


def use(memory: ReadingMemory | None, manual: str | None = None) -> None:
    """Set the memory and the manual the claims reading is given (a probe sets them with the samples it measures
    excluded). ``manual=None`` reads ``data/reading/manual.md``; ``MELAMPO_READING_MANUAL=off`` gives none."""
    _ACTIVE["memory"] = memory
    _ACTIVE["manual"] = manual


def active_manual() -> str:
    if os.environ.get("MELAMPO_READING_MANUAL", "").strip().lower() == "off":
        return ""
    m = _ACTIVE.get("manual")
    return manual_text() if m is None else str(m)


def active_examples(sentence: str, structure: str = "") -> list[str]:
    memory = _ACTIVE.get("memory")
    if not isinstance(memory, ReadingMemory):
        return []
    k = int(os.environ.get("MELAMPO_READING_EXAMPLES", "4") or 0)
    return [e.line() for e in memory.nearest(sentence, structure, k)]


# -- the entry: two sources of different mechanism ------------------------------------------------------------
def model_uncertainty(answer: dict) -> str:
    """'' (none), 'high' (only of level high) or 'below' (some below high), from a model's parsed answer, of the
    questions before and after 10 November 2026."""
    about = answer.get("uncertain_about") or []
    words = list(answer.get("uncertainty") or []) + list(answer.get("certainty") or []) \
        + list(answer.get("alternatives") or [])
    if not words and not about:
        return ""
    levels = [u.get("level", "possible") for u in about] or ["possible"]
    return "high" if all(v == "high" for v in levels) else "below"


def label_uncertainty(claims: set[str]) -> str:
    return "below" if "uncertain" in claims else "high" if "uncertain_high" in claims else ""


def verified(row: dict, outcome_of, attached_kinds) -> list[str]:
    """How a row of a claims probe is verified, or [] if it is not: the label has no doubt; the reading attached
    what the label says (not a doubt, not to a person); and every model's own reading agrees with the label on
    what the sentence states and on its uncertainty. ``outcome_of`` and ``attached_kinds`` are the probe's."""
    got = row.get("claims") or {}
    gold = set(row.get("gold_claims") or [])
    if not gold or row.get("gold_doubt") or got.get("verdict") not in ("here", "elsewhere"):
        return []
    if outcome_of(attached_kinds(got)) != outcome_of(gold):
        return []
    models = got.get("models") or {}
    if not models or not all(a.get("valid", True) for a in models.values()):
        return []
    want = label_uncertainty(gold)
    if any(model_uncertainty(a) != want for a in models.values()):
        return []
    return ["label written before any reading", *(f"reading of {m}" for m in sorted(models))]


__all__ = ["Example", "MANUAL", "MEMORY", "ReadingMemory", "active_examples", "active_manual", "label_uncertainty",
           "manual_text", "manual_version", "model_uncertainty", "use", "verified"]
