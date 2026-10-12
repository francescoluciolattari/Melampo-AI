"""A memory of collocations: which words follow which, and how much more often than by chance.

Frank's third point (10 October 2026): a reader who has read medicine knows that "vena", "cava" and
"inferior" come together, in that order, and so does "filter placement"; the familiarity keeps the words
of a unit together. That is statistical learning of sequences: the probability that a word follows
another (transitional probability; Saffran, Aslin & Newport 1996) and the association of the pair beyond
chance (pointwise mutual information; Church & Hanks 1990). A dip in it is where people put a boundary
between chunks (Christiansen & Chater 2016, "chunk-and-pass").

What this module keeps, from plain text and no label:

- ``unigrams`` and ordered ``bigrams`` of folded words, counted inside *runs*: a run is a stretch of words
  joined only by spaces, hyphens, apostrophes and slashes (the same rule as the chunk lattice), so that a
  comma, a full stop or a bracket is never crossed. An abbreviation defined in brackets is seen through
  ("inferior vena cava (IVC) filter" counts "cava filter"), as the lattice reads it.
- ``npmi(a, b)``: normalised PMI, between -1 (never together) and 1 (only together); 0 is chance. Pairs
  seen fewer than ``MIN_COUNT`` times have no value (``None``): a count of one or two says nothing.
- ``forward(a, b)`` = P(b | a) and ``backward(a, b)`` = P(a | b).
- ``cohesion(words)``: the weakest link inside a block (a chunk holds only as well as its weakest pair).
- ``forward3(a, b, c)`` = P(c | a b) and ``backward3(a, b, c)`` = P(a | b c): the pair in its longer
  context (ordered triples, kept with the same lossy counting).

For a large corpus (PubMed) the counts are kept with lossy counting (Manku & Motwani 2002): a pair seen
less than once every ``1/epsilon`` pairs may be dropped; the error on any kept count is below ``epsilon *
N``. The file says which ``epsilon`` it was written with.

Limits: English and Italian alike (the tokeniser is the lattice's), but the counts are only as good as the
text: they carry the domain and the house style of whatever was read.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from .chunk_lattice import _BREAK, _TOKEN, see_through_abbreviations
from .word_senses import _fold

MIN_COUNT = 3  # a pair seen fewer times has no association (no value, not zero)
_SENTENCE = re.compile(r"(?<=[.!?;:])\s+")


def runs(text: str) -> list[list[str]]:
    """Folded words in runs that no punctuation breaks (an abbreviation in brackets is seen through)."""
    out: list[list[str]] = []
    for sentence in _SENTENCE.split(text):
        sentence, _ = see_through_abbreviations(sentence)
        current: list[str] = []
        prev_end = None
        for m in _TOKEN.finditer(sentence):
            if prev_end is not None and _BREAK.search(sentence[prev_end : m.start()]):
                if current:
                    out.append(current)
                current = []
            current.append(_fold(m.group(0)))
            prev_end = m.end()
        if current:
            out.append(current)
    return out


class Collocations:
    def __init__(self, epsilon: float = 0.0):
        self.unigrams: Counter[str] = Counter()
        self.bigrams: Counter[tuple[str, str]] = Counter()
        # ordered word triples (step 2, 10 October): the pair seen in its longer context
        self.trigrams: Counter[tuple[str, str, str]] = Counter()
        self.tokens = 0
        self.pairs = 0
        self.epsilon = epsilon
        self._delta: dict[tuple, int] = {}
        self._bucket = 1
        self._width = int(math.ceil(1 / epsilon)) if epsilon > 0 else 0

    # -- counting ------------------------------------------------------------------------------

    def add_text(self, text: str) -> None:
        for run in runs(text):
            self.add_run(run)

    def add_run(self, words: list[str]) -> None:
        self.unigrams.update(words)
        self.tokens += len(words)
        for triple in zip(words, words[1:], words[2:]):
            self.trigrams[triple] += 1
            if self._width and triple not in self._delta:
                self._delta[triple] = self._bucket - 1
        for pair in zip(words, words[1:]):
            self.bigrams[pair] += 1
            if self._width and pair not in self._delta:
                self._delta[pair] = self._bucket - 1
            self.pairs += 1
            if self._width and self.pairs % self._width == 0:
                self._prune()
                self._bucket += 1

    def _prune(self) -> None:
        """Lossy counting: drop the pairs whose count plus its possible undercount is within the bucket."""
        for table in (self.bigrams, self.trigrams):
            drop = [p for p, n in table.items() if n + self._delta.get(p, 0) <= self._bucket]
            for p in drop:
                del table[p]
                self._delta.pop(p, None)

    def add_texts(self, texts: Iterable[str]) -> Collocations:
        for t in texts:
            self.add_text(t)
        return self

    # -- association --------------------------------------------------------------------------

    def count(self, a: str, b: str) -> int:
        return self.bigrams.get((a, b), 0)

    def npmi(self, a: str, b: str, min_count: int = MIN_COUNT) -> float | None:
        n_ab = self.count(a, b)
        if n_ab < min_count or not self.pairs:
            return None
        # one denominator for the pair and the words, so that a pair always seen together scores 1
        p_ab = n_ab / self.tokens
        p_a = self.unigrams[a] / self.tokens
        p_b = self.unigrams[b] / self.tokens
        if p_ab >= 1.0 or p_a <= 0 or p_b <= 0:
            return None
        return max(-1.0, min(1.0, math.log(p_ab / (p_a * p_b)) / -math.log(p_ab)))

    def forward(self, a: str, b: str) -> float:
        return self.count(a, b) / self.unigrams[a] if self.unigrams.get(a) else 0.0

    def backward(self, a: str, b: str) -> float:
        return self.count(a, b) / self.unigrams[b] if self.unigrams.get(b) else 0.0

    def forward3(self, a: str, b: str, c: str, min_count: int = MIN_COUNT) -> float | None:
        """P(c | a b): how predictable the next word is after the two before it (``None`` when the pair
        a b is seen fewer than ``min_count`` times)."""
        n = self.count(a, b)
        return self.trigrams.get((a, b, c), 0) / n if n >= min_count else None

    def backward3(self, a: str, b: str, c: str, min_count: int = MIN_COUNT) -> float | None:
        """P(a | b c): how predictable the word before is from the two after it."""
        n = self.count(b, c)
        return self.trigrams.get((a, b, c), 0) / n if n >= min_count else None

    def cohesion(self, words: list[str]) -> float | None:
        """The weakest association inside a block of words (``None`` if a pair is not known)."""
        if len(words) < 2:
            return None
        values = [self.npmi(a, b) for a, b in zip(words, words[1:])]
        if any(v is None for v in values):
            return None
        return min(values)  # type: ignore[type-var]

    # -- files --------------------------------------------------------------------------------

    def prune(self, min_count: int = MIN_COUNT) -> Collocations:
        """Keep only pairs that can have a value (the file stays small)."""
        self.bigrams = Counter({p: n for p, n in self.bigrams.items() if n >= min_count})
        self.trigrams = Counter({p: n for p, n in self.trigrams.items() if n >= min_count})
        return self

    def to_json(self) -> dict:
        return {
            "tokens": self.tokens,
            "pairs": self.pairs,
            "epsilon": self.epsilon,
            "min_count": MIN_COUNT,
            "unigrams": dict(self.unigrams),
            "bigrams": [[a, b, n] for (a, b), n in self.bigrams.items()],
            "trigrams": [[a, b, c, n] for (a, b, c), n in self.trigrams.items()],
        }

    @classmethod
    def from_json(cls, data: dict) -> Collocations:
        c = cls(float(data.get("epsilon", 0.0)))
        c.tokens, c.pairs = int(data["tokens"]), int(data["pairs"])
        c.unigrams = Counter(data["unigrams"])
        c.bigrams = Counter({(a, b): int(n) for a, b, n in data["bigrams"]})
        c.trigrams = Counter({(a, b, x): int(n) for a, b, x, n in data.get("trigrams", [])})
        c._width = 0  # a loaded memory is read, not counted further
        return c

    def save(self, path: Path) -> None:
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(self.to_json(), fh)

    @classmethod
    def load(cls, path: Path) -> Collocations:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return cls.from_json(json.load(fh))


__all__ = ["MIN_COUNT", "Collocations", "runs"]
