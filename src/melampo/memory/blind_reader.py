"""The blind reader (stage 7): a second reading of the mention by a different mechanism.

Two LLMs that agree are not two independent confirmations: when two models both err they give the
same wrong answer about 60% of the time (Kim et al., ICML 2025), and nine LLM judges of seven
families are worth about two independent votes (Kohli, arXiv 2605.29800). A second reader is worth
what it does *not* share with the first. This one shares as little as it can:

* **no LLM, no network, no randomness**: the same mention gives the same answer in any run, and the
  answer can be reproduced for an audit;
* **it never sees the first answer**: it reads only the mention and derives a concept from it, and
  the comparison with the linked class happens afterwards (blind re-derivation; Prinz 2020 found
  that meaning is checked best by rebuilding it, not by judging it);
* **a different matching principle**: the first reader recognises a *known name* by exact match
  after normalisation (lexicon, part table); this one scores *every* name and synonym of the
  ontology pool by character n-gram similarity (TF-IDF cosine, 3 to 5 characters inside words), so
  it finds what the exact reader cannot and disagrees where the exact reader is fooled by a
  look-alike word. It does not use the adjective map, the noise words or the abbreviation rules of
  the exact reader.

What it says. For a mention and the class the linker chose it returns one of:

* ``support``: its best concept is the same class (or the same structure on either side), or the
  same UBERON structure the class stands for;
* ``against``: it is confident (its best concept scores at least ``min_score``, and clearly above
  anything the linked structure scores) and that concept is a different structure that is not a
  parent or child of the linked one;
* ``silent``: it cannot tell (a short form, a code, a word it does not know, or a score too low).

It decides nothing alone. The linker records its reading, counts it as one more independent
mechanism when it supports, and as a conflict when it disagrees; ``blind_veto`` turns a
disagreement into an abstention, off by default until the gold set says what it costs.

Honest limits. It reads the mention only, so it cannot tell "liver" the organ from "liver" in
"liver function"; the context streams do that. Its names are the same UBERON and lexicon names as the
exact reader's, so a wrong name in both would pass; and it is weak on Italian (the ontology has
English names only). It is a character-level reader, not a model of meaning: where meaning is the
question, only a radiologist's label settles it.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

SUPPORT, AGAINST, SILENT = "support", "against", "silent"

_SIDES = frozenset(
    "right left bilateral destro destra destri destre sinistro sinistra sinistri sinistre "
    "bilaterale dx sx sn rt lt".split()
)
_STOP = frozenset(
    "the of and or in on at to a an with without di del dello della dei degli delle dell "
    "da e ed il lo la le i gli un una al alla alle allo nel nella nei nelle sul sulla".split()
)
_WORD = re.compile(r"[a-z0-9]+")
_NGRAMS = (3, 4, 5)
_SIDE_SUFFIX = re.compile(r"_(left|right)$")
_RIB = re.compile(r"^rib_(left|right)_(\d+)$")
_CODE = re.compile(r"[A-Za-z]\d{1,2}(?:[-/][A-Za-z]?\d{1,2})?")


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _words(text: str) -> list[str]:
    return [w for w in _WORD.findall(_fold(text)) if w not in _SIDES and w not in _STOP]


def _grams(words: Sequence[str]) -> list[str]:
    grams: list[str] = []
    for word in words:
        padded = f" {word} "
        for n in _NGRAMS:
            grams.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
    return grams


def structure_of(cid: str) -> str:
    """The class without its side: kidney_left -> kidney, rib_right_3 -> rib_3."""
    match = _RIB.match(cid)
    if match:
        return f"rib_{match.group(2)}"
    return _SIDE_SUFFIX.sub("", cid)


@dataclass(frozen=True)
class Reading:
    verdict: str
    reason: str
    best: str | None = (
        None  # the concept the blind reader derived (a class or an UBERON id)
    )
    label: str = ""
    score: float = 0.0
    margin: float = 0.0


@dataclass
class BlindReader:
    """Character n-gram TF-IDF over every name of the pool; see the module docstring."""

    min_score: float = 0.80
    min_margin: float = 0.08
    tie: float = 0.03
    depth: int = 6
    max_leaders: int = 2
    min_letters: int = 4
    graph: Any = None
    equivalent: Mapping[str, str] = field(default_factory=dict)
    _names: list[tuple[str, str]] = field(default_factory=list)  # (cid, name)
    _norm: list[float] = field(default_factory=list)
    _index: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    _idf: dict[str, float] = field(default_factory=dict)
    _labels: dict[str, str] = field(default_factory=dict)
    _words_of: dict[str, list[frozenset[str]]] = field(default_factory=dict)

    @classmethod
    def from_sources(
        cls,
        lexicon: Any,
        terms: Sequence[Mapping[str, Any]],
        equivalent: Mapping[str, str] | None = None,
        graph: Any = None,
        parts: Mapping[str, Any] | None = None,
        **settings: Any,
    ) -> "BlindReader":
        """Built from the written names themselves: the lexicon's names for each class and the
        name and synonyms of each ontology term (``anatomy_linker.load_obo_terms``).

        It does not use the linker's candidate names. Those have been through the exact reader's
        normalisation (adjectives mapped to nouns, noise words dropped: "pulmonary vein" becomes
        "lung"), and a reader that shares the normalisation shares its mistakes. Names the lexicon
        marks as context-dependent (``requires_context``) are left out: a reader that sees only the
        mention cannot use a name that is a name only in some sentences."""
        reader = cls(graph=graph, equivalent=dict(equivalent or {}), **settings)
        seen: set[tuple[str, str]] = set()

        def add(cid: str, label: str, name: str) -> None:
            text = " ".join(_words(name))
            if text and (cid, text) not in seen:
                seen.add((cid, text))
                reader._names.append((cid, text))
                reader._words_of.setdefault(cid, []).append(frozenset(text.split()))
                reader._labels.setdefault(cid, label)

        for cid, entry in lexicon.classes.items():
            skip = {" ".join(_words(n)) for n in entry.get("requires_context", {})}
            label = f"{entry['it'][0]} / {entry['en'][0]}"
            for name in [*entry["it"], *entry["en"]]:
                if " ".join(_words(name)) not in skip:
                    add(cid, label, name)
        for term in terms:
            for name in [term["name"], *term.get("synonyms", ())]:
                add(term["id"], term["name"], name)
        # The curated part table (``data/linking/anatomy_parts.json``, written from anatomy): "lingula"
        # is a part of the left upper lobe, whatever UBERON's cerebellar "lingula" says. Its raw names
        # are names of the whole's classes, so the reader knows what the curators know.
        for entry in (parts or {}).get("direct", ()):
            whole = entry["whole"]
            for cid in (whole, f"{whole}_left", f"{whole}_right"):
                if cid in lexicon.classes:
                    for name in entry["names"]:
                        add(cid, reader._labels.get(cid, cid), name)
        reader._build()
        return reader

    def _build(self) -> None:
        documents = [_grams(_words(text)) for _, text in self._names]
        frequency: dict[str, int] = defaultdict(int)
        for grams in documents:
            for gram in set(grams):
                frequency[gram] += 1
        total = len(documents)
        self._idf = {
            g: math.log((1 + total) / (1 + df)) + 1.0 for g, df in frequency.items()
        }
        index: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self._norm = []
        for i, grams in enumerate(documents):
            counts: dict[str, int] = defaultdict(int)
            for gram in grams:
                counts[gram] += 1
            weights = {g: (1 + math.log(c)) * self._idf[g] for g, c in counts.items()}
            norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
            self._norm.append(norm)
            for gram, weight in weights.items():
                index[gram].append((i, weight / norm))
        self._index = dict(index)

    # -- reading ---------------------------------------------------------------------------------

    def derive(self, mention: str, top: int = 5) -> list[tuple[str, float]]:
        """The best concepts for the mention alone, best first: (cid, cosine). One entry per
        concept (its best name)."""
        return sorted(self._scores(mention).items(), key=lambda kv: -kv[1])[:top]

    def _scores(self, mention: str) -> dict[str, float]:
        words = _words(mention)
        grams = _grams(words)
        if not grams:
            return {}
        counts: dict[str, int] = defaultdict(int)
        for gram in grams:
            counts[gram] += 1
        query = {
            g: (1 + math.log(c)) * self._idf[g]
            for g, c in counts.items()
            if g in self._idf
        }
        norm = math.sqrt(sum(w * w for w in query.values()))
        if not norm:
            return {}
        scores: dict[int, float] = defaultdict(float)
        for gram, weight in query.items():
            for i, doc_weight in self._index.get(gram, ()):
                scores[i] += (weight / norm) * doc_weight
        best: dict[str, float] = {}
        for i, score in scores.items():
            cid = self._names[i][0]
            if score > best.get(cid, 0.0):
                best[cid] = score
        return best

    def _structure(self, cid: str) -> str:
        """The class (without side) a concept stands for, or the concept itself."""
        cid = self.equivalent.get(cid, cid)
        return structure_of(cid)

    def _is_linked(
        self, cid: str, wanted: str, nodes: set[str], side: str | None
    ) -> bool:
        """The concept is the linked structure, an UBERON node it stands for, or a part of it: one of
        its UBERON ancestors (part-of and is-a, up to ``depth`` steps) is a node of the linked class."""
        if self._structure(cid) == wanted or cid in nodes:
            return True
        if self.graph is None or not cid.startswith("UBERON:"):
            return False
        frontier, seen = {cid}, {cid}
        for _ in range(self.depth):
            frontier = {
                parent
                for node in frontier
                for parent, _ in self.graph.parents.get(node, ())
                if parent not in seen
            }
            if frontier & nodes:
                return True
            seen |= frontier
            if not frontier:
                break
        return False

    def read(self, mention: str, linked: str) -> Reading:
        """The blind reading of ``mention`` against the class the linker chose.

        The reader takes every concept that scores within ``tie`` of its best one. If any of them is
        the linked structure (or an UBERON node of it, or a part of it that climbs to it) it cannot
        tell them apart and supports; it disagrees only when none is."""
        letters = sum(c.isalpha() for c in mention)
        if (
            letters < self.min_letters
            or _CODE.fullmatch(mention.strip())
            or any(c.isdigit() for c in mention)
        ):
            return Reading(SILENT, "too_short_or_a_code")
        scores = self._scores(mention)
        if not scores:
            return Reading(SILENT, "no_name_shares_a_character_sequence")
        wanted = structure_of(linked)
        nodes = self.graph.nodes_of(linked) if self.graph is not None else set()
        side = (
            "right"
            if linked.endswith("_right")
            else "left"
            if linked.endswith("_left")
            else None
        )
        best_cid, best_score = max(scores.items(), key=lambda kv: kv[1])
        label = self._labels.get(best_cid, best_cid)
        if best_score < self.min_score:
            return Reading(
                SILENT, "no_name_is_close_enough", best_cid, label, best_score
            )
        leaders = [c for c, v in scores.items() if v >= best_score - self.tie]
        if any(self._is_linked(c, wanted, nodes, side) for c in leaders):
            return Reading(
                SUPPORT, "same_structure_or_a_part", best_cid, label, best_score
            )
        linked_score = max(
            (v for c, v in scores.items() if self._is_linked(c, wanted, nodes, side)),
            default=0.0,
        )
        # A reading that leaves a word of the mention unexplained ("brain parenchyma" read as
        # "parenchyma") or that fits several structures equally ("lingula": lung, cerebellum,
        # mandible) is not a disagreement the reader can stand behind.
        mention_words = frozenset(_words(mention))
        if not any(mention_words <= w for w in self._words_of.get(best_cid, ())):
            return Reading(
                SILENT,
                "the_reading_leaves_a_word_unexplained",
                best_cid,
                label,
                best_score,
            )
        if len({self._structure(c) for c in leaders}) > self.max_leaders:
            return Reading(
                SILENT, "the_name_fits_several_structures", best_cid, label, best_score
            )
        gap = best_score - linked_score
        if (
            self.graph is not None
            and self.graph.related(best_cid, linked) == "parent_and_child"
        ):
            return Reading(
                SILENT,
                "parent_or_child_of_the_linked_class",
                best_cid,
                label,
                best_score,
                gap,
            )
        if gap < self.min_margin:
            return Reading(
                SILENT, "two_structures_read_alike", best_cid, label, best_score, gap
            )
        return Reading(AGAINST, f"reads_as:{label}", best_cid, label, best_score, gap)
