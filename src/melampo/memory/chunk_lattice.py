"""Read the phrase around a mention as blocks, then decide what the mention is (9 October 2026).

The streams of the linker read a word and vote on a link. A person reads the other way round: the
phrase is built first ("inferior vena cava filter placement" is one thing, a procedure on a device),
then the word is placed in it. This module does that, deterministically and without a model:

* **Nodes.** Every run of words that could be one block: the mention itself (a structure), a
  *remembered* name with its kind (``block_memory.json``, ``longer_names.json``), a *composed* block
  (modifiers and a head whose kind is known: English compounds are right-headed), or one word.
* **Segmentation.** The window is covered, without overlap, by the blocks of minimum total cost
  (dynamic programming). Remembered names are cheaper than composed blocks, which are cheaper than
  words; a head whose kind is shared among several kinds costs more.
* **Least commitment.** The cost of the best reading that puts the mention in a block of a
  *different decision class* is kept. If it is within ``MARGIN`` of the best, the phrase is
  *underspecified* and nothing is decided on it.
* **Then the decision**, only after the block is fixed: the block that holds the mention is a
  structure (link), a procedure / device / process / measurement of it (the structure is kept with a
  role), the name of something that is not a body site (protein, food...), or not read.
* A second, higher level joins blocks with "of": "heart of Maroilles cheese" (a part of an object),
  "removal of the gallbladder" (a procedure on it).

Limits, stated where they bind. English only (the heads come from NCIt): in Italian the head comes
first and the type names are Italian, so this is not yet usable on Italian reports. The window is the
noun phrase (stop words and punctuation end it), at most ``LEFT`` + ``RIGHT`` words (5-7 words is the
span in which people combine words). One sense per discourse is not here (it needs the document).
The costs are fixed by principle (memory < composition < word), not fitted to any corpus.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .word_senses import _PHRASE_STOP, _fold, locate

DEFAULT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "linking" / "block_memory.json"
)

LEFT, RIGHT = 5, 7
MARGIN = 0.15
# The cost of a block: remembered < composed < a word on its own.
C_MEMORY, C_COMPOSED, C_PER_WORD, C_SHARE = 0.6, 1.0, 0.1, 1.0
C_WORD_KNOWN, C_WORD_UNKNOWN = 1.2, 1.5
# A head inside a block that could have closed it (Nelson et al. 2017: a node stays open until words
# can be fused): "brain weight strains" is "[brain weight] strains", not one block headed by "strains".
C_OPEN = 2.0
# The file keeps a head only if it is at least this sure of its kind (the builder's threshold); the
# "of" constructions ask the same.
MIN_SHARE = 0.75

# To the right of the mention a verb form starts a new clause ("lobe measuring 7 mm", "silhouette
# appears normal") and a number starts a measurement: the noun phrase ends there. The verbs are a
# closed list of the ones reports use; a participle is not recognised by its ending ("binding").
_REPORT_VERBS = frozenset(
    """appear appears appeared show shows showed demonstrate demonstrates demonstrated suggest
    suggests suggested measure measures measured represent represents remain remains remained
    contain contains contained reveal reveals revealed represent consistent seen noted identified
    present persists persist measuring showing demonstrating appearing representing containing""".split()
)
_TOKEN = re.compile(r"[A-Za-zÀ-ÿ0-9]+")
# Between two words of one block: spaces, hyphens, apostrophes, slashes; anything else ends the run.
_BREAK = re.compile(r"[^\s\-'’/]")
_OF = re.compile(
    r"^\s+of\s+(?:the\s+|a\s+|an\s+|this\s+|these\s+|that\s+)?", re.IGNORECASE
)
_OF_BEFORE = re.compile(
    r"\bof\s+(?:the\s+|a\s+|an\s+|this\s+|these\s+|that\s+)?$", re.IGNORECASE
)

ROLE_OF_KIND = {
    "procedure": "procedure_site",
    "device": "device_site",
    "process": "inherent_location",
    "property": "inherent_location",
    "activity": "inherent_location",
    "protein": "inside_a_name",
    "gene": "inside_a_name",
    "chemical": "inside_a_name",
    "assessment_tool": "inside_a_name",
    "food": "not_a_body_site",
    "organism": "not_a_body_site",
}
# What the linker does with a block of this kind: keep the link; keep the structure with a role; say
# the word is not a structure here. Kinds not listed are not acted on ("conceptual" is too wide).
CLASS_OF_KIND = {
    "anatomy": "link",
    "disease": "link",
    "procedure": "role",
    "device": "role",
    "process": "role",
    "property": "role",
    "activity": "role",
    "protein": "not_a_site",
    "gene": "not_a_site",
    "chemical": "not_a_site",
    "assessment_tool": "not_a_site",
    "food": "not_a_site",
    "organism": "not_a_site",
}
SOURCE_KINDS = frozenset(("protein", "gene", "chemical"))
# NCIt puts nutrients and biochemicals under "food" ("sterol", "glutamate") and every "Whole ..." name
# under "organism": the kind of a *head word* is too coarse to say "this is not a body site". These
# kinds act only through a remembered name or the "of" construction, never through composition.
UNTRUSTED_COMPOSED = frozenset(("food", "organism", "conceptual"))
# For "<mention> of <phrase>": a phrase of this kind has the mention as a part of an object.
OBJECT_KINDS = frozenset(("food", "device"))
OF_ROLE_KINDS = frozenset(("procedure", "process", "property", "activity"))


def singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


@dataclass(frozen=True)
class BlockMemory:
    heads: dict[str, tuple[str, float]] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(
        cls, path: Path = DEFAULT_PATH, longer_names: dict[str, str] | None = None
    ) -> BlockMemory:
        data: dict[str, Any] = {}
        if path.exists():
            data = json.loads(path.read_text("utf-8"))
        return cls.from_json(data, longer_names)

    @classmethod
    def from_json(
        cls, data: dict[str, Any], longer_names: dict[str, str] | None = None
    ) -> BlockMemory:
        heads = {w: (v[0], float(v[1])) for w, v in data.get("heads", {}).items()}
        names = dict(data.get("names", {}))
        names.update(longer_names or {})
        return cls(heads, names)

    def head(self, word: str) -> tuple[str, float] | None:
        return self.heads.get(word) or self.heads.get(singular(word))


@dataclass(frozen=True)
class Block:
    start: int  # word index, inclusive
    end: int  # exclusive
    kind: str  # anatomy, procedure, device, ... or "" (unread)
    source: str  # mention, memory, composed, word
    cost: float
    share: float = 1.0


@dataclass(frozen=True)
class Reading:
    outcome: str  # link, role, not_a_site, underspecified, unread
    role: str = ""
    kind: str = ""
    source: str = ""
    block: str = ""
    margin: float = 0.0
    why: str = ""
    alternative: str = ""


UNREAD = Reading("unread")


def _words(
    sentence: str, found: re.Match[str]
) -> tuple[list[tuple[str, int, int]], int, int]:
    """The window of words around the mention (folded, with offsets) and the mention's word range."""
    toks = [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(sentence)]
    inside = [
        i for i, (_, a, b) in enumerate(toks) if a >= found.start() and b <= found.end()
    ]
    if not inside:
        return [], 0, 0
    mi, mj = inside[0], inside[-1] + 1
    lo = mi
    while lo > 0 and mi - lo < LEFT:
        prev = toks[lo - 1]
        if _fold(prev[0]) in _PHRASE_STOP or _BREAK.search(
            sentence[prev[2] : toks[lo][1]]
        ):
            break
        lo -= 1
    hi = mj
    while hi < len(toks) and hi - mj < RIGHT:
        nxt = toks[hi]
        word = _fold(nxt[0])
        if (
            word in _PHRASE_STOP
            or word in _REPORT_VERBS
            or word.isdigit()
            or _BREAK.search(sentence[toks[hi - 1][2] : nxt[1]])
        ):
            break
        hi += 1
    window = [(_fold(w), a, b) for w, a, b in toks[lo:hi]]
    return window, mi - lo, mj - lo


class ChunkLattice:
    def __init__(self, memory: BlockMemory):
        self.memory = memory

    # -- nodes ---------------------------------------------------------------------------------

    def _blocks(self, words: list[str], mi: int, mj: int) -> list[Block]:
        n, memory = len(words), self.memory
        blocks: list[Block] = [Block(mi, mj, "anatomy", "mention", C_MEMORY)]
        for i in range(n):
            for j in range(i + 1, n + 1):
                crosses = i < mi < j < mj or mi < i < mj < j
                if crosses or (i >= mi and j <= mj):
                    continue  # never split the mention
                length = j - i
                if length == 1:
                    if mi <= i < mj:
                        continue
                    head = memory.head(words[i])
                    blocks.append(
                        Block(
                            i,
                            j,
                            head[0] if head else "",
                            "word",
                            C_WORD_KNOWN if head else C_WORD_UNKNOWN,
                            head[1] if head else 0.0,
                        )
                    )
                    continue
                if length > RIGHT + 1:
                    continue
                known = memory.names.get(" ".join(words[i:j]))
                if known:
                    blocks.append(Block(i, j, known, "memory", C_MEMORY))
                # Composed: the kind of the last word, or a structure when the block ends with the mention.
                bonus = C_COMPOSED + C_PER_WORD * (length - 1)
                if j == mj and i <= mi:
                    blocks.append(Block(i, j, "anatomy", "composed", bonus))
                elif j > mj or j <= mi:
                    head = memory.head(words[j - 1])
                    if head:
                        inner = sum(
                            1
                            for w in words[max(mj, i) : j - 1]
                            if (h := memory.head(w)) and h[0] != "anatomy"
                        )
                        cost = bonus + C_SHARE * (1.0 - head[1]) + C_OPEN * inner
                        blocks.append(Block(i, j, head[0], "composed", cost, head[1]))
        return blocks

    # -- reading -------------------------------------------------------------------------------

    def candidates(
        self, mention: str, sentence: str, start: int | None = None
    ) -> tuple[list[tuple[str, int, int]], int, int, list[tuple[float, Block]], re.Match[str]] | None:
        """Every block that holds the mention with the best cost of covering the rest (the whole
        construction, cheapest first), the window and the mention's word range. ``None`` if unread."""
        found = locate(mention, sentence, start)
        if not found:
            return None
        window, mi, mj = _words(sentence, found)
        if not window:
            return None
        words = [w for w, _, _ in window]
        blocks = self._blocks(words, mi, mj)
        n = len(words)
        inf = float("inf")
        by_end: dict[int, list[Block]] = {}
        by_start: dict[int, list[Block]] = {}
        for b in blocks:
            by_end.setdefault(b.end, []).append(b)
            by_start.setdefault(b.start, []).append(b)
        left = [0.0] + [inf] * n  # best cost to cover words[:k]
        for k in range(1, n + 1):
            left[k] = min(
                (left[b.start] + b.cost for b in by_end.get(k, ())), default=inf
            )
        right = [inf] * n + [0.0]  # best cost to cover words[k:]
        for k in range(n - 1, -1, -1):
            right[k] = min(
                (right[b.end] + b.cost for b in by_start.get(k, ())), default=inf
            )
        readings = []
        for b in blocks:
            if b.start <= mi and b.end >= mj:
                total = left[b.start] + b.cost + right[b.end]
                if total < inf:
                    readings.append((total, b))
        if not readings:
            return None
        readings.sort(key=lambda r: (r[0], -(r[1].end - r[1].start)))
        return window, mi, mj, readings, found

    def read(self, mention: str, sentence: str, start: int | None = None) -> Reading:
        got = self.candidates(mention, sentence, start)
        if got is None:
            return UNREAD
        window, mi, mj, readings, found = got
        words = [w for w, _, _ in window]
        best_cost, best = readings[0]
        best_class = CLASS_OF_KIND.get(best.kind, "")
        alt = next(
            (
                (c, b)
                for c, b in readings[1:]
                if CLASS_OF_KIND.get(b.kind, "") not in ("", best_class)
            ),
            None,
        )
        margin = (alt[0] - best_cost) if alt else float("inf")
        text = " ".join(words[best.start : best.end])
        reading = self._decide(
            best, best_class, margin, alt, text, sentence, window, found
        )
        return reading

    def _decide(
        self, best, best_class, margin, alt, text, sentence, window, found
    ) -> Reading:
        alternative = ""
        if alt:
            alternative = f"{' '.join(w for w, _, _ in window[alt[1].start : alt[1].end])}:{alt[1].kind}"
        if best.source == "mention" or not best.kind:
            # The mention is alone as a block: look at the "of" constructions around it.
            return self._with_of(
                best, text, margin, alternative, sentence, window, found
            )
        if best.source == "composed" and best.kind in UNTRUSTED_COMPOSED:
            return Reading(
                "unread",
                kind=best.kind,
                source=best.source,
                block=text,
                margin=margin,
                why=f"composed_kind_not_trusted:{best.kind}",
                alternative=alternative,
            )
        if not best_class:
            return Reading(
                "unread",
                kind=best.kind,
                source=best.source,
                block=text,
                margin=margin,
                why="head_not_sure",
                alternative=alternative,
            )
        if (
            margin < MARGIN
            and alt is not None
            and CLASS_OF_KIND.get(alt[1].kind, "") != best_class
        ):
            return Reading(
                "underspecified",
                kind=best.kind,
                source=best.source,
                block=text,
                margin=margin,
                why=f"two_readings:{best.kind}|{alt[1].kind}",
                alternative=alternative,
            )
        if best_class == "link":
            return self._with_of(
                best, text, margin, alternative, sentence, window, found
            )
        role = ROLE_OF_KIND.get(best.kind, "")
        if best.source == "composed" and best.kind in SOURCE_KINDS:
            # "liver extracts", "brain cDNA", "liver genes": the structure is where the molecule comes
            # from; only a remembered name ("liver fatty acid binding protein") is not a structure.
            best_class, role = "role", "source_of"
        return Reading(
            best_class,
            role,
            best.kind,
            best.source,
            text,
            margin,
            f"block_of_kind:{best.kind}",
            alternative,
        )

    # -- "of" ----------------------------------------------------------------------------------

    def _phrase_head(self, sentence: str, end: int) -> tuple[str, str, float]:
        """Head word, kind and share of the noun phrase that starts at ``end``."""
        words: list[str] = []
        prev_end = end
        for m in _TOKEN.finditer(sentence, end):
            if _fold(m.group(0)) in _PHRASE_STOP or _BREAK.search(
                sentence[prev_end : m.start()]
            ):
                break
            words.append(_fold(m.group(0)))
            prev_end = m.end()
            if len(words) >= RIGHT:
                break
        if not words:
            return "", "", 0.0
        phrase = " ".join(words)
        known = self.memory.names.get(phrase)
        if known:
            return phrase, known, 1.0
        head = self.memory.head(words[-1])
        return (words[-1], head[0], head[1]) if head else (words[-1], "", 0.0)

    def _with_of(
        self, best, text, margin, alternative, sentence, window, found
    ) -> Reading:
        base = Reading(
            "link",
            kind=best.kind or "anatomy",
            source=best.source,
            block=text,
            margin=margin,
            alternative=alternative,
        )
        # the mention's block ends at the end of the window; "of <phrase>" may follow it
        block_end = window[best.end - 1][2]
        after = _OF.match(sentence[block_end:])
        if after:
            head, kind, share = self._phrase_head(sentence, block_end + after.end())
            if kind in OBJECT_KINDS and share >= MIN_SHARE:
                return Reading(
                    "not_a_site",
                    ROLE_OF_KIND[kind] if kind == "food" else "device_site",
                    kind,
                    "of",
                    f"{text} of {head}",
                    margin,
                    f"part_of_an_object:{head}",
                    alternative,
                )
        block_start = window[best.start][1]
        before = _OF_BEFORE.search(sentence[:block_start])
        if before:
            head, kind, share = self._phrase_head_before(sentence, before.start())
            if kind in OF_ROLE_KINDS and share >= MIN_SHARE:
                return Reading(
                    "role",
                    ROLE_OF_KIND[kind],
                    kind,
                    "of",
                    f"{head} of {text}",
                    margin,
                    f"object_of:{head}",
                    alternative,
                )
        return base

    def _phrase_head_before(self, sentence: str, end: int) -> tuple[str, str, float]:
        """Head, kind and share of the noun phrase that ends at ``end`` (its last word is the head)."""
        toks = [m for m in _TOKEN.finditer(sentence[:end])]
        if not toks:
            return "", "", 0.0
        last = toks[-1]
        if _BREAK.search(sentence[last.end() : end]):
            return "", "", 0.0
        word = _fold(last.group(0))
        if word in _PHRASE_STOP:
            return "", "", 0.0
        # the whole phrase may be remembered ("liver transplantation")
        words = [word]
        prev_start = last.start()
        for m in reversed(toks[:-1]):
            w = _fold(m.group(0))
            if (
                w in _PHRASE_STOP
                or _BREAK.search(sentence[m.end() : prev_start])
                or len(words) >= RIGHT
            ):
                break
            words.insert(0, w)
            prev_start = m.start()
        for i in range(len(words)):
            known = self.memory.names.get(" ".join(words[i:]))
            if known:
                return " ".join(words[i:]), known, 1.0
        head = self.memory.head(word)
        return (word, head[0], head[1]) if head else (word, "", 0.0)
