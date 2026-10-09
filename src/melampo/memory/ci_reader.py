"""A construction-integration reader for the mention in its phrase (10 October 2026, experiment E5).

What the sources say a person does (Kintsch 1998; Kintsch 2001 on predication; Kintsch & Mangalath
2011 on dual memory; Ericsson & Kintsch 1991 on long-term working memory; Landauer & Dumais 1997 on
the semantic space) and what this module does with it, one to one:

1. **Construction is dumb and exhaustive.** Every reading of the mention is built, whether or not the
   context will keep it: the mention as the *structure* itself, as the site of a *procedure*, of a
   *device*, as the thing a *measurement* is about, as the place a *molecule* comes from, as a piece
   of the *name of something that is not a body site*. Every block of the chunk lattice that holds
   the mention (``ChunkLattice.candidates``) is a node, not only the cheapest one.
2. **Integration is constraint satisfaction.** The nodes (readings and the evidence for them) form a
   network with excitatory links where they fit together and inhibitory links among the alternative
   readings of the same string. Activation is spread, ``A(t+1) = W A(t)``, negatives set to zero and
   the vector divided by its maximum, until it changes by less than 0.001 (Kintsch's stopping rule).
   What survives is the reading that fits all the evidence at once, not the reading of the loudest
   stream. Nothing is voted.
3. **Predication.** The meaning of the mention in this phrase is not its meaning out of context: it
   is the argument adjusted by the predicate (Kintsch 2001): the vector of the head word, the vector
   of the mention and the ``k`` neighbours of the head (out of ``m``) that are most related to the
   mention. "heart" with "rate" and "heart" with "cheese" end in different places of the space.
4. **Dual memory (CI-II).** Two memories feed the network. The *gist* is the topic of the document, a
   vector in the semantic space, compared with the contexts in which readings were seen. The *explicit
   traces* are the readings annotators gave to (mention word, next word, phrase head, previous word)
   in training documents. Retrieval is by cue, from the most specific to the least, and a retrieved
   trace counts as much as its cue is specific (the encoding-specificity principle of retrieval
   structures; a person with domain knowledge retrieves in about 300-400 ms what a novice computes).
   The document being read is **never** in the memory it is read with (leave-one-document-out).
5. **A semantic space.** Latent semantic analysis (log-entropy weighting, truncated SVD) over text of
   the domain and over the definitions of the vocabulary (NCIt). Prototypes of a reading are the
   centroids of the definitions of the classes of its kind.

Constants are set by principle, written here, and **not fitted** on the errors they are measured on:
``INHIBITION`` (competitors inhibit each other as much as half of the strongest support), the weights
of the retrieval levels (the more specific the cue, the more weight), the equal total weight of the
predication and of the bag of context words, and the stopping rule. A run over a grid of them is part
of the probe, so that a result that depends on one setting says so.

Limits: English only (the heads and the definitions are English). The space is only as good as the
text it is built from; with a few thousand abstracts it is small (Landauer used 4.6 million words).
The traces carry the convention of the annotators they come from (MedMentions marks the whole
phrase, CRAFT the organ inside it): a reader trained on one corpus applies that corpus's convention
to the other; that is measured, not hidden. This module decides nothing in the linker.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .chunk_lattice import (
    CLASS_OF_KIND,
    ROLE_OF_KIND,
    BlockMemory,
    ChunkLattice,
)
from .word_senses import _PHRASE_STOP, _fold

STRUCTURE = "structure"
READINGS = (
    STRUCTURE,
    "procedure_site",
    "device_site",
    "inherent_location",
    "inside_a_name",
    "not_a_body_site",
)
# The classes of definitions that give a reading its prototype (NCIt kinds).
PROTOTYPE_KINDS = {
    STRUCTURE: ("anatomy",),
    "procedure_site": ("procedure",),
    "device_site": ("device",),
    "inherent_location": ("process", "property", "activity"),
    "inside_a_name": ("protein", "gene", "chemical"),
    "not_a_body_site": ("food", "organism", "conceptual"),
}

INHIBITION = 0.5  # among the readings of one string, as a share of a unit excitatory link
STOP_CHANGE = 0.001  # Kintsch's stopping rule on the change of the activation vector
MAX_CYCLES = 200
MARGIN = 0.10  # share of the activation by which the best reading must beat the second
PREDICATION_M, PREDICATION_K = 100, 3
CONTEXT_WORDS = 16
# Retrieval levels, most specific first: weight of the cue (encoding specificity).
LEVELS = (
    ("word+next", 1.0), ("word+head", 0.8), ("word+previous", 0.6),
    ("word+next-kind", 0.5), ("word+head-kind", 0.45), ("word+previous-kind", 0.4),
    ("anatomy+next", 0.4), ("anatomy+head", 0.35),
    ("anatomy+next-kind", 0.3), ("anatomy+head-kind", 0.25), ("anatomy+previous-kind", 0.2),
    ("word", 0.15),
)
MIN_KIND_SHARE = 0.6  # a neighbour has a kind when the names ending with it agree this much
TRACE_SUPPORT = 2.0  # a cue seen n times counts n / (n + TRACE_SUPPORT)
NAME_WEIGHT = 0.5  # the string is a known name of a structure

_WORD = re.compile(r"[A-Za-z][A-Za-z\-]+")
_FUNCTION = frozenset(
    """the and for with from that this these those were was are is been being have has had not but
    also than then there their they them which while when where into onto over under between within
    without about after before during among through both each such other more most less some any all
    only very can may might could would should will shall""".split()
)


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def content_words(text: str, minimum: int = 4) -> list[str]:
    """Folded, singular words that carry meaning (no function words)."""
    out = []
    for w in _WORD.findall(text):
        f = _singular(_fold(w))
        if len(f) >= minimum and f not in _FUNCTION and f not in _PHRASE_STOP:
            out.append(f)
    return out


def reading_of_kind(kind: str) -> str | None:
    """The reading a block of this NCIt kind stands for (``None`` when the kind is not acted on)."""
    klass = CLASS_OF_KIND.get(kind)
    if klass == "link":
        return STRUCTURE
    return ROLE_OF_KIND.get(kind) if klass else None


# -- the semantic space ---------------------------------------------------------------------------


class SemanticSpace:
    """Latent semantic analysis: word vectors (unit length) and the global weight of each word."""

    def __init__(self, words: list[str], vectors: np.ndarray, weight: np.ndarray):
        self.words = words
        self.index = {w: i for i, w in enumerate(words)}
        self.vectors = vectors.astype(np.float32)
        self.weight = weight.astype(np.float32)

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        dim: int = 100,
        min_count: int = 3,
        chunk: int = 40,
    ) -> SemanticSpace:
        """Log-entropy weighted term-by-chunk matrix, truncated SVD (Landauer & Dumais 1997)."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import svds

        chunks: list[list[str]] = []
        freq: Counter[str] = Counter()
        for text in texts:
            words = content_words(text, minimum=3)
            for i in range(0, len(words), chunk):
                part = words[i : i + chunk]
                if len(part) >= 8:
                    chunks.append(part)
                    freq.update(part)
        vocab = sorted(w for w, n in freq.items() if n >= min_count)
        index = {w: i for i, w in enumerate(vocab)}
        rows, cols, vals = [], [], []
        for j, part in enumerate(chunks):
            for w, n in Counter(x for x in part if x in index).items():
                rows.append(index[w])
                cols.append(j)
                vals.append(float(n))
        counts = csr_matrix((vals, (rows, cols)), shape=(len(vocab), len(chunks)))
        total = np.asarray(counts.sum(axis=1)).ravel()
        n_chunks = counts.shape[1]
        # global weight: 1 - entropy of the word's distribution over chunks / log(chunks)
        p = counts.multiply(1.0 / np.maximum(total, 1.0)[:, None]).tocsr()
        plogp = p.copy()
        plogp.data = plogp.data * np.log(plogp.data)
        entropy = -np.asarray(plogp.sum(axis=1)).ravel()
        g = 1.0 - entropy / np.log(max(n_chunks, 2))
        local = counts.copy()
        local.data = np.log1p(local.data)
        weighted = local.multiply(g[:, None]).tocsr()
        k = min(dim, min(weighted.shape) - 1)
        u, s, _ = svds(weighted, k=k, random_state=0)
        order = np.argsort(-s)
        vectors = u[:, order] * s[order]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-9)
        return cls(vocab, vectors, g)

    def save(self, path: Path) -> None:
        np.savez_compressed(path, words=np.array(self.words), vectors=self.vectors, weight=self.weight)

    @classmethod
    def load(cls, path: Path) -> SemanticSpace:
        data = np.load(path, allow_pickle=False)
        return cls([str(w) for w in data["words"]], data["vectors"], data["weight"])

    def vec(self, word: str) -> np.ndarray | None:
        i = self.index.get(_singular(_fold(word)))
        return None if i is None else self.vectors[i]

    def vector(self, words: Iterable[str]) -> np.ndarray | None:
        """Sum of the word vectors weighted by their global weight, unit length; ``None`` if no word is known."""
        total = None
        for w in words:
            i = self.index.get(_singular(_fold(w)))
            if i is None:
                continue
            v = self.vectors[i] * max(self.weight[i], 0.05)
            total = v if total is None else total + v
        if total is None:
            return None
        n = float(np.linalg.norm(total))
        return total / n if n > 0 else None

    def neighbours(self, v: np.ndarray, m: int) -> list[int]:
        sims = self.vectors @ v
        return list(np.argpartition(-sims, min(m, len(sims) - 1))[:m])

    def predicate(
        self, predicate: str, argument: Iterable[str], m: int = PREDICATION_M, k: int = PREDICATION_K
    ) -> np.ndarray | None:
        """Kintsch (2001): the predicate, the argument and the ``k`` of the predicate's ``m`` nearest
        neighbours that are most related to the argument, summed. ``None`` if the predicate is unknown."""
        p = self.vec(predicate)
        a = self.vector(argument)
        if p is None:
            return None
        if a is None:
            return p
        near = self.neighbours(p, m)
        best = sorted(near, key=lambda i: -float(self.vectors[i] @ a))[:k]
        total = p + a + sum(self.vectors[i] for i in best)
        n = float(np.linalg.norm(total))
        return total / n if n > 0 else p


def prototypes(
    space: SemanticSpace, definitions: Iterable[tuple[str, str]], cap: int = 20000
) -> dict[str, np.ndarray]:
    """One vector per reading: the centroid of the definitions of the classes of its kinds
    (``definitions`` yields ``(kind, text)``). Not fitted on any labelled mention."""
    sums: dict[str, np.ndarray] = {}
    counts: Counter[str] = Counter()
    of_kind = {k: r for r, ks in PROTOTYPE_KINDS.items() for k in ks}
    for kind, text in definitions:
        reading = of_kind.get(kind)
        if reading is None or counts[reading] >= cap:
            continue
        v = space.vector(content_words(text))
        if v is None:
            continue
        sums[reading] = sums.get(reading, 0) + v
        counts[reading] += 1
    out = {}
    for reading, total in sums.items():
        n = float(np.linalg.norm(total))
        if n > 0:
            out[reading] = (total / n).astype(np.float32)
    return out


# -- the two memories -----------------------------------------------------------------------------


@dataclass
class Cues:
    word: str
    nxt: str = ""
    head: str = ""
    previous: str = ""
    nxt_kind: str = ""
    head_kind: str = ""
    previous_kind: str = ""
    anatomy: bool = True

    def keys(self) -> list[tuple[str, tuple[str, ...]]]:
        out = []
        for level, value in (("word+next", self.nxt), ("word+head", self.head), ("word+previous", self.previous),
                             ("word+next-kind", self.nxt_kind), ("word+head-kind", self.head_kind),
                             ("word+previous-kind", self.previous_kind)):
            if value:
                out.append((level, (self.word, value)))
        if self.anatomy:
            # the head learned from the domain: any structure word followed by this word / ending with it
            for level, value in (("anatomy+next", self.nxt), ("anatomy+head", self.head),
                                 ("anatomy+next-kind", self.nxt_kind), ("anatomy+head-kind", self.head_kind),
                                 ("anatomy+previous-kind", self.previous_kind)):
                if value:
                    out.append((level, ("*", value)))
        out.append(("word", (self.word,)))
        return out


def kind_of_word(memory: BlockMemory, word: str) -> str:
    head = memory.head(word) if word else None
    return head[0] if head and head[1] >= MIN_KIND_SHARE else ""


def make_cues(lattice: ChunkLattice, text: str, start: int, end: int) -> Cues | None:
    """The cues of the word at ``text[start:end]``: next and previous word (inside the noun phrase),
    the head of the phrase to its right, and the kind of each. Same function for the memory (written
    from annotated documents) and for the reading, so a cue means the same on both sides."""
    memory = lattice.memory
    word = _singular(_fold(text[start:end]))
    if not word:
        return None
    cues = Cues(word=word, anatomy=kind_of_word(memory, word) == "anatomy")
    nxt = _WORD.match(text, end + 1) if end < len(text) and text[end] in " -" else None
    if nxt and _fold(nxt.group(0)) not in _PHRASE_STOP:
        cues.nxt = _singular(_fold(nxt.group(0)))
    before = list(_WORD.finditer(text[max(0, start - 40) : start]))
    if before and text[max(0, start - 40) : start].rstrip(" -").endswith(before[-1].group(0)):
        prev = _fold(before[-1].group(0))
        if prev not in _PHRASE_STOP:
            cues.previous = _singular(prev)
    if cues.nxt:
        head, kind, share = lattice._phrase_head(text, end)
        if head:
            cues.head = _singular(head.split()[-1])
            cues.head_kind = kind if kind and share >= MIN_KIND_SHARE else kind_of_word(memory, cues.head)
    cues.nxt_kind = kind_of_word(memory, cues.nxt)
    cues.previous_kind = kind_of_word(memory, cues.previous)
    return cues


class TraceMemory:
    """Explicit traces and gist, written from annotated training documents.

    ``add(doc, text, start, end, role, lattice)`` stores one annotation: for every structure word of
    the annotated phrase, the cues it had in the text (the next word, the phrase head, the previous
    word, and the kind of each) and the role the annotators gave the phrase. The gist of the role is the sum of the document's word
    vectors. ``recall`` excludes the document being read.
    """

    def __init__(self, space: SemanticSpace | None = None):
        self.space = space
        self.total: dict[tuple, Counter[str]] = defaultdict(Counter)
        self.by_doc: dict[tuple[str, tuple], Counter[str]] = defaultdict(Counter)
        self.gist_sum: dict[str, np.ndarray] = {}
        self.gist_doc: dict[tuple[str, str], np.ndarray] = {}
        self.docs: dict[str, np.ndarray] = {}
        self.size = 0

    def add(self, doc: str, text: str, start: int, end: int, role: str, lattice: ChunkLattice, doc_vector=None) -> None:
        """One annotated phrase ``text[start:end]`` with its role: every word of the phrase that is a
        structure name leaves a trace with the cues it had in the text around it."""
        for m in _WORD.finditer(text, start, end):
            cues = make_cues(lattice, text, m.start(), m.end())
            if cues is None or not cues.anatomy:
                continue
            for key in cues.keys():
                self.total[key][role] += 1
                self.by_doc[(doc, key)][role] += 1
        self.size += 1
        if doc_vector is not None and (role, doc) not in self.gist_doc:
            self.gist_doc[(role, doc)] = doc_vector
            self.gist_sum[role] = self.gist_sum.get(role, 0) + doc_vector

    def recall(self, cues: Cues, doc: str | None) -> list[tuple[str, float, Counter[str]]]:
        """(level, weight, role counts) for every cue that finds traces outside ``doc``."""
        out = []
        weight = dict(LEVELS)
        for level, key in cues.keys():
            counts = Counter(self.total.get((level, key), ()))
            for role, n in self.by_doc.get((doc, (level, key)), {}).items():
                counts[role] -= n
            counts = +counts
            if counts:
                out.append((level, weight[level], counts))
        return out

    def gist(self, doc: str | None, vector: np.ndarray | None) -> dict[str, float]:
        """How close the document's topic is to the documents in which each role was seen
        (cosine to the centroid of the others; the document itself is subtracted)."""
        if vector is None:
            return {}
        out = {}
        for role, total in self.gist_sum.items():
            own = self.gist_doc.get((role, doc))
            centroid = total - own if own is not None else total
            n = float(np.linalg.norm(centroid))
            if n > 0:
                out[role] = max(0.0, float(vector @ (centroid / n)))
        return out


# -- the reading ----------------------------------------------------------------------------------


@dataclass
class CIReading:
    decision: str  # a reading id, or "underspecified"
    top: str
    margin: float
    shares: dict[str, float]
    cycles: int
    nodes: list[str] = field(default_factory=list)  # the evidence that was in the network, for the audit

    @property
    def p_not_structure(self) -> float:
        return 1.0 - self.shares.get(STRUCTURE, 0.0)


class CIReader:
    def __init__(
        self,
        lattice: ChunkLattice,
        space: SemanticSpace | None = None,
        protos: dict[str, np.ndarray] | None = None,
        traces: TraceMemory | None = None,
        inhibition: float = INHIBITION,
        use: frozenset[str] | None = None,
    ):
        self.lattice = lattice
        self.space = space
        self.protos = protos or {}
        self.traces = traces
        self.inhibition = inhibition
        # which families of evidence are in the network (for the ablations of the probe)
        self.use = use or frozenset({"blocks", "head", "predication", "context", "traces", "gist"})
        self.scale = self._calibrate()

    def _calibrate(self) -> float:
        """The contrast that counts as strong relatedness: the 95th percentile of the contrast of 2000
        words of the space (no label is involved), so that a strong relation weighs as much as a block."""
        if self.space is None or not self.protos:
            return 1.0
        names = [r for r in READINGS if r in self.protos]
        matrix = np.stack([self.protos[r] for r in names])
        sims = self.space.vectors[:: max(1, len(self.space.words) // 2000)] @ matrix.T
        above = sims - sims.mean(axis=1, keepdims=True)
        return float(max(np.percentile(above[above > 0], 95), 1e-6))

    # -- construction -------------------------------------------------------------------------

    def construct(self, mention: str, sentence: str, start: int | None, doc: str | None = None, doc_vector=None):
        """The evidence nodes: ``(label, activation, {reading: link weight})``."""
        nodes: list[tuple[str, float, dict[str, float]]] = []
        if "name" in self.use:
            nodes.append(("name", 1.0, {STRUCTURE: NAME_WEIGHT}))
        got = self.lattice.candidates(mention, sentence, start)
        cues = Cues(_singular(_fold(mention.split()[-1])) if mention.split() else "")
        if got is not None:
            window, mi, mj, blocks, found = got
            words = [w for w, _, _ in window]
            cues = make_cues(self.lattice, sentence, found.start(), found.end()) or cues
            if "blocks" in self.use:
                best = blocks[0][0]
                seen: set[str] = set()
                for total, b in blocks:
                    reading = reading_of_kind(b.kind) if b.kind else None
                    if reading is None or reading in seen:
                        continue
                    seen.add(reading)
                    act = float(np.exp(-(total - best)))
                    nodes.append((f"block:{' '.join(words[b.start:b.end])}:{b.kind}", act, {reading: 1.0}))
            if mj < len(words):
                head, kind, share = self.lattice._phrase_head(sentence, found.end())
                reading = reading_of_kind(kind) if kind else None
                if "head" in self.use and reading is not None:
                    nodes.append((f"head:{head}:{kind}", float(share), {reading: 1.0}))
            elif mi > 0:
                head, kind, share = self.lattice._phrase_head_before(sentence, found.start())
                reading = reading_of_kind(kind) if kind else None
                if "head" in self.use and reading is not None and reading != STRUCTURE:
                    nodes.append((f"before:{head}:{kind}", float(share) * 0.5, {reading: 1.0}))
        space = self.space
        if space is not None and self.protos:
            pro = self.protos
            names = [r for r in READINGS if r in pro]
            matrix = np.stack([pro[r] for r in names])

            def contrast(vector) -> dict[str, float]:
                sims = matrix @ vector
                above = sims - float(sims.mean())
                return {r: float(min(1.0, max(0.0, a) / self.scale)) for r, a in zip(names, above, strict=True)}

            ctx = [w for w in content_words(sentence) if w not in cues.word.split()][:CONTEXT_WORDS * 4]
            if "predication" in self.use:
                predicate = cues.head or cues.nxt or cues.previous
                if predicate:
                    v = space.predicate(predicate, [cues.word])
                    if v is not None:
                        nodes.append((f"predication:{predicate}", 1.0, contrast(v)))
            if "context" in self.use and ctx:
                # the bag of context words counts as much in total as the predication does
                near = ctx[:CONTEXT_WORDS]
                share = 1.0 / len(near)
                for w in near:
                    v = space.vec(w)
                    if v is not None:
                        nodes.append((f"context:{w}", share, contrast(v)))
        if self.traces is not None and "traces" in self.use:
            for level, weight, counts in self.traces.recall(cues, doc):
                n = sum(counts.values())
                act = weight * n / (n + TRACE_SUPPORT)
                nodes.append((f"trace:{level}", act, {r: c / n for r, c in counts.items() if r in READINGS}))
        if self.traces is not None and "gist" in self.use and doc_vector is not None:
            gist = self.traces.gist(doc, doc_vector)
            if gist:
                mean = sum(gist.values()) / len(gist)
                links = {r: max(0.0, g - mean) for r, g in gist.items() if r in READINGS}
                if any(links.values()):
                    nodes.append(("gist", 1.0, links))
        return nodes

    # -- integration --------------------------------------------------------------------------

    def integrate(self, nodes) -> tuple[dict[str, float], int]:
        n_read = len(READINGS)
        size = n_read + len(nodes)
        w = np.zeros((size, size), dtype=np.float64)
        for i in range(n_read):
            for j in range(n_read):
                if i != j:
                    w[i, j] = -self.inhibition / (n_read - 1)
        a = np.zeros(size)
        for k, (_, act, links) in enumerate(nodes):
            row = n_read + k
            a[row] = act
            for reading, weight in links.items():
                if reading in READINGS and weight > 0:
                    i = READINGS.index(reading)
                    w[i, row] = w[row, i] = weight
        cycles = 0
        for cycles in range(1, MAX_CYCLES + 1):
            new = np.maximum(w @ a + a, 0.0)  # an activated node keeps its own activation
            top = new.max()
            if top <= 0:
                break
            new = new / top
            if np.abs(new - a).max() < STOP_CHANGE:
                a = new
                break
            a = new
        reading_act = {r: float(a[i]) for i, r in enumerate(READINGS)}
        return reading_act, cycles

    def read(self, mention: str, sentence: str, start: int | None = None, doc: str | None = None,
             doc_vector=None) -> CIReading:
        nodes = self.construct(mention, sentence, start, doc, doc_vector)
        act, cycles = self.integrate(nodes)
        total = sum(act.values())
        shares = {r: (v / total if total > 0 else 0.0) for r, v in act.items()}
        ranked = sorted(shares.items(), key=lambda kv: -kv[1])
        top, best = ranked[0]
        margin = best - ranked[1][1]
        decision = top if margin >= MARGIN else "underspecified"
        return CIReading(decision, top, margin, shares, cycles, [f"{n[0]}={n[1]:.2f}" for n in nodes])


def role_to_reading(role: str) -> str:
    """The reading a gold role stands for ("structure" and "disease" are one: the link is kept)."""
    return STRUCTURE if role in ("structure", "disease", "", None) else role


__all__ = [
    "READINGS",
    "STRUCTURE",
    "BlockMemory",
    "CIReader",
    "CIReading",
    "Cues",
    "SemanticSpace",
    "TraceMemory",
    "content_words",
    "prototypes",
    "reading_of_kind",
    "role_to_reading",
]
