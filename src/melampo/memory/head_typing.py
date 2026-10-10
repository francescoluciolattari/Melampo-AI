"""Type the head word of a noun phrase from UMLS, for the words the block memory does not know.

The block memory (NCIt, ``block_memory.json``) gives the kind of a head word ("rate": property,
"donation": procedure). A word it has not seen has no kind, and the phrase then cannot be read. UMLS
names far more words. ``UmlsHeadTyper`` asks it for the concepts whose name is exactly the word, maps
each concept's semantic types to the lattice's kinds, and answers ``(kind, share)`` where ``share`` is
the fraction of the concepts that give that kind -- the same quantity the block memory keeps, so the
lattice treats both alike (a share below ``MIN_SHARE`` is not acted on).

What it will not do: say anything when UMLS is not reachable (the answer is ``None``, never a guess; the
failure is counted in ``lost``), or give the catch-all kind ("other" says nothing). The semantic-type
groups are the ones ``scripts/phrase_knowledge.py`` uses for the phrase arm (E4a), so the two arms read
UMLS the same way. A TUI group is a coarse map: a single word often has concepts of several kinds ("rate"
is a quantity, a measurement and a rating), which is why the share matters.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

ANATOMY = frozenset("T017 T018 T021 T022 T023 T024 T029 T030 T031 T025 T026".split())
DISEASE = frozenset("T047 T191 T046 T048 T019 T020 T033 T184 T037 T049 T190 T050".split())
PROCEDURE = frozenset("T058 T059 T060 T061 T063 T062".split())
DEVICE = frozenset("T073 T074 T075 T203".split())
PROCESS = frozenset("T038 T039 T040 T041 T042 T043 T044 T045 T067 T068 T069 T070 T169 T032".split())
PROPERTY = frozenset("T201 T081 T080 T034".split())
PROTEIN = frozenset("T116 T126 T028 T087 T114 T192 T129 T085 T086".split())
CHEMICAL = frozenset("T109 T121 T197 T196 T131 T125 T127 T195 T200 T104 T120 T130 T122 T123".split())
ORGANISM = frozenset("T001 T002 T004 T005 T007 T008 T010 T011 T012 T013 T014 T015 T016 T194 T204".split())
FOOD = frozenset({"T168"})

# first matching group wins; the value is the lattice's kind (``block_memory.json``)
KIND_OF_GROUP = (
    ("anatomy", ANATOMY),
    ("disease", DISEASE),
    ("procedure", PROCEDURE),
    ("device", DEVICE),
    ("food", FOOD),
    ("protein", PROTEIN),
    ("chemical", CHEMICAL),
    ("organism", ORGANISM),
    ("process", PROCESS),
    ("property", PROPERTY),
)


def kind_of_tuis(tuis) -> str:
    have = set(tuis)
    for kind, group in KIND_OF_GROUP:
        if have & group:
            return kind
    return ""


class UmlsHeadTyper:
    """``typer(word) -> (kind, share) | None`` from ``exact(string) -> [{"types": [TUI...]}] | None``."""

    def __init__(self, exact: Callable[[str], list[dict] | None], min_concepts: int = 1):
        self.exact = exact
        self.min_concepts = min_concepts
        self.lost = 0
        self.asked = 0
        self._memo: dict[str, tuple[str, float] | None] = {}

    def __call__(self, word: str) -> tuple[str, float] | None:
        if word in self._memo:
            return self._memo[word]
        self.asked += 1
        found = self.exact(word)
        if found is None:
            self.lost += 1
            return None  # not memoised as an answer: the word may be asked again after a retry
        kinds = Counter(k for c in found if (k := kind_of_tuis(c.get("types", ()))))
        total = sum(kinds.values())
        answer = None
        if total >= self.min_concepts:
            kind, n = sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            answer = (kind, n / total)
        self._memo[word] = answer
        return answer


__all__ = ["KIND_OF_GROUP", "UmlsHeadTyper", "kind_of_tuis"]
