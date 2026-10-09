"""NCIt kinds: the kind of a name, of a word, and of the words that end names (shared by the scripts).

The kind of a name is the first of ``KIND_ROOTS`` among its ancestors in the NCI Thesaurus OBO file;
retired classes are skipped. Used by ``phrase_probe.py`` (experiment) and ``build_block_memory.py``
(the data the chunk lattice reads).
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melampo.memory.word_senses import _fold  # noqa: E402

WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")

# NCIt classes that decide the kind of a name; the first ancestor in this order wins.
KIND_ROOTS = (
    ("anatomy", "NCIT:C12219"),
    ("disease", "NCIT:C7057"),
    ("procedure", "NCIT:C25218"),
    ("device", "NCIT:C97325"),
    ("food", "NCIT:C1949"),
    ("protein", "NCIT:C17021"),
    ("gene", "NCIT:C16612"),
    ("chemical", "NCIT:C1908"),
    ("organism", "NCIT:C14250"),
    ("process", "NCIT:C17828"),
    ("property", "NCIT:C20189"),
    ("activity", "NCIT:C43431"),
    ("conceptual", "NCIT:C20181"),
)
RETIRED = "NCIT:C28428"


# The same roles from the semantic types of a MedMentions label (UMLS semantic network).
ROLE_OF_TYPE = {
    **dict.fromkeys(("T058", "T059", "T060", "T061", "T063"), "procedure_site"),
    **dict.fromkeys(("T073", "T074", "T075", "T203"), "device_site"),
    **dict.fromkeys(
        (
            "T032",
            "T038",
            "T039",
            "T040",
            "T041",
            "T042",
            "T043",
            "T044",
            "T045",
            "T067",
            "T068",
            "T069",
            "T070",
            "T169",
            "T201",
            "T081",
            "T080",
        ),
        "inherent_location",
    ),
    **dict.fromkeys(
        (
            "T028",
            "T087",
            "T114",
            "T116",
            "T121",
            "T123",
            "T126",
            "T129",
            "T109",
            "T130",
            "T131",
            "T125",
            "T127",
            "T167",
            "T104",
            "T120",
        ),
        "inside_a_name",
    ),
    **dict.fromkeys(
        (
            "T082",
            "T098",
            "T099",
            "T100",
            "T101",
            "T097",
            "T168",
            "T170",
            "T071",
            "T072",
            "T204",
            "T007",
            "T005",
            "T004",
            "T001",
        ),
        "not_a_body_site",
    ),
}


def singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


# -- NCIt kinds ----------------------------------------------------------------------------------


def read_obo(lines):
    """(id, name, synonyms, parents, obsolete) for each [Term]."""
    term = None
    for line in lines:
        line = line.rstrip("\n")
        if line == "[Term]":
            if term:
                yield term
            term = {"id": "", "name": "", "syn": [], "isa": [], "obs": False}
        elif line.startswith("["):
            if term:
                yield term
            term = None
        elif term is not None:
            if line.startswith("id: "):
                term["id"] = line[4:]
            elif line.startswith("name: "):
                term["name"] = line[6:]
            elif line.startswith("is_obsolete: true"):
                term["obs"] = True
            elif line.startswith("is_a: "):
                term["isa"].append(line[6:].split(" ")[0])
            elif line.startswith("synonym: "):
                found = re.match(r'synonym: "(.*)" (\w+)', line)
                if found:
                    term["syn"].append(found.group(1))
    if term:
        yield term


class Kinds:
    """Kind of an NCIt name, of a single word, and of the words that end NCIt names."""

    def __init__(self, terms, skip_name=None):
        terms = [t for t in terms if t["id"] and not t["obs"]]
        parents = {t["id"]: t["isa"] for t in terms}
        memo: dict[str, frozenset[str]] = {}

        def ancestors(node: str) -> frozenset[str]:
            if node in memo:
                return memo[node]
            memo[node] = frozenset()
            seen, stack = set(), list(parents.get(node, ()))
            while stack:
                p = stack.pop()
                if p not in seen:
                    seen.add(p)
                    stack.extend(parents.get(p, ()))
            memo[node] = frozenset(seen)
            return memo[node]

        self.kind_of_id: dict[str, str] = {}
        self.names: dict[str, str] = {}
        self.preferred: list[tuple[str, str]] = []
        last = defaultdict(Counter)
        last_single = defaultdict(Counter)
        self.names_single: dict[str, str] = {}
        self.vocabulary: set[str] = set()
        for t in terms:
            up = ancestors(t["id"]) | {t["id"]}
            if RETIRED in up or t["name"].lower().startswith("obsolete"):
                continue
            if skip_name and skip_name(t["name"]):
                continue
            pairs = [(k, root) for k, root in KIND_ROOTS if root in up]
            matched = [k for k, _ in pairs]
            kind = matched[0] if matched else None
            if kind is None:
                continue
            self.kind_of_id[t["id"]] = kind
            # Under two kind roots that are not one inside the other ("sterol": food and chemical) the
            # kind is ambiguous; "procedure" inside "activity" is not.
            specific = [
                k
                for k, root in pairs
                if not any(
                    root in ancestors(other) for _, other in pairs if other != root
                )
            ]
            single = len(specific) == 1
            for i, name in enumerate([t["name"], *t["syn"]]):
                words = [_fold(w) for w in WORD.findall(name)]
                if not words or len(words) > 6:
                    continue
                self.vocabulary.update(w for w in words if len(w) >= 4)
                key = " ".join(words)
                self.names.setdefault(key, kind)
                last[singular(words[-1])][kind] += 1
                if single:
                    self.names_single.setdefault(key, kind)
                    last_single[singular(words[-1])][kind] += 1
                if i == 0 and len(words) <= 4:
                    self.preferred.append((name, kind))
        self.last = last
        self.last_single = last_single  # same, from names of one kind only

    @classmethod
    def from_obo(cls, path: Path, skip_name=None) -> Kinds:
        with open(path, encoding="utf-8") as handle:
            return cls(list(read_obo(handle)), skip_name)

    def word_kind(self, word: str) -> tuple[str | None, float]:
        """Kind of a head word: the NCIt class named by the word itself, else the dominant kind of
        the names ending with it (share of those names), else None."""
        w = _fold(word)
        for form in (w, singular(w)):
            if form in self.names:
                return self.names[form], 1.0
        counts = self.last.get(singular(w))
        if not counts:
            return None, 0.0
        kind, n = counts.most_common(1)[0]
        total = sum(counts.values())
        if total < 3:
            return None, 0.0
        return kind, n / total
