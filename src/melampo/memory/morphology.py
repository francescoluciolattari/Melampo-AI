"""Stage 3: morphological candidates from Greek and Latin roots ("epatico", "nefrolitiasi").

A reader who meets "epatomegalia" or "nephrectomy" for the first time still knows the organ: the
word is decomposed into its parts before its meaning is checked (Rastle, Davis & New 2004, morpho-
orthographic segmentation). Teaching morphemes helps with the words taught much more than with new
words, and very little with comprehension (meta-analysis: SMD 0.83, 0.31, 0.13), so roots are used
to **propose**, never to accept:

* on an abstention, the structures the roots point to are written in the trace (``morphology``),
  for the review queue and as a hint of what the mention may be about;
* on an accepted link, a root of the same structure is one more independent route to it (it does
  not use the curated names) and counts as a mechanism in the profile; a root of another structure
  is recorded and nothing more ("biopsia epatica" may name the liver or the needle path).

The roots and the words that start like a root but are not one ("cardinale", "gastrocnemio") are
data (``data/linking/morphology_roots.json``) with their sources.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "linking" / "morphology_roots.json"
)
_WORD = re.compile(r"[a-z]+")


def _fold(text: str) -> str:
    return (
        unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    )


@dataclass
class Morphology:
    roots: dict[str, tuple[str, ...]] = field(default_factory=dict)
    not_words: tuple[str, ...] = ()
    prefixes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    min_extra_letters: int = 2

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Morphology:
        return cls(
            roots={k: tuple(v) for k, v in data["roots"].items()},
            not_words=tuple(data.get("not", ())),
            prefixes={k: tuple(v) for k, v in data.get("prefixes", {}).items()},
            min_extra_letters=int(data.get("min_extra_letters", 2)),
        )

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> Morphology:
        return cls.from_json(json.loads(path.read_text("utf-8")))

    def proposes(self, mention: str) -> tuple[str, ...]:
        """The structures the roots of the mention's words point to, in order of appearance."""
        found: list[str] = []
        for word in _WORD.findall(_fold(mention)):
            if any(word.startswith(n) for n in self.not_words):
                continue
            for structure, roots in self.roots.items():
                if any(
                    word.startswith(root)
                    and (
                        len(word) - len(root) >= self.min_extra_letters or word == root
                    )
                    for root in roots
                ):
                    if structure not in found:
                        found.append(structure)
        return tuple(found)

    def names(self, structure: str, cid: str) -> bool:
        """Is ``cid`` (a class, with its side or number) the ``structure`` a root proposes?"""
        prefixes = self.prefixes.get(structure)
        if prefixes:
            return any(cid == p or cid.startswith(p) for p in prefixes)
        return (
            cid == structure
            or re.fullmatch(
                rf"{re.escape(structure)}(?:_(?:left|right))?(?:_\d+)?", cid
            )
            is not None
        )
