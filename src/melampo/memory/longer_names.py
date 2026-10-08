"""A mention inside the known name of another kind of thing (8-9 October 2026).

"International **Prostate** Symptom Score", "**Liver** Fatty Acid Binding Protein": the text writes the
name of a scale, a protein, a gene or a substance, and the structure's name is one of its words.
Syntax does not tell this from "liver gene expression" (experiment ``head_probe``, AUC 0.55);
knowing the longer name does. The names come from open ontologies and carry the kind the
ontology gives them (``data/linking/longer_names.json``, built by ``scripts/build_longer_names.py``):
protein, gene, chemical, assessment_tool. Names of anatomy, diseases, organisms, cells and procedures
are not in the file, so "liver cancer", "mouse brain" and "liver biopsy" keep the structure.

The check is exact: the words of a listed name must stand in the sentence in the same order, with
the mention among them. Nothing is guessed from a part of a name.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .word_senses import locate

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "linking" / "longer_names.json"
_WORD = re.compile(r"[A-Za-z0-9]+")
MAX_WORDS = 9


def _tokens(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0).lower(), m.start(), m.end()) for m in _WORD.finditer(text)]


@dataclass(frozen=True)
class LongerNames:
    names: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> LongerNames:
        if not path.exists():
            return cls()
        return cls.from_json(json.loads(path.read_text("utf-8")))

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> LongerNames:
        return cls(dict(data.get("names", {})))

    def containing(
        self, mention: str, sentence: str, start: int | None = None
    ) -> tuple[str, str]:
        """(name, kind) of the longest listed name that the sentence writes and that has the whole
        mention among its words, else ("", "")."""
        if not self.names:
            return "", ""
        found = locate(mention, sentence, start)
        if not found:
            return "", ""
        words = _tokens(sentence)
        inside = [
            i
            for i, (_, a, b) in enumerate(words)
            if a >= found.start() and b <= found.end()
        ]
        if not inside:
            return "", ""
        first, last = inside[0], inside[-1] + 1
        best = ("", "")
        for a in range(max(0, last - MAX_WORDS), first + 1):
            for b in range(last, min(len(words), a + MAX_WORDS) + 1):
                if (a, b) == (first, last):
                    continue
                key = " ".join(w for w, _, _ in words[a:b])
                kind = self.names.get(key)
                if kind and len(key) > len(best[0]):
                    best = (key, kind)
        return best
