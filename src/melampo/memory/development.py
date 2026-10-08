"""Does the text talk about a developing organism? And does the mention's name also name a developing structure?

The ontology says it: UBERON has ``developing anatomical structure`` and ``embryonic structure``, and
a few names are shared between such a structure and an adult one ("aortic arch" is a synonym of
the pharyngeal arch artery, an embryonic vessel, and of the arch of the aorta; "phallus", "mesenteron",
"otocyst"). The mention's name is read from the ontology, so there is no entry per structure.

What the text says is read from stage words (embryo, fetal, E12.5, gestation): a closed class,
as for units and sides. The two together make a **piece of evidence**, not a veto. Measured on
CRAFT (8 October 2026): in documents about embryos, the annotators linked "aortic arch" to the adult
arch of the aorta 25 times out of 28 ("the definitive aortic arch", "the left-sided aortic arch" of
an embryo), and to the embryonic vessel 3 times. A veto on the frame would lose 25 correct links to
avoid 3 errors. What separates the 3 is local ("fourth aortic arch artery", "aortic arch arteries":
a longer name of the ontology), which another check reads. The frame is therefore recorded in the
profile as a conflict, for the review queue and for the gold set to price.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "linking" / "developmental_frame.json"
)


@dataclass(frozen=True)
class DevelopmentalFrame:
    pattern: re.Pattern[str] | None = None
    document_terms: int = 3
    sentence_terms: int = 1

    @classmethod
    def empty(cls) -> DevelopmentalFrame:
        return cls()

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> DevelopmentalFrame:
        data = json.loads(path.read_text("utf-8"))
        joined = "|".join(f"(?:{p})" for p in data["patterns"])
        return cls(
            re.compile(rf"(?<![A-Za-z0-9])(?:{joined})(?![A-Za-z0-9])", re.IGNORECASE),
            int(data.get("document_terms", 3)),
            int(data.get("sentence_terms", 1)),
        )

    def terms(self, text: str) -> set[str]:
        """The distinct stage words in the text."""
        if not self.pattern or not text:
            return set()
        return {m.group(0).lower() for m in self.pattern.finditer(text)}

    def developmental(self, sentence: str, document: str = "") -> tuple[bool, str]:
        """Whether the text is about a developing organism, and the words that say so."""
        here = self.terms(sentence)
        if len(here) >= self.sentence_terms:
            return True, ",".join(sorted(here)[:3])
        there = self.terms(document)
        if len(there) >= self.document_terms:
            return True, "document:" + ",".join(sorted(there)[:3])
        return False, ""
