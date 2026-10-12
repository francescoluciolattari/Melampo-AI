"""Where a lexicon entry, a rule or a constant comes from, and whether it was added after an error was seen.

The proposal reviewed on 12 October 2026 asks that every alias, rule or constant added *after* seeing an
error be marked, so that it is never counted on the items that suggested it: a fix measured on the case it
was written for is not evidence. ``data/linking/provenance.json`` lists those entries, with the surface
forms they act on; ``gold_set.evaluate`` reports the items they touch apart from the others.

An entry::

    {"id": "gb_blood_count", "file": "src/melampo/memory/anatomy_linker.py", "kind": "rule",
     "added": "2026-10-06", "post_error": true, "seen_on": "E3C Italian cases",
     "surfaces": ["GB"], "note": "..."}

``surfaces`` are matched as whole words (case-folded) in the mention and in its sentence.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "linking" / "provenance.json"


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


@dataclass(frozen=True)
class Entry:
    id: str
    file: str
    kind: str
    added: str
    post_error: bool
    seen_on: str
    surfaces: tuple[str, ...]
    note: str = ""


class Provenance:
    def __init__(self, entries: list[Entry]):
        self.entries = entries
        self._patterns = [
            (e, re.compile(r"(?<!\w)(?:" + "|".join(re.escape(_fold(s)) for s in e.surfaces) + r")(?!\w)"))
            for e in entries
            if e.surfaces
        ]

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> Provenance:
        if not Path(path).exists():
            return cls([])
        data = json.loads(Path(path).read_text("utf-8"))
        return cls([
            Entry(e["id"], e.get("file", ""), e.get("kind", ""), e.get("added", ""), bool(e.get("post_error")),
                  e.get("seen_on", ""), tuple(e.get("surfaces", ())), e.get("note", ""))
            for e in data.get("entries", [])
        ])

    def post_error(self) -> list[Entry]:
        return [e for e in self.entries if e.post_error]

    def touches(self, mention: str, sentence: str = "") -> list[str]:
        """Ids of the post-error entries whose surface forms are in the mention or its sentence."""
        text = _fold(f"{mention} {sentence}")
        return [e.id for e, rx in self._patterns if e.post_error and rx.search(text)]


__all__ = ["DEFAULT_PATH", "Entry", "Provenance"]
