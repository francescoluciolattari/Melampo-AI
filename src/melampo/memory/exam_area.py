"""The area of the exam: the body region the exam studies, read from the exam's name.

A radiologist reading "RM pelvi: anca sinistra con regolare segnale" knows before reading "anca"
that the exam is of the pelvis, and reads every later name against that expectation. The area is
the "Region Imaged" of the LOINC/RSNA Radiology Playbook (with Imaging Focus and Laterality, the
anatomic location of a radiology procedure). It is read only where the exam is named: a modality
word ("RM", "CT", "ecografia", "X-ray") with the region words right before or after it ("TC
torace-addome", "CT of the chest, abdomen and pelvis", "brain MRI", "abdominal CT"). Reading stops
at the first word that is neither a region nor a connecting word, so "MRI showed a lesion of the
femur" names no area, and the clinical history ("pregressa frattura del femore") never makes one.

The area is an expectation, not a rule. A structure of the area (or of the whole body) supports
the link; a structure of a neighbouring area is neutral (a chest CT shows the upper abdomen, a
pelvic CT the femoral heads); a structure far from the area is a prediction error, which the
linker records and which sends an ambiguous form to the models. An incidental finding outside the
area is normal radiology, so the area never vetoes a name on its own.

Cues and adjacency are data (``data/linking/exam_areas.json``) with their sources.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "linking" / "exam_areas.json"
)
WHOLE_BODY = "whole_body"
EXPECTED, ADJACENT, OUTSIDE, UNKNOWN = "expected", "adjacent", "outside", "unknown"

_TOKEN = re.compile(r"[A-Za-zÀ-ÿ0-9]+")
_STOP = re.compile(r"[.;:!?\n()\[\]]")


@dataclass
class ExamAreas:
    modality: re.Pattern[str] | None = None
    whole_body: re.Pattern[str] | None = None
    regions: dict[str, re.Pattern[str]] = field(default_factory=dict)
    connectors: frozenset[str] = frozenset()
    adjacent: dict[str, frozenset[str]] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> ExamAreas:
        return cls()

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ExamAreas:
        adjacent: dict[str, set[str]] = {}
        for a, b in data.get("adjacent", ()):
            adjacent.setdefault(a, set()).add(b)
            adjacent.setdefault(b, set()).add(a)
        return cls(
            modality=re.compile(data["modality"], re.IGNORECASE),
            whole_body=re.compile(data["whole_body"], re.IGNORECASE),
            regions={
                name: re.compile(pattern, re.IGNORECASE)
                for name, pattern in data["regions"].items()
            },
            connectors=frozenset(w.lower() for w in data.get("connectors", ())),
            adjacent={k: frozenset(v) for k, v in adjacent.items()},
        )

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> ExamAreas:
        return cls.from_json(json.loads(path.read_text("utf-8")))

    def _regions_of(self, word: str) -> set[str]:
        return {
            name for name, pattern in self.regions.items() if pattern.fullmatch(word)
        }

    def of(self, text: str) -> frozenset[str]:
        """The areas named by the exam names in ``text`` (empty when no exam is named)."""
        if self.modality is None or not text:
            return frozenset()
        found: set[str] = set()
        for match in self.modality.finditer(text):
            if self.whole_body is not None and self.whole_body.search(
                text, match.start(), min(len(text), match.end() + 30)
            ):
                found.add(WHOLE_BODY)
            found |= self._walk(text, match.end(), forward=True)
            found |= self._walk(text, match.start(), forward=False)
        if self.whole_body is not None and self.whole_body.search(text):
            found.add(WHOLE_BODY)
        return frozenset(found)

    def _walk(self, text: str, at: int, forward: bool) -> set[str]:
        """Region words next to the modality word, through connecting words only."""
        stretch = text[at:] if forward else text[:at]
        tokens = list(_TOKEN.finditer(stretch))
        if not forward:
            tokens.reverse()
        found: set[str] = set()
        previous = 0 if forward else len(stretch)
        for count, token in enumerate(tokens):
            gap = (
                stretch[previous : token.start()]
                if forward
                else stretch[token.end() : previous]
            )
            if _STOP.search(gap) or count >= 8:
                break
            word = token.group(0)
            regions = self._regions_of(word)
            if regions:
                found |= regions
            elif word.lower() not in self.connectors and not word.isdigit():
                break
            previous = token.end() if forward else token.start()
        return found

    def relation(self, region: str | None, areas: frozenset[str]) -> str:
        """How a structure of ``region`` stands to the exam's ``areas``."""
        if not region or not areas:
            return UNKNOWN
        if WHOLE_BODY in areas or region in areas:
            return EXPECTED
        if any(region in self.adjacent.get(area, ()) for area in areas):
            return ADJACENT
        return OUTSIDE
