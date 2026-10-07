"""The frame a sentence is written in: findings of an image, laboratory results, vital signs.

A structure named in a sentence that reports laboratory values or vital signs ("heart rate 112",
"thyroid, parathyroid and vitamin D assay were normal") is the modifier of a measurement, not a
finding about the structure. A reader knows this before reading the name, from the kind of text:
the same reason clinical documents are divided into sections (LOINC/HL7 section codes: laboratory
data, vital signs, physical findings) and radiology reports into history, technique, comparison,
findings and impression (ACR, RSNA RadReport), where the section changes what a concept means
(SecTag). The cues are words and unit patterns of the frame, in ``data/linking/exam_frames.json``;
none of them is an anatomical name, so the frame holds for every structure.

It decides nothing alone: the linker uses it as one stream among the others, and records it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from melampo.memory.word_senses import _fold, words_of

DEFAULT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "linking" / "exam_frames.json"
)

IMAGING, LABORATORY, VITAL_SIGNS, UNKNOWN = (
    "imaging",
    "laboratory",
    "vital_signs",
    "unknown",
)
# Frames in which a structure's name is a modifier of a measurement, not a finding.
MEASUREMENT_FRAMES = frozenset({LABORATORY, VITAL_SIGNS})
_THRESHOLD = 4
_STRONG, _WEAK, _PATTERN = 2, 1, 2


@dataclass(frozen=True)
class FrameVerdict:
    frame: str
    scores: dict[str, int] = field(default_factory=dict)
    cues: dict[str, list[str]] = field(default_factory=dict)

    @property
    def measurement(self) -> bool:
        return self.frame in MEASUREMENT_FRAMES


@dataclass
class ExamFrames:
    strong: dict[str, frozenset[str]] = field(default_factory=dict)
    weak: dict[str, frozenset[str]] = field(default_factory=dict)
    patterns: dict[str, tuple[re.Pattern[str], ...]] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "ExamFrames":
        return cls()

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ExamFrames":
        frames = data["frames"]
        return cls(
            strong={
                k: frozenset(_fold(w) for w in v.get("strong", ()))
                for k, v in frames.items()
            },
            weak={
                k: frozenset(_fold(w) for w in v.get("weak", ()))
                for k, v in frames.items()
            },
            patterns={
                k: tuple(re.compile(p, re.IGNORECASE) for p in v.get("patterns", ()))
                for k, v in frames.items()
            },
        )

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "ExamFrames":
        return cls.from_json(json.loads(path.read_text("utf-8")))

    def of(self, sentence: str) -> FrameVerdict:
        """The frame of the sentence: a measurement frame, imaging, or unknown."""
        if not self.strong:
            return FrameVerdict(UNKNOWN)
        words = set(words_of(sentence))
        scores: dict[str, int] = {}
        cues: dict[str, list[str]] = {}
        for frame in self.strong:
            found = [w for w in sorted(words & self.strong[frame])]
            weak = [w for w in sorted(words & self.weak.get(frame, frozenset()))]
            shapes = [
                m.group(0)
                for p in self.patterns.get(frame, ())
                if (m := p.search(sentence))
            ]
            score = len(found) * _STRONG + len(weak) * _WEAK + len(shapes) * _PATTERN
            if score:
                scores[frame], cues[frame] = score, found + weak + shapes
        if scores.get(IMAGING, 0) >= _STRONG:
            return FrameVerdict(IMAGING, scores, cues)
        measurement = {f: s for f, s in scores.items() if f in MEASUREMENT_FRAMES}
        if measurement:
            best = max(measurement, key=lambda f: measurement[f])
            if measurement[best] >= _THRESHOLD and not scores.get(IMAGING):
                return FrameVerdict(best, scores, cues)
        return FrameVerdict(UNKNOWN, scores, cues)
