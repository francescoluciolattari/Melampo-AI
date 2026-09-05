"""Read a model's clinical reasoning the way a colleague's would be read.

A diagnostic model reasons in clinical language, and that language is not
noise to be forced into a schema. "I think heart failure, though the fever
bothers me — I would rule out pneumonia first" carries a leading hypothesis,
a discordant finding flagged as not fitting, a candidate to exclude, and an
implicit request for a test. A rigid output format loses most of that; a
reader of clinical discourse keeps it.

So the model is not fine-tuned to emit a script. It is read as a clinician
presenting a case, with the same semantic machinery that reads a radiologist's
report: concepts resolved against the ontology, and each one placed in the
mental space the discourse assigns it — asserted, hedged, proposed for
exclusion, denied. The illness script is the **output of that reading**, and
the verifier then says what each element rests on.

This is also the position clinical decision-making actually takes. The
decision is not made by the presenter; it is made in the room, after the
presentation has been heard, questioned and checked. The model presents, the
reader structures, the graph checks, the calibrator decides.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..memory.assertion import (
    CERTAINTY_FACTUAL,
    CERTAINTY_HYPOTHETICAL,
    CERTAINTY_POSSIBLE,
    ENGLISH_CUES,
    POLARITY_NEGATED,
    AssertionDetector,
    CueSet,
)
from ..memory.concept_resolution import (
    ConceptResolver,
    ResolvedConcept,
    attach_modifiers,
)
from .illness_script import (
    ORIGIN_MODEL,
    DifferentialEntry,
    IllnessScript,
    ScriptElement,
)

# Connectives that mark what follows as a proposed explanation rather than an
# observation, and how strongly the speaker commits to it.
DIAGNOSTIC_CONNECTIVES: dict[str, int] = {
    "most likely": 1,
    "most consistent with": 1,
    "diagnosis of": 1,
    "diagnostic of": 1,
    "likely": 2,
    "probable": 2,
    "consistent with": 2,
    "compatible with": 2,
    "suggestive of": 2,
    "suggests": 2,
    "in keeping with": 2,
    "differential includes": 3,
    "consider": 3,
    "possible": 3,
    "cannot exclude": 3,
    "rule out": 4,
    "to exclude": 4,
    "less likely": 4,
    "unlikely": 5,
}

ITALIAN_CONNECTIVES: dict[str, int] = {
    "diagnosi di": 1,
    "piu probabile": 1,
    "verosimile": 2,
    "compatibile con": 2,
    "coerente con": 2,
    "suggestivo di": 2,
    "probabile": 2,
    "da considerare": 3,
    "possibile": 3,
    "non escludibile": 3,
    "da escludere": 4,
    "meno probabile": 4,
    "improbabile": 5,
}

RANK_BY_CERTAINTY = {CERTAINTY_FACTUAL: 2, CERTAINTY_POSSIBLE: 3, CERTAINTY_HYPOTHETICAL: 4}


@dataclass
class DiscourseReading:
    """What the reader extracted, before it becomes a script."""

    findings: list[ResolvedConcept] = field(default_factory=list)
    candidates: list[tuple[ResolvedConcept, int, str]] = field(default_factory=list)
    denied: list[ResolvedConcept] = field(default_factory=list)
    discordant: list[ResolvedConcept] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "findings": [item.label for item in self.findings],
            "candidates": [
                {"condition": item.label, "rank": rank, "commitment": commitment}
                for item, rank, commitment in self.candidates
            ],
            "denied": [item.label for item in self.denied],
            "discordant": [item.label for item in self.discordant],
        }


@dataclass
class ClinicalDiscourseReader:
    """Turn free clinical reasoning into an illness script."""

    resolver: ConceptResolver
    cues: CueSet = ENGLISH_CUES
    connectives: dict[str, int] = field(default_factory=lambda: dict(DIAGNOSTIC_CONNECTIVES))
    window: int = 60

    def read(self, text: str) -> DiscourseReading:
        reading = DiscourseReading()
        if not text or not text.strip():
            return reading
        detector = AssertionDetector(cues=self.cues)
        lowered = text.lower()
        extraction = attach_modifiers(self.resolver.resolve_text(text))
        ordered = sorted(extraction.findings, key=lambda item: item.concept.char_start or 0)

        previous_end = 0
        for finding in ordered:
            concept = finding.concept
            start, end = concept.char_start or 0, concept.char_end or 0
            status = detector.detect(text, start, end)
            # A connective binds the concept immediately after it. The window is
            # clipped at the previous concept so "consistent with X, though the Y"
            # does not carry X's connective onto Y.
            connective, rank = self._preceding_connective(lowered, start, floor=previous_end)
            previous_end = end

            if status.polarity == POLARITY_NEGATED:
                reading.denied.append(concept)
                continue
            if connective is not None:
                reading.candidates.append((concept, rank, connective))
                continue
            if status.certainty in {CERTAINTY_POSSIBLE, CERTAINTY_HYPOTHETICAL}:
                reading.candidates.append((concept, RANK_BY_CERTAINTY[status.certainty], status.certainty))
                continue
            if self._flagged_discordant(lowered, start):
                reading.discordant.append(concept)
            reading.findings.append(concept)
        return reading

    def to_script(self, text: str) -> IllnessScript:
        """Read the discourse and render it as a script for the verifier."""
        reading = self.read(text)
        ordered = sorted(reading.candidates, key=lambda item: item[1])
        differential = [
            DifferentialEntry(
                condition=concept.label,
                term_id=concept.term_id,
                rank=position,
                origin=ORIGIN_MODEL,
                discriminating_features=(),
            )
            for position, (concept, _, _) in enumerate(ordered, start=1)
        ]
        return IllnessScript(
            consequences=[
                ScriptElement(
                    label=item.label,
                    term_id=item.term_id,
                    note="discordant" if item in reading.discordant else None,
                )
                for item in reading.findings
            ],
            differential=differential,
        )

    def _preceding_connective(
        self, lowered: str, start: int, *, floor: int = 0
    ) -> tuple[str | None, int]:
        window = lowered[max(0, start - self.window, floor) : start]
        best: tuple[str | None, int, int] = (None, 99, -1)
        for phrase, rank in self.connectives.items():
            position = window.rfind(phrase)
            if position >= 0 and position > best[2]:
                best = (phrase, rank, position)
        return best[0], best[1]

    def _flagged_discordant(self, lowered: str, start: int) -> bool:
        window = lowered[max(0, start - self.window) : start + self.window]
        return any(
            marker in window
            for marker in ("does not fit", "bothers me", "discordant", "atypical", "unexpected", "non torna", "atipico")
        )


def read_and_verify(
    text: str,
    reader: ClinicalDiscourseReader,
    verifier: Any,
    case_findings: Sequence[str],
) -> dict[str, Any]:
    """The full path: presentation → reading → script → what each part rests on."""
    reading = reader.read(text)
    script = reader.to_script(text)
    verification = verifier.verify(script, case_findings)
    return {
        "reading": reading.as_dict(),
        "script": script.as_dict(),
        "verification": verification.as_dict(),
    }
