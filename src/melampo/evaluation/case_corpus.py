"""Load evaluation cases from the formats published case corpora use.

No corpus is committed here. The credible sources — MedCaseReasoning derived
from PubMed Central, PMC-Patients, the NEJM clinicopathological conferences —
are obtained under their own terms, and two of the three require either PMC
access or a licence. What is committed is the reader, so that a corpus plugs in
without changes to the harness.

Every loader enforces the same discipline: the **documented diagnosis is held
apart from the presentation**, and a record whose presentation still contains
its own diagnosis is rejected rather than silently evaluated. A case that leaks
its answer measures nothing, and it looks exactly like a case that does not.
"""

import json
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from .dream_capture_benchmark import EvaluationCase

REJECT_LEAKED_DIAGNOSIS = "presentation_contains_the_diagnosis"
REJECT_MISSING_FIELD = "missing_presentation_or_diagnosis"
REJECT_TOO_SHORT = "presentation_too_short_to_evaluate"

MIN_PRESENTATION_CHARACTERS = 120


@dataclass
class LoadReport:
    """What loaded, and what was rejected with the reason."""

    cases: list[EvaluationCase] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        reasons: dict[str, int] = {}
        for _, reason in self.rejected:
            reasons[reason] = reasons.get(reason, 0) + 1
        return {
            "loaded": len(self.cases),
            "rejected": len(self.rejected),
            "rejected_by_reason": dict(sorted(reasons.items())),
        }


def load_records(
    records: Iterable[dict[str, Any]],
    *,
    presentation_key: str = "presentation",
    diagnosis_key: str = "diagnosis",
    id_key: str = "case_id",
    candidates_key: str = "candidate_conditions",
    source: str | None = None,
) -> LoadReport:
    """Load from any record shape by naming its fields.

    Field names differ between corpora; the discipline does not.
    """
    report = LoadReport()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        case_id = str(record.get(id_key) or f"case_{index}")
        presentation = str(record.get(presentation_key) or "").strip()
        diagnosis = str(record.get(diagnosis_key) or "").strip()

        reason = _rejection(presentation, diagnosis)
        if reason is not None:
            report.rejected.append((case_id, reason))
            continue

        report.cases.append(
            EvaluationCase(
                case_id=case_id,
                presentation=presentation,
                documented_diagnosis=diagnosis,
                candidate_conditions=tuple(
                    str(item) for item in (record.get(candidates_key) or ()) if str(item).strip()
                ),
                source=source or record.get("source"),
            )
        )
    return report


def load_jsonl(lines: Iterable[str], **kwargs: Any) -> LoadReport:
    """Read newline-delimited JSON, the format most released corpora ship."""
    return load_records(_parse_jsonl(lines), **kwargs)


def load_medcasereasoning(lines: Iterable[str]) -> LoadReport:
    """MedCaseReasoning field names, as produced by its extraction pipeline."""
    return load_jsonl(
        lines,
        presentation_key="case_prompt",
        diagnosis_key="final_diagnosis",
        id_key="pmcid",
        source="medcasereasoning",
    )


def load_pmc_patients(lines: Iterable[str]) -> LoadReport:
    """PMC-Patients field names."""
    return load_jsonl(
        lines,
        presentation_key="patient",
        diagnosis_key="diagnosis",
        id_key="patient_uid",
        source="pmc_patients",
    )


def split_presentation(text: str, diagnosis_headings: Sequence[str] = ()) -> tuple[str, str]:
    """Split a case report into presentation and the section that reveals the outcome.

    Case reports state the diagnosis in a later section. Splitting on that
    heading is what makes the earlier text usable as a presentation; without the
    split the evaluation reads the answer and reports success.
    """
    headings = tuple(diagnosis_headings) or (
        "final diagnosis",
        "diagnosis:",
        "discussion",
        "outcome and follow-up",
        "conclusion",
    )
    lowered = text.lower()
    cut = len(text)
    for heading in headings:
        position = lowered.find(heading)
        if 0 <= position < cut:
            cut = position
    return text[:cut].strip(), text[cut:].strip()


def _rejection(presentation: str, diagnosis: str) -> str | None:
    if not presentation or not diagnosis:
        return REJECT_MISSING_FIELD
    if len(presentation) < MIN_PRESENTATION_CHARACTERS:
        return REJECT_TOO_SHORT
    if _mentions(presentation, diagnosis):
        return REJECT_LEAKED_DIAGNOSIS
    return None


def _mentions(presentation: str, diagnosis: str) -> bool:
    """Whether the presentation already states the diagnosis.

    Matched on the normalised phrase rather than on shared words: a diagnosis
    and a presentation naturally share vocabulary, and rejecting on overlap
    would discard usable cases.
    """
    needle = _normalise(diagnosis)
    if len(needle) < 4:
        return False
    return needle in _normalise(presentation)


def _parse_jsonl(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            yield record


def _normalise(value: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", str(value).lower()).split())
