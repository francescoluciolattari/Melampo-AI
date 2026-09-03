"""The single point at which patient findings are assembled.

Two guards existed before this module — one rejecting synthetic hypotheses, one
rejecting family-history screening items — and neither was invoked anywhere on
the production path. A guard that is never called does not guard; the separation
was a convention, and conventions hold until someone forgets.

This module makes it a constraint. Everything entering the findings set passes
through `assemble`, which admits only what is a finding of *this* patient, right
now, asserted rather than negated or hypothesised, and rejects everything else
with a reason attached.

The rejections are not failures to be suppressed. Each routes somewhere:

| Rejected | Routes to |
|---|---|
| Negated | Documented exclusion — the graph can hold it as such |
| Hypothetical, "rule out" | Open question — discriminating test selection |
| Attributed to a relative | Family history channel |
| Historical and resolved | Context, not current state |
| Synthetic hypothesis | Differential, as exclusion hypothesis only |

The distinction that governs all of them is direction rather than magnitude. A
finding is an entry point: the graph is walked *from* it. Anything admitted here
by mistake generates paths toward the consequences of something the patient does
not have, and those paths are structurally correct on a false premise — which is
the hardest error to notice, because nothing downstream looks wrong.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..memory.assertion import (
    CERTAINTY_FACTUAL,
    EXPERIENCER_PATIENT,
    POLARITY_AFFIRMED,
    TEMPORALITY_CURRENT,
    AssertionStatus,
)
from ..training.hypothesis_channel import HYPOTHESIS_ROLE
from .family_history import ROLE_SCREENING_HYPOTHESIS

REJECT_NEGATED = "negated"
REJECT_HYPOTHETICAL = "hypothetical"
REJECT_OTHER_EXPERIENCER = "other_experiencer"
REJECT_HISTORICAL = "historical"
REJECT_SYNTHETIC = "synthetic_hypothesis"
REJECT_SCREENING = "family_history_screening"

ROUTE = {
    REJECT_NEGATED: "documented_exclusion",
    REJECT_HYPOTHETICAL: "open_question",
    REJECT_OTHER_EXPERIENCER: "family_history_channel",
    REJECT_HISTORICAL: "clinical_context",
    REJECT_SYNTHETIC: "differential_as_exclusion_hypothesis",
    REJECT_SCREENING: "screening_considerations",
}


@dataclass(frozen=True)
class AdmittedFinding:
    """A finding of this patient, currently present, as asserted."""

    label: str
    term_id: str | None = None
    assertion: AssertionStatus | None = None
    modifiers: tuple[str, ...] = ()
    char_start: int | None = None
    char_end: int | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label,
            "term_id": self.term_id,
            "modifiers": list(self.modifiers),
            "char_start": self.char_start,
            "char_end": self.char_end,
        }
        if self.assertion is not None:
            payload["assertion"] = self.assertion.as_dict()
        return payload


@dataclass(frozen=True)
class RejectedItem:
    """Something that is not a patient finding, and where it belongs instead."""

    label: str
    reason: str

    @property
    def route(self) -> str:
        return ROUTE.get(self.reason, "discarded")

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "reason": self.reason, "route": self.route}


@dataclass
class FindingsSet:
    admitted: list[AdmittedFinding] = field(default_factory=list)
    rejected: list[RejectedItem] = field(default_factory=list)

    @property
    def concepts(self) -> list[str]:
        """Graph entry points. Only these are traversed from."""
        return [item.label for item in self.admitted]

    def rejected_for(self, reason: str) -> list[RejectedItem]:
        return [item for item in self.rejected if item.reason == reason]

    def as_dict(self) -> dict[str, Any]:
        return {
            "admitted": [item.as_dict() for item in self.admitted],
            "rejected": [item.as_dict() for item in self.rejected],
            "admitted_count": len(self.admitted),
            "rejected_count": len(self.rejected),
        }


def assemble(candidates: Sequence[dict[str, Any]]) -> FindingsSet:
    """Admit only what is a current, asserted finding of this patient.

    Each candidate is a dict carrying at least ``label``, optionally ``term_id``,
    ``assertion``, ``modifiers`` and offsets. Items marked as synthetic
    hypotheses or screening considerations are rejected before any assertion
    check, because their role already disqualifies them regardless of how they
    are phrased.
    """
    result = FindingsSet()

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        label = str(candidate.get("label", "")).strip()
        if not label:
            continue

        role_rejection = _role_rejection(candidate)
        if role_rejection is not None:
            result.rejected.append(RejectedItem(label, role_rejection))
            continue

        assertion = candidate.get("assertion")
        if isinstance(assertion, AssertionStatus):
            reason = _assertion_rejection(assertion)
            if reason is not None:
                result.rejected.append(RejectedItem(label, reason))
                continue

        result.admitted.append(
            AdmittedFinding(
                label=label,
                term_id=candidate.get("term_id"),
                assertion=assertion if isinstance(assertion, AssertionStatus) else None,
                modifiers=tuple(candidate.get("modifiers", ()) or ()),
                char_start=candidate.get("char_start"),
                char_end=candidate.get("char_end"),
            )
        )

    return result


def assert_findings_only(candidates: Sequence[dict[str, Any]]) -> None:
    """Raise if anything that is not a patient finding reaches the findings path.

    Called at the boundary rather than relied upon by convention. The two
    previous guards were defined and never invoked, so the isolation they
    described was not enforced anywhere.
    """
    rejected = assemble(candidates).rejected
    if rejected:
        detail = "; ".join(f"{item.label} ({item.reason} -> {item.route})" for item in rejected)
        raise ValueError(f"non-finding item reached the findings path: {detail}")


def _role_rejection(candidate: dict[str, Any]) -> str | None:
    role = candidate.get("role")
    if role == HYPOTHESIS_ROLE or candidate.get("synthetic_candidate_not_clinical_truth") is True:
        return REJECT_SYNTHETIC
    if role == ROLE_SCREENING_HYPOTHESIS or candidate.get("belongs_in_differential") is False:
        return REJECT_SCREENING
    if candidate.get("usable_as_evidence") is False:
        return REJECT_SYNTHETIC
    return None


def _assertion_rejection(assertion: AssertionStatus) -> str | None:
    if assertion.experiencer != EXPERIENCER_PATIENT:
        return REJECT_OTHER_EXPERIENCER
    if assertion.certainty != CERTAINTY_FACTUAL:
        return REJECT_HYPOTHETICAL
    if assertion.polarity != POLARITY_AFFIRMED:
        return REJECT_NEGATED
    if assertion.temporality != TEMPORALITY_CURRENT:
        return REJECT_HISTORICAL
    return None
