"""Admit a case into the learning cycle only when its confirmation is independent.

A confirmed case that feeds recalibration or retraining is legitimate learning.
A case "confirmed" because the system proposed a diagnosis and nobody
contradicted it is not: the system would learn from its own proposals, growing
more certain of what it already believed with every cycle. The mechanism is
documented as automation bias in clinical decision support, and it is silent —
each individual step looks correct, and the drift only becomes visible in
aggregate, long after it started.

The safeguard is not "never learn from cases". It is that the **confirmation
must have a source other than the system's output**, and that the source is
recorded rather than assumed. Histology, clinical outcome, an independent
review: each is evidence produced by something that did not participate in the
reasoning. Acceptance of a suggestion is not.

The distinction cannot be inferred after the fact. Whether a clinician
independently reached the same diagnosis or simply accepted the one on screen
looks identical in the record unless it was captured at the time — so the source
is required at registration, not reconstructed later.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

SOURCE_HISTOPATHOLOGY = "histopathology"
SOURCE_CLINICAL_OUTCOME = "clinical_outcome"
SOURCE_INDEPENDENT_REVIEW = "independent_review"
SOURCE_REFERENCE_STANDARD = "reference_standard"
SOURCE_SYSTEM_ACCEPTED = "system_suggestion_accepted"
SOURCE_UNSPECIFIED = "unspecified"

# Sources produced by something that did not take part in the system's
# reasoning. Everything else is excluded, including an accepted suggestion.
INDEPENDENT_SOURCES = frozenset(
    {
        SOURCE_HISTOPATHOLOGY,
        SOURCE_CLINICAL_OUTCOME,
        SOURCE_INDEPENDENT_REVIEW,
        SOURCE_REFERENCE_STANDARD,
    }
)

REJECT_NOT_INDEPENDENT = "confirmation_not_independent_of_system_output"
REJECT_UNSPECIFIED_SOURCE = "confirmation_source_not_recorded"
REJECT_REVIEWER_SAW_SUGGESTION = "reviewer_saw_the_suggestion_before_confirming"
REJECT_DUPLICATE = "case_already_registered"


@dataclass(frozen=True)
class Confirmation:
    """A confirmed diagnosis, with how the confirmation was obtained."""

    case_id: str
    diagnosis: str
    source: str = SOURCE_UNSPECIFIED
    reviewer_blinded_to_suggestion: bool | None = None
    confirmed_on: date | None = None
    term_id: str | None = None
    note: str | None = None

    @property
    def is_independent(self) -> bool:
        """Whether the confirmation came from outside the system's reasoning.

        An independent review counts only when the reviewer did not see the
        suggestion first. Reading it and agreeing is the failure mode, not a
        weaker form of confirmation, so an unrecorded blinding status is treated
        as unblinded rather than assumed favourable.
        """
        if self.source not in INDEPENDENT_SOURCES:
            return False
        if self.source == SOURCE_INDEPENDENT_REVIEW:
            return self.reviewer_blinded_to_suggestion is True
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "diagnosis": self.diagnosis,
            "term_id": self.term_id,
            "source": self.source,
            "reviewer_blinded_to_suggestion": self.reviewer_blinded_to_suggestion,
            "confirmed_on": self.confirmed_on.isoformat() if self.confirmed_on else None,
            "independent": self.is_independent,
            "note": self.note,
        }


@dataclass(frozen=True)
class RejectedConfirmation:
    confirmation: Confirmation
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"case_id": self.confirmation.case_id, "reason": self.reason}


@dataclass
class ConfirmationRegistry:
    """The gate between confirmed cases and the learning cycle."""

    admitted: list[Confirmation] = field(default_factory=list)
    rejected: list[RejectedConfirmation] = field(default_factory=list)

    def register(self, confirmation: Confirmation) -> bool:
        """Admit a confirmation if independent. Returns whether it was admitted."""
        reason = self._rejection_reason(confirmation)
        if reason is not None:
            self.rejected.append(RejectedConfirmation(confirmation, reason))
            return False
        self.admitted.append(confirmation)
        return True

    def register_many(self, confirmations: Sequence[Confirmation]) -> dict[str, int]:
        for confirmation in confirmations:
            self.register(confirmation)
        return {"admitted": len(self.admitted), "rejected": len(self.rejected)}

    def learning_set(self) -> list[Confirmation]:
        """The only cases that may feed recalibration or retraining."""
        return list(self.admitted)

    def independence_rate(self) -> float:
        """Fraction of confirmations that were independent.

        Worth watching over time rather than only at registration: a falling
        rate means the system is increasingly being confirmed by agreement with
        itself, which is the drift this module exists to make visible.
        """
        total = len(self.admitted) + len(self.rejected)
        return len(self.admitted) / total if total else 0.0

    def report(self) -> dict[str, Any]:
        by_source: dict[str, int] = {}
        for item in self.admitted:
            by_source[item.source] = by_source.get(item.source, 0) + 1
        by_reason: dict[str, int] = {}
        for item in self.rejected:
            by_reason[item.reason] = by_reason.get(item.reason, 0) + 1
        return {
            "admitted": len(self.admitted),
            "rejected": len(self.rejected),
            "independence_rate": round(self.independence_rate(), 4),
            "admitted_by_source": dict(sorted(by_source.items())),
            "rejected_by_reason": dict(sorted(by_reason.items())),
        }

    def _rejection_reason(self, confirmation: Confirmation) -> str | None:
        if any(item.case_id == confirmation.case_id for item in self.admitted):
            return REJECT_DUPLICATE
        if confirmation.source == SOURCE_UNSPECIFIED:
            return REJECT_UNSPECIFIED_SOURCE
        if confirmation.source == SOURCE_SYSTEM_ACCEPTED:
            return REJECT_NOT_INDEPENDENT
        if confirmation.source not in INDEPENDENT_SOURCES:
            return REJECT_NOT_INDEPENDENT
        if confirmation.source == SOURCE_INDEPENDENT_REVIEW and confirmation.reviewer_blinded_to_suggestion is not True:
            return REJECT_REVIEWER_SAW_SUGGESTION
        return None


def assert_independent(confirmations: Sequence[Confirmation]) -> None:
    """Raise if a non-independent confirmation reaches the learning cycle."""
    offenders = [item for item in confirmations if not item.is_independent]
    if offenders:
        detail = "; ".join(f"{item.case_id} ({item.source})" for item in offenders)
        raise ValueError(f"non-independent confirmation in the learning set: {detail}")
