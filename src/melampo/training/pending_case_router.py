"""Routes an incoming payload against a case already sitting at needs_review.

Built from a design worked through directly, not assumed: a case held at
needs_review (NexusCandidateStore, via NexusScheduler -> PromotionPolicy,
which holds by default -- see recursive_engine_decision_record.md) can
receive one of two genuinely different follow-ups, and both should arrive
through the same payload shape pipeline.run() already accepts, not a
second ingestion procedure:

- New diagnostic data (more findings, a follow-up report) -- merged with
  what is already on file, the diagnostic process re-run on the combined
  picture, the case staying at needs_review with an updated record.
- A confirmed diagnosis -- routed toward training (see outcome_feedback.py,
  preference_pairs.py), whether it confirms or contradicts what the
  system proposed, then the raw record deleted once its training use is
  served (see the same decision record entry on data retention).

This module decides which of the three things is happening -- a brand
new case, more data for a pending one, or a confirmation for a pending
one -- and, for the merge case, builds the combined report text. It does
not itself run the diagnostic pipeline or the training/promotion chain;
those stay where they already are, called by whoever invokes this first.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .nexus_candidate_store import NexusCandidateRecord, NexusCandidateStore

PENDING_STATUSES = ("candidate", "needs_review")

# A case with no confirmation for this long is closed unconfirmed, not
# left open indefinitely -- a decision made directly, not defaulted into
# (see recursive_engine_decision_record.md).
DEFAULT_RETENTION_SECONDS = 365 * 24 * 60 * 60


@dataclass(frozen=True)
class RoutingDecision:
    """What to do with this payload, and why -- data the caller acts on, not a decision this module enforces itself."""

    action: str  # "new_case" | "merge_and_rerun" | "confirm_and_train"
    case_id: str
    existing_record: NexusCandidateRecord | None = None
    merged_report_text: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "case_id": self.case_id,
            "existing_candidate_id": self.existing_record.candidate_id if self.existing_record else None,
            "merged_report_text": self.merged_report_text,
        }


def merge_report_text(previous_text: str, new_text: str, *, new_date: str | None = None, previous_date: str | None = None) -> str:
    """The new report first, referencing the one before it, which follows below -- not a replacement.

    Each call wraps the entire prior text under a "previous report"
    section, so a chain of updates reads as a stack, most recent on top,
    with nothing discarded -- confirmed directly: partial information
    should not silently replace what was already known about the case.
    """
    new_date = new_date or datetime.now(UTC).date().isoformat()
    if not previous_text.strip():
        return f"[Aggiornamento del {new_date}] {new_text}"
    reference = f" (fa riferimento al referto del {previous_date})" if previous_date else " (fa riferimento al referto precedente)"
    return (
        f"[Aggiornamento del {new_date}]{reference}\n{new_text}\n\n"
        f"--- Referto precedente ---\n{previous_text}"
    )


def route_payload(payload: dict[str, Any], store: NexusCandidateStore) -> RoutingDecision:
    """Whether this payload is a new case, more data for a pending one, or a confirmation for one.

    A payload is treated as referring to a pending case only when its
    case_id matches an existing needs_review/candidate record -- a case_id
    Melampo itself assigned when the case was first analysed, never a
    name, date of birth or clinical narrative match. Confirmation is
    signalled by `confirmed_diagnosis` being present in the payload; its
    absence with a matching case_id means more diagnostic data arrived
    instead.
    """
    case_id = str(payload.get("case_id") or "")
    existing = store.find_by_case_id(case_id, statuses=PENDING_STATUSES) if case_id else None

    if existing is None:
        return RoutingDecision(action="new_case", case_id=case_id)

    if payload.get("confirmed_diagnosis"):
        return RoutingDecision(action="confirm_and_train", case_id=case_id, existing_record=existing)

    previous_report = str(existing.payload.get("case_context", {}).get("report_text", ""))
    previous_date = existing.payload.get("report_date")
    merged = merge_report_text(previous_report, str(payload.get("report_text", "")), previous_date=previous_date)
    return RoutingDecision(action="merge_and_rerun", case_id=case_id, existing_record=existing, merged_report_text=merged)


def sweep_expired_pending_cases(
    store: NexusCandidateStore, *, retention_seconds: float = DEFAULT_RETENTION_SECONDS, now: float | None = None
) -> list[str]:
    """Delete every pending case older than the retention window with no confirmation -- returns the deleted candidate_ids.

    Only pending (candidate/needs_review) records are touched -- a case
    already promoted or rejected has already had its raw data handled by
    the confirm-and-train path and is not this function's concern.
    """
    import time as _time

    now = _time.time() if now is None else now
    expired = [
        record.candidate_id
        for record in store.list_by_status(statuses=PENDING_STATUSES)
        if (now - record.created_at) >= retention_seconds
    ]
    for candidate_id in expired:
        store.delete(candidate_id)
    return expired
