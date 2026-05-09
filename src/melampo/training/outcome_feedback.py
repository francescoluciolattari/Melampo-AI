from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..memory.vector_memory import InMemoryVectorStore
from .dream_candidate_store import DreamCandidateStore


def _stable_feedback_id(case_id: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(f"feedback:{case_id}:{canonical}:{time.time()}".encode("utf-8")).hexdigest()
    return f"feedback:{case_id}:{digest[:16]}"


@dataclass(slots=True)
class OutcomeFeedbackRecord:
    feedback_id: str
    case_id: str
    created_at: float
    result_label: str
    accepted_labels: list[str] = field(default_factory=list)
    correct: bool = False
    confidence: float = 0.0
    review_status: str = "needs_review"
    notes: str = ""
    source: str = "outcome_feedback"

    def as_dict(self) -> dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "case_id": self.case_id,
            "created_at": self.created_at,
            "result_label": self.result_label,
            "accepted_labels": self.accepted_labels,
            "correct": self.correct,
            "confidence": self.confidence,
            "review_status": self.review_status,
            "notes": self.notes,
            "source": self.source,
        }


@dataclass(slots=True)
class OutcomeFeedbackIngestor:
    """Attach reviewed outcomes to dream candidates and optional memory traces."""

    default_source: str = "reviewed_outcome"

    def build_feedback(self, diagnostic_result: dict[str, Any], outcome: dict[str, Any]) -> OutcomeFeedbackRecord:
        diagnostic_result = diagnostic_result or {}
        outcome = outcome or {}
        case_id = str(diagnostic_result.get("case_id", outcome.get("case_id", "unknown_case")))
        result_label = str(diagnostic_result.get("result_label", diagnostic_result.get("top_hypothesis", {}).get("label", ""))).strip().lower()
        accepted_labels = [str(label).strip().lower() for label in outcome.get("accepted_labels", [])]
        correct = bool(result_label and result_label in accepted_labels)
        confidence = float(diagnostic_result.get("top_hypothesis", {}).get("score", outcome.get("confidence", 0.0)) or 0.0)
        review_status = "accepted" if correct else "needs_review"
        payload = {"diagnostic_result": diagnostic_result, "outcome": outcome}
        return OutcomeFeedbackRecord(
            feedback_id=_stable_feedback_id(case_id=case_id, payload=payload),
            case_id=case_id,
            created_at=time.time(),
            result_label=result_label,
            accepted_labels=accepted_labels,
            correct=correct,
            confidence=max(0.0, min(1.0, confidence)),
            review_status=str(outcome.get("review_status", review_status)),
            notes=str(outcome.get("notes", "")),
            source=self.default_source,
        )

    def attach_to_candidate(
        self,
        store: DreamCandidateStore,
        candidate_id: str,
        diagnostic_result: dict[str, Any],
        outcome: dict[str, Any],
    ) -> dict[str, Any]:
        feedback = self.build_feedback(diagnostic_result=diagnostic_result, outcome=outcome)
        return store.attach_outcome_feedback(candidate_id=candidate_id, feedback=feedback.as_dict())

    def consolidate_to_memory(
        self,
        vector_store: InMemoryVectorStore,
        feedback: OutcomeFeedbackRecord,
        learning_status: str = "candidate",
    ) -> dict[str, Any]:
        text = (
            f"Reviewed outcome feedback for {feedback.case_id}: "
            f"result={feedback.result_label}; accepted={feedback.accepted_labels}; correct={feedback.correct}."
        )
        return vector_store.upsert_text(
            record_id=feedback.feedback_id,
            text=text,
            metadata={
                "record_id": feedback.feedback_id,
                "case_id": feedback.case_id,
                "focus": "outcome_feedback",
                "memory_role": "reviewed_outcome_feedback",
                "correct": feedback.correct,
                "accepted_labels": feedback.accepted_labels,
                "review_status": feedback.review_status,
                "provenance_quality": 1.0,
            },
            source=feedback.source,
            learning_status=learning_status,
        )
