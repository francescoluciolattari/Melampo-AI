from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _hash_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class ProspectivePrediction:
    case_id: str
    prediction_id: str
    created_at: float
    payload_hash: str
    diagnostic_result: dict[str, Any]
    locked: bool = True
    outcome: dict[str, Any] | None = None
    audit: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "prediction_id": self.prediction_id,
            "created_at": self.created_at,
            "payload_hash": self.payload_hash,
            "diagnostic_result": self.diagnostic_result,
            "locked": self.locked,
            "outcome": self.outcome,
            "audit": self.audit,
        }


@dataclass(slots=True)
class ProspectiveValidationRegistry:
    """Append-only registry for prospective validation.

    A prospective study must lock predictions before outcomes are known. This
    registry provides file-based primitives for research governance; production
    use should replace storage with an immutable database or compliant audit log.
    """

    predictions: dict[str, ProspectivePrediction] = field(default_factory=dict)

    def create_prediction(self, case_payload: dict[str, Any], diagnostic_result: dict[str, Any], protocol_id: str = "melampo_prospective_protocol") -> ProspectivePrediction:
        case_id = str(case_payload.get("case_id", diagnostic_result.get("case_id", "unknown_case")))
        payload_hash = _hash_payload(case_payload)
        timestamp = time.time()
        prediction_id = hashlib.sha256(f"{protocol_id}:{case_id}:{payload_hash}:{timestamp}".encode("utf-8")).hexdigest()[:24]
        prediction = ProspectivePrediction(
            case_id=case_id,
            prediction_id=prediction_id,
            created_at=timestamp,
            payload_hash=payload_hash,
            diagnostic_result=diagnostic_result,
            locked=True,
            audit={
                "protocol_id": protocol_id,
                "created_before_outcome": True,
                "clinical_warning": "Research validation registry; not a medical record system.",
            },
        )
        self.predictions[prediction_id] = prediction
        return prediction

    def attach_outcome(self, prediction_id: str, outcome: dict[str, Any]) -> dict[str, Any]:
        if prediction_id not in self.predictions:
            return {"status": "rejected", "reason": "unknown_prediction_id", "prediction_id": prediction_id}
        prediction = self.predictions[prediction_id]
        if prediction.outcome is not None:
            return {"status": "rejected", "reason": "outcome_already_attached", "prediction_id": prediction_id}
        prediction.outcome = {
            **outcome,
            "attached_at": time.time(),
            "outcome_after_prediction": True,
        }
        return {"status": "completed", "prediction": prediction.as_dict()}

    def evaluate(self) -> dict[str, Any]:
        completed = [prediction for prediction in self.predictions.values() if prediction.outcome is not None]
        rows = []
        for prediction in completed:
            result_label = str(prediction.diagnostic_result.get("result_label", "")).strip().lower()
            accepted_labels = [str(label).strip().lower() for label in prediction.outcome.get("accepted_labels", [])]
            correct = bool(result_label and result_label in accepted_labels)
            rows.append(
                {
                    "case_id": prediction.case_id,
                    "prediction_id": prediction.prediction_id,
                    "result_label": result_label,
                    "accepted_labels": accepted_labels,
                    "correct": correct,
                    "locked": prediction.locked,
                }
            )
        sample_count = len(rows)
        correct_count = sum(1 for row in rows if row["correct"])
        return {
            "status": "completed",
            "registered_predictions": len(self.predictions),
            "completed_outcomes": sample_count,
            "prospective_accuracy": correct_count / max(sample_count, 1),
            "rows": rows,
            "governance": {
                "requires_irb_or_ethics_review": True,
                "requires_pre_registered_protocol": True,
                "requires_prediction_lock_before_outcome": True,
                "clinical_warning": "Prospective accuracy is research evidence only until formal clinical validation and regulatory review.",
            },
        }

    def save_jsonl(self, path: str | Path) -> None:
        lines = [json.dumps(prediction.as_dict(), sort_keys=True, default=str) for prediction in self.predictions.values()]
        Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "ProspectiveValidationRegistry":
        registry = cls()
        path = Path(path)
        if not path.exists():
            return registry
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            prediction = ProspectivePrediction(
                case_id=item["case_id"],
                prediction_id=item["prediction_id"],
                created_at=float(item["created_at"]),
                payload_hash=item["payload_hash"],
                diagnostic_result=dict(item["diagnostic_result"]),
                locked=bool(item.get("locked", True)),
                outcome=item.get("outcome"),
                audit=dict(item.get("audit", {})),
            )
            registry.predictions[prediction.prediction_id] = prediction
        return registry
