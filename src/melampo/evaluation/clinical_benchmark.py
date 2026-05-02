from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


PredictionFn = Callable[[dict[str, Any]], dict[str, Any]]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_label(label: Any) -> str:
    return str(label or "").strip().lower()


def _extract_prediction_label(result: dict[str, Any]) -> str:
    diagnostic_result = result.get("diagnostic_result", result)
    if not isinstance(diagnostic_result, dict):
        return ""
    if diagnostic_result.get("result_label") and diagnostic_result.get("result_label") != "abstain_or_escalate":
        return _normalize_label(diagnostic_result.get("result_label"))
    top = diagnostic_result.get("top_hypothesis", {})
    if isinstance(top, dict):
        return _normalize_label(top.get("label"))
    return ""


def _extract_confidence(result: dict[str, Any]) -> float:
    diagnostic_result = result.get("diagnostic_result", result)
    if isinstance(diagnostic_result, dict):
        top = diagnostic_result.get("top_hypothesis", {})
        if isinstance(top, dict) and top.get("score") is not None:
            try:
                return max(0.0, min(1.0, float(top.get("score"))))
            except (TypeError, ValueError):
                pass
        metrics = diagnostic_result.get("melampo_metrics", {})
        if isinstance(metrics, dict) and metrics.get("pi_score") is not None:
            try:
                return max(0.0, min(1.0, float(metrics.get("pi_score"))))
            except (TypeError, ValueError):
                pass
    return 0.0


@dataclass(slots=True)
class ClinicalBenchmarkRecord:
    case_id: str
    payload: dict[str, Any]
    gold_labels: list[str]
    slices: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_jsonl(cls, item: dict[str, Any]) -> "ClinicalBenchmarkRecord":
        payload = dict(item.get("payload", {}))
        case_id = str(item.get("case_id", payload.get("case_id", "unknown_case")))
        labels = [_normalize_label(label) for label in _as_list(item.get("gold_labels", item.get("gold_label")))]
        return cls(
            case_id=case_id,
            payload=payload,
            gold_labels=[label for label in labels if label],
            slices=dict(item.get("slices", {})),
            provenance=dict(item.get("provenance", {})),
        )


def load_benchmark_jsonl(path: str | Path) -> list[ClinicalBenchmarkRecord]:
    records: list[ClinicalBenchmarkRecord] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"Benchmark JSONL line {line_number} must contain an object.")
        records.append(ClinicalBenchmarkRecord.from_jsonl(item))
    return records


@dataclass(slots=True)
class ClinicalBenchmarkReport:
    benchmark_name: str
    sample_count: int
    answered_count: int
    abstained_count: int
    top1_accuracy: float
    coverage: float
    selective_accuracy: float
    records: list[dict[str, Any]] = field(default_factory=list)
    slices: dict[str, dict[str, Any]] = field(default_factory=dict)
    governance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "sample_count": self.sample_count,
            "answered_count": self.answered_count,
            "abstained_count": self.abstained_count,
            "top1_accuracy": round(self.top1_accuracy, 6),
            "coverage": round(self.coverage, 6),
            "selective_accuracy": round(self.selective_accuracy, 6),
            "records": self.records,
            "slices": self.slices,
            "governance": self.governance,
        }


@dataclass(slots=True)
class ClinicalBenchmarkRunner:
    """Dataset-agnostic clinical benchmark runner.

    Input records must include payloads and gold labels. The runner does not
    ship clinical data and does not assume a specific disease taxonomy.
    """

    benchmark_name: str = "melampo_clinical_benchmark"

    def run(self, records: Iterable[ClinicalBenchmarkRecord], prediction_fn: PredictionFn) -> ClinicalBenchmarkReport:
        rows: list[dict[str, Any]] = []
        slice_accumulator: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            result = prediction_fn(record.payload)
            diagnostic_result = result.get("diagnostic_result", result) if isinstance(result, dict) else {}
            policy = diagnostic_result.get("policy", {}) if isinstance(diagnostic_result, dict) else {}
            abstained = bool(policy.get("abstain", False)) or diagnostic_result.get("result_label") == "abstain_or_escalate"
            predicted_label = _extract_prediction_label(result if isinstance(result, dict) else {})
            correct = bool(predicted_label and predicted_label in record.gold_labels and not abstained)
            row = {
                "case_id": record.case_id,
                "predicted_label": predicted_label,
                "gold_labels": record.gold_labels,
                "correct": correct,
                "abstained": abstained,
                "confidence": _extract_confidence(result if isinstance(result, dict) else {}),
                "slices": record.slices,
                "provenance": record.provenance,
            }
            rows.append(row)
            for key, value in record.slices.items():
                slice_accumulator.setdefault(f"{key}:{value}", []).append(row)

        sample_count = len(rows)
        answered = [row for row in rows if not row["abstained"]]
        correct_count = sum(1 for row in rows if row["correct"])
        answered_correct_count = sum(1 for row in answered if row["correct"])
        slice_reports: dict[str, dict[str, Any]] = {}
        for slice_key, slice_rows in slice_accumulator.items():
            slice_answered = [row for row in slice_rows if not row["abstained"]]
            slice_reports[slice_key] = {
                "sample_count": len(slice_rows),
                "answered_count": len(slice_answered),
                "coverage": len(slice_answered) / max(len(slice_rows), 1),
                "selective_accuracy": sum(1 for row in slice_answered if row["correct"]) / max(len(slice_answered), 1),
            }
        return ClinicalBenchmarkReport(
            benchmark_name=self.benchmark_name,
            sample_count=sample_count,
            answered_count=len(answered),
            abstained_count=sample_count - len(answered),
            top1_accuracy=correct_count / max(sample_count, 1),
            coverage=len(answered) / max(sample_count, 1),
            selective_accuracy=answered_correct_count / max(len(answered), 1),
            records=rows,
            slices=slice_reports,
            governance={
                "mode": "retrospective_labeled_benchmark",
                "requires_dataset_card": True,
                "requires_deidentification": True,
                "clinical_warning": "Benchmark results are not prospective clinical validation.",
            },
        )
