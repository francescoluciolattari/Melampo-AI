from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


def _as_rows(records: Any) -> list[dict[str, Any]]:
    if records is None:
        return []
    if hasattr(records, "records"):
        return list(getattr(records, "records"))
    if isinstance(records, dict) and isinstance(records.get("records"), list):
        return list(records["records"])
    return [dict(item) for item in records]


@dataclass(slots=True)
class SliceMetric:
    slice_key: str
    sample_count: int
    answered_count: int
    correct_count: int
    coverage: float
    selective_accuracy: float
    mean_confidence: float
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "slice_key": self.slice_key,
            "sample_count": self.sample_count,
            "answered_count": self.answered_count,
            "correct_count": self.correct_count,
            "coverage": round(self.coverage, 6),
            "selective_accuracy": round(self.selective_accuracy, 6),
            "mean_confidence": round(self.mean_confidence, 6),
            "status": self.status,
        }


@dataclass(slots=True)
class SliceAnalysisReport:
    slice_count: int
    metrics: dict[str, SliceMetric] = field(default_factory=dict)
    underperforming_slices: list[str] = field(default_factory=list)
    missing_required_slices: list[str] = field(default_factory=list)
    governance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "slice_count": self.slice_count,
            "metrics": {key: metric.as_dict() for key, metric in self.metrics.items()},
            "underperforming_slices": list(self.underperforming_slices),
            "missing_required_slices": list(self.missing_required_slices),
            "governance": self.governance,
        }


@dataclass(slots=True)
class SliceAnalysisRunner:
    min_slice_size: int = 1
    min_selective_accuracy: float = 0.6
    min_coverage: float = 0.4

    def run(self, records: Iterable[dict[str, Any]] | Any, required_slices: list[str] | None = None) -> SliceAnalysisReport:
        rows = _as_rows(records)
        grouped: dict[str, list[dict[str, Any]]] = {}
        observed_slice_names: set[str] = set()
        for row in rows:
            slices = row.get("slices", {}) if isinstance(row, dict) else {}
            if not isinstance(slices, dict):
                slices = {}
            for name, value in slices.items():
                observed_slice_names.add(str(name))
                grouped.setdefault(f"{name}:{value}", []).append(row)
        metrics: dict[str, SliceMetric] = {}
        underperforming: list[str] = []
        for slice_key, slice_rows in sorted(grouped.items()):
            answered = [row for row in slice_rows if not row.get("abstained", False)]
            correct_count = sum(1 for row in answered if row.get("correct", False))
            confidence_values = [float(row.get("confidence", 0.0) or 0.0) for row in slice_rows]
            coverage = len(answered) / max(len(slice_rows), 1)
            selective_accuracy = correct_count / max(len(answered), 1)
            mean_confidence = sum(confidence_values) / max(len(confidence_values), 1)
            status = "pass"
            if len(slice_rows) < self.min_slice_size:
                status = "too_small"
            elif coverage < self.min_coverage or selective_accuracy < self.min_selective_accuracy:
                status = "underperforming"
                underperforming.append(slice_key)
            metrics[slice_key] = SliceMetric(
                slice_key=slice_key,
                sample_count=len(slice_rows),
                answered_count=len(answered),
                correct_count=correct_count,
                coverage=coverage,
                selective_accuracy=selective_accuracy,
                mean_confidence=mean_confidence,
                status=status,
            )
        required_slices = required_slices or []
        missing_required = sorted(set(required_slices) - observed_slice_names)
        return SliceAnalysisReport(
            slice_count=len(metrics),
            metrics=metrics,
            underperforming_slices=underperforming,
            missing_required_slices=missing_required,
            governance={
                "min_slice_size": self.min_slice_size,
                "min_selective_accuracy": self.min_selective_accuracy,
                "min_coverage": self.min_coverage,
                "clinical_warning": "Slice analysis is a research fairness/safety primitive, not clinical validation.",
            },
        )
