from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class CalibrationBin:
    bin_index: int
    lower: float
    upper: float
    count: int = 0
    mean_confidence: float = 0.0
    empirical_accuracy: float = 0.0

    @property
    def gap(self) -> float:
        return abs(self.mean_confidence - self.empirical_accuracy)

    def as_dict(self) -> dict[str, Any]:
        return {
            "bin_index": self.bin_index,
            "lower": round(self.lower, 4),
            "upper": round(self.upper, 4),
            "count": self.count,
            "mean_confidence": round(self.mean_confidence, 4),
            "empirical_accuracy": round(self.empirical_accuracy, 4),
            "gap": round(self.gap, 4),
        }


@dataclass(slots=True)
class CalibrationReport:
    sample_count: int
    bin_count: int
    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score: float
    bins: list[CalibrationBin] = field(default_factory=list)
    governance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "bin_count": self.bin_count,
            "expected_calibration_error": round(self.expected_calibration_error, 6),
            "maximum_calibration_error": round(self.maximum_calibration_error, 6),
            "brier_score": round(self.brier_score, 6),
            "bins": [bin_.as_dict() for bin_ in self.bins],
            "governance": self.governance,
        }


@dataclass(slots=True)
class ConfidenceCalibrationEvaluator:
    """Calibration metrics for diagnostic confidence scores.

    This evaluator treats each prediction as a binary event: the selected label
    was correct or incorrect. It is dependency-free and suitable for CI, smoke
    tests and dataset-agnostic evaluation. Production calibration should add
    per-disease, per-modality and per-site slices.
    """

    bin_count: int = 10

    def evaluate(self, records: Iterable[dict[str, Any]]) -> CalibrationReport:
        parsed: list[tuple[float, float]] = []
        for record in records:
            confidence = _clamp(_safe_float(record.get("confidence", record.get("score", 0.0))))
            correct = 1.0 if bool(record.get("correct", False)) else 0.0
            parsed.append((confidence, correct))

        bins: list[list[tuple[float, float]]] = [[] for _ in range(self.bin_count)]
        for confidence, correct in parsed:
            index = min(int(confidence * self.bin_count), self.bin_count - 1)
            bins[index].append((confidence, correct))

        calibration_bins: list[CalibrationBin] = []
        sample_count = len(parsed)
        expected_calibration_error = 0.0
        maximum_calibration_error = 0.0
        brier_score = 0.0
        for index, values in enumerate(bins):
            lower = index / self.bin_count
            upper = (index + 1) / self.bin_count
            if values:
                mean_confidence = sum(confidence for confidence, _ in values) / len(values)
                empirical_accuracy = sum(correct for _, correct in values) / len(values)
                gap = abs(mean_confidence - empirical_accuracy)
                expected_calibration_error += (len(values) / max(sample_count, 1)) * gap
                maximum_calibration_error = max(maximum_calibration_error, gap)
                brier_score += sum((confidence - correct) ** 2 for confidence, correct in values)
            else:
                mean_confidence = 0.0
                empirical_accuracy = 0.0
            calibration_bins.append(
                CalibrationBin(
                    bin_index=index,
                    lower=lower,
                    upper=upper,
                    count=len(values),
                    mean_confidence=mean_confidence,
                    empirical_accuracy=empirical_accuracy,
                )
            )
        brier_score = brier_score / max(sample_count, 1)
        return CalibrationReport(
            sample_count=sample_count,
            bin_count=self.bin_count,
            expected_calibration_error=expected_calibration_error,
            maximum_calibration_error=maximum_calibration_error,
            brier_score=brier_score,
            bins=calibration_bins,
            governance={
                "interpretation": "lower_ece_mce_brier_are_better",
                "scope": "selected_prediction_binary_correctness",
                "required_next_slices": ["modality", "pathology_family", "site", "prevalence_band", "learning_status"],
                "clinical_warning": "Calibration metrics do not imply clinical validity without prospective validation.",
            },
        )

    def suggest_thresholds(self, report: CalibrationReport, target_min_accuracy: float = 0.8) -> dict[str, Any]:
        eligible_bins = [bin_ for bin_ in report.bins if bin_.count > 0 and bin_.empirical_accuracy >= target_min_accuracy]
        if not eligible_bins:
            return {
                "status": "no_threshold_suggested",
                "reason": "no_confidence_bin_reached_target_accuracy",
                "target_min_accuracy": target_min_accuracy,
            }
        threshold = min(bin_.lower for bin_ in eligible_bins)
        return {
            "status": "threshold_suggested",
            "target_min_accuracy": target_min_accuracy,
            "suggested_min_confidence": round(threshold, 4),
            "eligible_bins": [bin_.as_dict() for bin_ in eligible_bins],
            "governance": "Use as research guidance only; validate prospectively before clinical use.",
        }


@dataclass(slots=True)
class CalibrationAnalysis:
    """Backward-compatible facade for older smoke tests."""

    def summarize(self) -> dict:
        empty_report = ConfidenceCalibrationEvaluator().evaluate([])
        return {
            "ece": empty_report.expected_calibration_error,
            "brier_score": empty_report.brier_score,
            "mce": empty_report.maximum_calibration_error,
        }
