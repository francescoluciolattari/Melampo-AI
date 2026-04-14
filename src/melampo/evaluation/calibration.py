from dataclasses import dataclass


@dataclass
class CalibrationAnalysis:
    """Placeholder helper for calibration-focused evaluation."""

    def summarize(self) -> dict:
        return {
            "ece": "pending",
            "brier_score": "pending",
            "nll": "pending",
        }
