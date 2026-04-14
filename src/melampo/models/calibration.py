from dataclasses import dataclass


@dataclass
class CalibrationLayer:
    """Placeholder calibration layer for confidence alignment."""

    def calibrate(self, score: float) -> dict:
        return {"raw": score, "calibrated": score}
