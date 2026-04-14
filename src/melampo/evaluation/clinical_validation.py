from dataclasses import dataclass


@dataclass
class ClinicalValidationPlan:
    """Minimal placeholder for reader studies and workflow validation."""

    def endpoints(self) -> list:
        return [
            "diagnostic_accuracy",
            "differential_quality",
            "time_to_escalation",
            "false_reassurance_rate",
            "calibrated_trust",
        ]
