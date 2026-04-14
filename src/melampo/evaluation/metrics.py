from dataclasses import dataclass


@dataclass
class MetricsCatalog:
    """Minimal catalog of metrics expected in Melampo validation."""

    def list_metrics(self) -> list:
        return [
            "diagnostic_accuracy",
            "ece",
            "brier_score",
            "risk_coverage",
            "ood_auroc",
            "false_reassurance_rate",
        ]
