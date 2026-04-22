from dataclasses import dataclass


@dataclass
class DifferentialEngine:
    """Baseline differential engine using grounded evidence counts and simple support ranking."""

    def rank(self, evidence: list) -> dict:
        hypotheses = [
            {"label": "working_hypothesis", "score": 0.7, "support": len(evidence)},
            {"label": "alternative_hypothesis", "score": 0.3, "support": max(len(evidence) - 1, 0)},
        ]
        return {
            "status": "grounded_differential_ready",
            "evidence_count": len(evidence),
            "hypotheses": hypotheses,
        }
