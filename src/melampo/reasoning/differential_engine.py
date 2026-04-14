from dataclasses import dataclass


@dataclass
class DifferentialEngine:
    """Minimal differential engine placeholder."""

    def rank(self, evidence: list) -> dict:
        return {"status": "differential_placeholder", "evidence_count": len(evidence)}
