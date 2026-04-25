from dataclasses import dataclass


@dataclass
class QuantumResearchGate:
    """Gate for deciding when the optional quantum-like path is worth evaluating."""

    min_contextuality_score: float = 0.6

    def assess(self, contextuality_score: float) -> dict:
        allow = contextuality_score >= self.min_contextuality_score
        margin = round(contextuality_score - self.min_contextuality_score, 3)
        level = "high" if contextuality_score >= (self.min_contextuality_score + 0.2) else "guarded" if allow else "low"
        reasons = ["contextuality_above_quantum_gate_threshold"] if allow else ["contextuality_below_quantum_gate_threshold"]
        return {
            "allow": allow,
            "contextuality_score": contextuality_score,
            "threshold": self.min_contextuality_score,
            "margin": margin,
            "level": level,
            "reasons": reasons,
        }

    def allow(self, contextuality_score: float) -> bool:
        return self.assess(contextuality_score=contextuality_score)["allow"]
