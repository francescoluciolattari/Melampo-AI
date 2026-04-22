from dataclasses import dataclass


@dataclass
class QuantumResearchGate:
    """Gate for deciding when the optional quantum-like path is worth evaluating."""

    min_contextuality_score: float = 0.6

    def allow(self, contextuality_score: float) -> bool:
        return contextuality_score >= self.min_contextuality_score
