from dataclasses import dataclass


@dataclass
class ReplayFilter:
    """Placeholder filter for accepting or rejecting synthetic replay cases."""

    min_coherence: float = 0.7
    max_risk: float = 0.3

    def accept(self, coherence: float, risk: float) -> bool:
        return coherence >= self.min_coherence and risk <= self.max_risk
