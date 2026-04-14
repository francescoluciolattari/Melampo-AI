from dataclasses import dataclass


@dataclass
class RiskGate:
    """Placeholder risk gate for abstention and escalation."""

    threshold: float = 0.35

    def allow(self, risk: float) -> bool:
        return risk <= self.threshold
