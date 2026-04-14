from dataclasses import dataclass


@dataclass
class EWCPenalty:
    """Placeholder for elastic weight consolidation."""

    coefficient: float = 0.4

    def describe(self) -> str:
        return "ewc_penalty_placeholder"
