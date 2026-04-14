from dataclasses import dataclass


@dataclass
class UncertaintyEstimator:
    """Minimal uncertainty scaffold."""

    def estimate(self) -> dict:
        return {"aleatoric": 0.0, "epistemic": 0.0, "grounding": 0.0, "shift": 0.0}
