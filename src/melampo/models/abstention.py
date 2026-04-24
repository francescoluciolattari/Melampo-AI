from dataclasses import dataclass


@dataclass
class AbstentionPolicy:
    """Abstention policy for safe clinical output control."""

    threshold: float = 0.65

    def assess(self, uncertainty: float) -> dict:
        abstain = uncertainty >= self.threshold
        margin = round(uncertainty - self.threshold, 3)
        level = "high" if uncertainty >= (self.threshold + 0.15) else "guarded" if abstain else "low"
        reasons = ["uncertainty_above_abstention_threshold"] if abstain else []
        return {
            "abstain": abstain,
            "threshold": self.threshold,
            "uncertainty": uncertainty,
            "margin": margin,
            "level": level,
            "reasons": reasons,
        }

    def should_abstain(self, uncertainty: float) -> bool:
        return self.assess(uncertainty=uncertainty)["abstain"]
