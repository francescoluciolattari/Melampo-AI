from dataclasses import dataclass


@dataclass
class AbstentionPolicy:
    """Placeholder abstention policy for safe clinical output control."""

    threshold: float = 0.65

    def should_abstain(self, uncertainty: float) -> bool:
        return uncertainty >= self.threshold
