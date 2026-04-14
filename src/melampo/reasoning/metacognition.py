from dataclasses import dataclass


@dataclass
class MetacognitiveController:
    config: object

    def should_abstain(self, total_uncertainty: float) -> bool:
        threshold = getattr(self.config, "abstention_threshold", 0.65)
        return total_uncertainty >= threshold
