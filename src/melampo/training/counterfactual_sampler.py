from dataclasses import dataclass


@dataclass
class CounterfactualSampler:
    """Generate simple counterfactual case variants for replay and review."""

    def sample(self, case_context: dict) -> dict:
        return {
            "source": case_context,
            "mode": "counterfactual_variant",
        }
