from dataclasses import dataclass


@dataclass
class TrainingCurriculum:
    """Progressive curriculum placeholder for simple, rare, and ambiguous cases."""

    def stages(self) -> list:
        return ["core_cases", "rare_cases", "ambiguous_cases", "counterfactual_cases"]
