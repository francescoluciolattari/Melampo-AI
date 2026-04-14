from dataclasses import dataclass
from typing import Dict


@dataclass
class ValidationMatrix:
    config: object
    logger: object

    def summarize(self) -> Dict[str, str]:
        return {
            "technical": "pending",
            "clinical": "pending",
            "metacognitive": "pending",
            "theoretical_quantum": "pending",
        }
