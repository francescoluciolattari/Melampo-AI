from dataclasses import dataclass
from typing import Dict


@dataclass
class ValidationMatrix:
    config: object
    logger: object

    def summarize(self) -> Dict[str, str]:
        allow_remote = bool(getattr(self.config, "allow_remote_models", False))
        runtime_profile = getattr(self.config, "runtime_profile", "local_research")
        quantum_status = "enabled_research" if allow_remote else "disabled_local_research"
        return {
            "technical": "scaffold_ready",
            "clinical": "research_only",
            "metacognitive": "traceable_scaffold",
            "theoretical_quantum": quantum_status,
            "runtime_profile": runtime_profile,
        }
