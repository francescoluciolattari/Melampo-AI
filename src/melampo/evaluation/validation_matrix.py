from dataclasses import dataclass


@dataclass
class ValidationMatrix:
    config: object
    logger: object

    def summarize(self) -> dict[str, str]:
        allow_remote = bool(getattr(self.config, "allow_remote_models", False))
        runtime_profile = getattr(self.config, "runtime_profile", "local_research")
        return {
            "technical": "scaffold_ready",
            "clinical": "research_only",
            "metacognitive": "traceable_scaffold",
            "remote_models": "enabled_research" if allow_remote else "disabled_local_research",
            "runtime_profile": runtime_profile,
        }
