from dataclasses import dataclass


@dataclass
class StatisticalDreamer:
    provider: str = "api_for_service_synthetic_case_generator"

    def dream(self, context: dict) -> dict:
        return {"provider": self.provider, "context_keys": sorted(context.keys())}
