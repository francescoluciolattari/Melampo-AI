from dataclasses import dataclass


@dataclass
class GenerativeReplayEngine:
    config: object
    logger: object

    def generate(self, context: dict) -> dict:
        return {"provider": "api_for_service_synthetic_case_generator", "context": context}
