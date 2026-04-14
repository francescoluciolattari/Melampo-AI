from dataclasses import dataclass


@dataclass
class DecisionHead:
    provider: str = "api_for_service_multimodal_reasoner"

    def predict(self, features: dict) -> dict:
        return {"provider": self.provider, "feature_count": len(features)}
