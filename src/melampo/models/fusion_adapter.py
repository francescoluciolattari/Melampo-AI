from dataclasses import dataclass


@dataclass
class FusionAdapter:
    provider: str = "api_for_service_multimodal_reasoner"

    def fuse(self, inputs: dict) -> dict:
        return {"provider": self.provider, "input_keys": sorted(inputs.keys())}
