from dataclasses import dataclass

from .fusion_adapter import FusionAdapter


@dataclass
class MultimodalFusionEngine:
    """Runtime-facing fusion engine built on top of the provider-neutral adapter."""

    config: object

    def __post_init__(self):
        self.adapter = FusionAdapter()

    def fuse(self, inputs: dict) -> dict:
        result = self.adapter.fuse(inputs)
        result["engine"] = "multimodal_fusion_engine"
        return result
