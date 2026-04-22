from dataclasses import dataclass


@dataclass
class ClinicalTextEncoder:
    """Runtime-facing clinical text encoder placeholder."""

    config: object
    provider: str = "api_for_service_clinical_text_encoder"

    def encode(self, text: str) -> dict:
        return {
            "provider": self.provider,
            "text_length": len(text),
            "engine": "clinical_text_encoder",
        }
