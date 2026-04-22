from dataclasses import dataclass


@dataclass
class LanguageListeningArea:
    """Aggregate patient narrative, reports, and vocal-prosodic surrogates."""

    def integrate(self, report_text: str, ehr_text: str = "", patient_complaints: str = "", voice_features: dict | None = None) -> dict:
        merged_text = " | ".join([chunk for chunk in [patient_complaints, report_text, ehr_text] if chunk])
        return {
            "area": "language_listening",
            "merged_text": merged_text,
            "voice_features": voice_features or {},
            "text_length": len(merged_text),
        }
