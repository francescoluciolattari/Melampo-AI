from dataclasses import dataclass


@dataclass
class LanguageListeningArea:
    """Aggregate patient narrative, reports, and vocal-prosodic surrogates."""

    def integrate(self, report_text: str, ehr_text: str = "", patient_complaints: str = "", voice_features: dict | None = None) -> dict:
        voice_features = voice_features or {}
        text_chunks = [chunk for chunk in [patient_complaints, report_text, ehr_text] if chunk]
        merged_text = " | ".join(text_chunks)
        salient_streams = []
        if patient_complaints:
            salient_streams.append("patient_complaints")
        if report_text:
            salient_streams.append("report_text")
        if ehr_text:
            salient_streams.append("ehr_text")
        if voice_features:
            salient_streams.append("voice_features")
        return {
            "area": "language_listening",
            "focus": "language_led",
            "merged_text": merged_text,
            "voice_features": voice_features,
            "text_length": len(merged_text),
            "salient_streams": salient_streams,
            "signal_count": len(salient_streams),
            "salience_score": round(0.15 * len(salient_streams), 3),
        }
