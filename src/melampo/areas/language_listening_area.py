from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _specialist_area_signal(specialist_signal: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(specialist_signal, dict):
        return {}
    area_signal = specialist_signal.get("area_signal", specialist_signal)
    return area_signal if isinstance(area_signal, dict) else {}


@dataclass
class LanguageListeningArea:
    """Aggregate patient narrative, reports, and vocal-prosodic surrogates."""

    def integrate(
        self,
        report_text: str,
        ehr_text: str = "",
        patient_complaints: str = "",
        voice_features: dict | None = None,
        specialist_signal: dict[str, Any] | None = None,
    ) -> dict:
        voice_features = voice_features or {}
        external_area = _specialist_area_signal(specialist_signal)
        text_chunks = [chunk for chunk in [patient_complaints, report_text, ehr_text] if chunk]
        grounded_summary = external_area.get("signals", {}).get("grounded_summary") if isinstance(external_area.get("signals"), dict) else None
        if grounded_summary:
            text_chunks.append(str(grounded_summary))
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
        if external_area.get("status") not in {None, "not_called"} or external_area.get("claims"):
            salient_streams.append("grounded_specialist_text_signal")

        base_salience = _clamp(0.15 * len([item for item in salient_streams if item != "grounded_specialist_text_signal"]))
        specialist_salience = _clamp(float(external_area.get("salience_score", 0.0) or 0.0))
        specialist_uncertainty = _clamp(float(external_area.get("uncertainty_score", 1.0) or 1.0))
        specialist_active = bool(external_area.get("claims")) or external_area.get("status") not in {None, "not_called"}
        salience_score = _clamp(base_salience + (0.25 * specialist_salience if specialist_active else 0.0))
        uncertainty_score = _clamp((1.0 - base_salience) * 0.75 + specialist_uncertainty * 0.25) if specialist_active else _clamp(1.0 - base_salience)

        return {
            "area": "language_listening",
            "focus": "language_led",
            "merged_text": merged_text,
            "voice_features": voice_features,
            "text_length": len(merged_text),
            "specialist_signal": external_area,
            "claims": list(external_area.get("claims", [])),
            "missing_evidence": list(external_area.get("missing_evidence", [])),
            "contradictions": list(external_area.get("contradictions", [])),
            "salient_streams": salient_streams,
            "signal_count": len(salient_streams),
            "salience_score": round(salience_score, 3),
            "uncertainty_score": round(uncertainty_score, 3),
            "governance": {
                "language_model_must_be_grounded": True,
                "specialist_models_are_signal_providers_only": True,
                "final_authority": "MelampoDiagnosticOrchestrator",
                "no_hidden_network_calls": True,
            },
        }
