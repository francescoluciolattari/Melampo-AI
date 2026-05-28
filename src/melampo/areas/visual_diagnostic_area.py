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
class VisualDiagnosticArea:
    """Aggregate imaging-like and visually encoded clinical stimuli."""

    def integrate(
        self,
        volume_features: dict,
        pathology_features: dict,
        patient_visual: dict | None = None,
        labs_snapshot: dict | None = None,
        specialist_signal: dict[str, Any] | None = None,
    ) -> dict:
        patient_visual = patient_visual or {}
        labs_snapshot = labs_snapshot or {}
        external_area = _specialist_area_signal(specialist_signal)
        salient_streams = []
        if volume_features:
            salient_streams.append("volume")
        if pathology_features:
            salient_streams.append("pathology")
        if patient_visual:
            salient_streams.append("patient_visual")
        if labs_snapshot:
            salient_streams.append("labs_snapshot")
        if external_area.get("status") not in {None, "not_called"} or external_area.get("claims"):
            salient_streams.append("specialist_radiology_signal")

        base_salience = _clamp(0.2 * len([item for item in salient_streams if item != "specialist_radiology_signal"]))
        specialist_salience = _clamp(float(external_area.get("salience_score", 0.0) or 0.0))
        specialist_uncertainty = _clamp(float(external_area.get("uncertainty_score", 1.0) or 1.0))
        specialist_active = bool(external_area.get("claims")) or external_area.get("status") not in {None, "not_called"}
        salience_score = _clamp(base_salience + (0.25 * specialist_salience if specialist_active else 0.0))
        uncertainty_score = _clamp((1.0 - base_salience) * 0.75 + specialist_uncertainty * 0.25) if specialist_active else _clamp(1.0 - base_salience)

        return {
            "area": "visual_diagnostic",
            "focus": "imaging_led",
            "volume": volume_features,
            "pathology": pathology_features,
            "patient_visual": patient_visual,
            "labs_snapshot": labs_snapshot,
            "specialist_signal": external_area,
            "claims": list(external_area.get("claims", [])),
            "missing_evidence": list(external_area.get("missing_evidence", [])),
            "contradictions": list(external_area.get("contradictions", [])),
            "salient_streams": salient_streams,
            "signal_count": len(salient_streams),
            "salience_score": round(salience_score, 3),
            "uncertainty_score": round(uncertainty_score, 3),
            "governance": {
                "specialist_models_are_signal_providers_only": True,
                "final_authority": "MelampoDiagnosticOrchestrator",
                "no_hidden_network_calls": True,
            },
        }
