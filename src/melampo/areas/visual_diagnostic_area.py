from dataclasses import dataclass


@dataclass
class VisualDiagnosticArea:
    """Aggregate imaging-like and visually encoded clinical stimuli."""

    def integrate(self, volume_features: dict, pathology_features: dict, patient_visual: dict | None = None, labs_snapshot: dict | None = None) -> dict:
        patient_visual = patient_visual or {}
        labs_snapshot = labs_snapshot or {}
        salient_streams = []
        if volume_features:
            salient_streams.append("volume")
        if pathology_features:
            salient_streams.append("pathology")
        if patient_visual:
            salient_streams.append("patient_visual")
        if labs_snapshot:
            salient_streams.append("labs_snapshot")
        return {
            "area": "visual_diagnostic",
            "focus": "imaging_led",
            "volume": volume_features,
            "pathology": pathology_features,
            "patient_visual": patient_visual,
            "labs_snapshot": labs_snapshot,
            "salient_streams": salient_streams,
            "signal_count": len(salient_streams),
            "salience_score": round(0.2 * len(salient_streams), 3),
        }
