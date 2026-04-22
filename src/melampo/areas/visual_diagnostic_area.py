from dataclasses import dataclass


@dataclass
class VisualDiagnosticArea:
    """Aggregate imaging-like and visually encoded clinical stimuli."""

    def integrate(self, volume_features: dict, pathology_features: dict, patient_visual: dict | None = None, labs_snapshot: dict | None = None) -> dict:
        return {
            "area": "visual_diagnostic",
            "volume": volume_features,
            "pathology": pathology_features,
            "patient_visual": patient_visual or {},
            "labs_snapshot": labs_snapshot or {},
            "salient_streams": [key for key in ["volume", "pathology", "patient_visual", "labs_snapshot"]],
        }
