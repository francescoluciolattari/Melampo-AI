from dataclasses import dataclass


@dataclass
class EpidemiologyArea:
    """Aggregate prevalence, exposure, and population context signals."""

    def integrate(self, demographics: dict | None = None, provenance: dict | None = None, exposures: dict | None = None) -> dict:
        demographics = demographics or {}
        provenance = provenance or {}
        exposures = exposures or {}
        salient_streams = []
        if demographics:
            salient_streams.append("demographics")
        if provenance:
            salient_streams.append("provenance")
        if exposures:
            salient_streams.append("exposures")
        return {
            "area": "epidemiology",
            "focus": "epidemiology_led",
            "demographics": demographics,
            "provenance": provenance,
            "exposures": exposures,
            "salient_streams": salient_streams,
            "signal_count": len(salient_streams),
            "salience_score": round(0.18 * len(salient_streams), 3),
        }
