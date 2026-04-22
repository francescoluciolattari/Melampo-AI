from dataclasses import dataclass


@dataclass
class EpidemiologyArea:
    """Aggregate prevalence, exposure, and population context signals."""

    def integrate(self, demographics: dict | None = None, provenance: dict | None = None, exposures: dict | None = None) -> dict:
        demographics = demographics or {}
        provenance = provenance or {}
        exposures = exposures or {}
        return {
            "area": "epidemiology",
            "demographics": demographics,
            "provenance": provenance,
            "exposures": exposures,
            "signal_count": len(demographics) + len(provenance) + len(exposures),
        }
