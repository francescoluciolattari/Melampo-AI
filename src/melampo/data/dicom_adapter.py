from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(slots=True)
class DICOMAdapter:
    """Placeholder DICOM adapter for future PACS and DICOM-SR integration."""

    def inspect_series(self, series_path: str) -> Dict[str, Any]:
        return {
            "series_path": series_path,
            "reader": "api_for_service_dicom_reader",
            "status": "placeholder",
        }
