from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(slots=True)
class ChestXray14CsvLoader:
    """Convert ChestX-ray14-style CSV rows into Melampo prototype payloads.

    The loader is intentionally metadata-only. It does not download or bundle
    images. Local image paths can be provided by the caller after accepting the
    source dataset terms.
    """

    image_root: str | None = None
    source_name: str = "NIH ChestX-ray14-style metadata"

    def _find(self, row: Mapping[str, str], candidates: Iterable[str], default: str = "") -> str:
        for candidate in candidates:
            if candidate in row and row[candidate] is not None:
                value = str(row[candidate]).strip()
                if value:
                    return value
        return default

    def row_to_payload(self, row: Mapping[str, str], row_index: int = 0) -> dict:
        image_id = self._find(row, ["Image Index", "image_id", "image", "filename"], f"cxr-row-{row_index}")
        findings = self._find(row, ["Finding Labels", "findings", "labels"], "No Finding")
        age = self._find(row, ["Patient Age", "age"], "")
        sex = self._find(row, ["Patient Gender", "Patient Sex", "sex", "gender"], "")
        view_position = self._find(row, ["View Position", "view_position"], "")
        follow_up = self._find(row, ["Follow-up #", "follow_up"], "")
        patient_id = self._find(row, ["Patient ID", "patient_id"], f"synthetic-cxr-patient-{row_index}")

        series_paths = []
        if self.image_root:
            series_paths.append(str(Path(self.image_root) / image_id))

        report_text = (
            f"ChestX-ray14-style weak-label case. Finding labels: {findings}. "
            "Labels are metadata-derived and should be treated as weak supervision only."
        )
        return {
            "case_id": f"cxr14-{Path(image_id).stem or row_index}",
            "patient_id": str(patient_id),
            "demographics": {
                "age": age,
                "sex": sex,
            },
            "report_text": report_text,
            "ehr_text": "",
            "patient_complaints": "",
            "observations": [],
            "imaging": [
                {
                    "study_id": image_id,
                    "modality": "CR",
                    "series_paths": series_paths,
                    "metadata": {
                        "source_dataset_style": self.source_name,
                        "image_id": image_id,
                        "finding_labels": findings,
                        "view_position": view_position,
                        "follow_up": follow_up,
                    },
                }
            ],
            "provenance": {
                "source": self.source_name,
                "contains_real_patient_data": False,
                "note": "Generated from metadata. Verify source terms and de-identification status before use.",
            },
        }

    def load_csv(self, csv_path: str | Path, limit: int | None = None) -> list[dict]:
        payloads = []
        with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader):
                if limit is not None and len(payloads) >= limit:
                    break
                payloads.append(self.row_to_payload(row, row_index=index))
        return payloads
