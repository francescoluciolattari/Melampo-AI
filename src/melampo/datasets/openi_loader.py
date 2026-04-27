from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(slots=True)
class OpenIReportCsvLoader:
    """Convert Open-i / Indiana-style report metadata into Melampo prototype payloads.

    This loader is metadata-only. It does not redistribute reports or images.
    Use it with locally available metadata after checking source license terms.
    """

    image_root: str | None = None
    source_name: str = "Open-i / Indiana-style chest X-ray metadata"

    def _find(self, row: Mapping[str, str], candidates: Iterable[str], default: str = "") -> str:
        for candidate in candidates:
            if candidate in row and row[candidate] is not None:
                value = str(row[candidate]).strip()
                if value:
                    return value
        return default

    def row_to_payload(self, row: Mapping[str, str], row_index: int = 0) -> dict:
        uid = self._find(row, ["uid", "report_id", "id", "case_id"], f"openi-row-{row_index}")
        image_id = self._find(row, ["image", "image_id", "filename", "image_filename"], f"openi-image-{row_index}.png")
        indication = self._find(row, ["indication", "clinical_indication"], "")
        findings = self._find(row, ["findings", "report_findings"], "")
        impression = self._find(row, ["impression", "report_impression"], "")
        problems = self._find(row, ["problems", "labels", "mesh_terms"], "")

        report_parts = []
        if indication:
            report_parts.append(f"Indication: {indication}")
        if findings:
            report_parts.append(f"Findings: {findings}")
        if impression:
            report_parts.append(f"Impression: {impression}")
        if problems:
            report_parts.append(f"Problem terms: {problems}")
        report_text = " ".join(report_parts) or "Open-i-style report metadata without narrative text."

        series_paths = []
        if self.image_root:
            series_paths.append(str(Path(self.image_root) / image_id))

        return {
            "case_id": f"openi-{Path(str(uid)).stem or row_index}",
            "patient_id": f"openi-synthetic-patient-{row_index}",
            "demographics": {},
            "report_text": report_text,
            "ehr_text": "",
            "patient_complaints": indication,
            "observations": [],
            "imaging": [
                {
                    "study_id": image_id,
                    "modality": "CR",
                    "series_paths": series_paths,
                    "metadata": {
                        "source_dataset_style": self.source_name,
                        "uid": uid,
                        "image_id": image_id,
                        "problem_terms": problems,
                    },
                }
            ],
            "provenance": {
                "source": self.source_name,
                "contains_real_patient_data": False,
                "note": "Generated from local metadata. Verify Open-i/Indiana license terms before use or redistribution.",
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
