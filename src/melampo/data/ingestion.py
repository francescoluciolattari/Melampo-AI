from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..types import CaseContext, ClinicalObservation, ImagingStudy, Modality

ATTACHMENT_BUNDLE_KEY = "_attachment_bundle"


@dataclass(slots=True)
class ClinicalIngestionPipeline:
    """Collect raw case assets into a single canonical case object.

    Uploaded files (payload["attachments"]: PDF, JPG/PNG, DICOM) are
    processed by prepare_payload() -- once, in memory -- before anything
    else reads the payload. clinical_pipeline.run() calls it first,
    deliberately ahead of the pending-case routing: that routing merges
    this submission's report_text into a pending case's previous report, so
    the attachments' extracted text must already be part of report_text by
    then, or it would be silently lost on exactly the merge_and_rerun path.
    """

    document_processor: Any = None
    # Out-of-range laboratory rows in the attachments become HPO phenotype
    # findings (data/lab_phenotypes.py), appended after the caller's own.
    derive_lab_findings: bool = True

    def prepare_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Process attachments once: report_text becomes note + labelled attachment text; raw bytes leave the payload.

        Idempotent -- a payload already prepared is returned unchanged, so
        a caller that also goes through from_payload() never triggers a
        second parse (a second Nemotron-Parse call per page, for nothing).
        """
        from .case_attachments import CaseAttachment, process_case_attachments
        from .document_processing import ClinicalDocumentProcessor

        prepared = dict(payload)
        raw = prepared.get("attachments")
        if ATTACHMENT_BUNDLE_KEY in prepared or not isinstance(raw, list) or not raw:
            return prepared
        items = [CaseAttachment.from_payload_item(item) for item in raw if isinstance(item, dict)]
        processor = self.document_processor or ClinicalDocumentProcessor.from_env()
        bundle = process_case_attachments(items, processor=processor)
        prepared.pop("attachments")
        prepared["report_text"] = bundle.combined_text(str(payload.get("report_text", "") or ""))
        prepared[ATTACHMENT_BUNDLE_KEY] = bundle
        prepared["attachment_summary"] = bundle.summary()
        if self.derive_lab_findings:
            self._add_lab_findings(prepared, bundle)
        return prepared

    @staticmethod
    def _add_lab_findings(prepared: dict[str, Any], bundle: Any) -> None:
        """The attachments' out-of-range laboratory rows as HPO phenotype findings -- the caller's findings first, none duplicated.

        Here, in prepare_payload, because this is the one point before
        anything reads payload["findings"]: clinical_pipeline.run() calls it
        first, then routes on and enumerates from those findings. A case
        whose attachments carry abnormal laboratory values therefore now
        reaches the diagnostic graph even when the physician typed no
        findings at all -- and, with it, the graph's own load and the
        per-candidate enumeration cost (see clinical_pipeline's
        NEXUS_ENUMERATION_CANDIDATE_CAP). Provenance is kept beside it:
        which phenotype came from which attachment line, and why every
        other row was not used.
        """
        from .lab_phenotypes import map_observations_to_phenotypes

        mapping = map_observations_to_phenotypes(bundle.observations())
        raw = prepared.get("findings")
        provenance = mapping.as_dict()
        if raw is not None and not isinstance(raw, (list, tuple)):
            # An unexpected shape is left exactly as the caller sent it.
            provenance["added_findings"] = []
            provenance["not_merged_reason"] = "caller_findings_not_a_list"
            prepared["lab_phenotypes"] = provenance
            return
        existing = [str(item) for item in (raw or [])]
        seen = {item.strip().lower() for item in existing}
        added = [label for label in mapping.finding_labels() if label.strip().lower() not in seen]
        if added:
            prepared["findings"] = existing + added
        provenance["added_findings"] = added
        prepared["lab_phenotypes"] = provenance

    def from_payload(self, payload: Mapping[str, Any]) -> CaseContext:
        payload = self.prepare_payload(payload)
        raw_observations = payload.get("observations", [])
        observation_items = raw_observations if isinstance(raw_observations, list) else []
        observations = [
            ClinicalObservation(code=str(item["code"]), value=item.get("value"), unit=item.get("unit"), source=item.get("source"))
            for item in observation_items
            if isinstance(item, dict) and "code" in item
        ]
        raw_imaging = payload.get("imaging", [])
        imaging_items = raw_imaging if isinstance(raw_imaging, list) else []
        imaging = [
            ImagingStudy(
                study_id=str(item["study_id"]),
                modality=Modality(str(item["modality"])),
                series_paths=list(item.get("series_paths", [])) if isinstance(item.get("series_paths", []), list) else [],
                metadata=dict(item.get("metadata", {})) if isinstance(item.get("metadata", {}), dict) else {},
            )
            for item in imaging_items
            if isinstance(item, dict) and "study_id" in item and "modality" in item
        ]
        bundle = payload.get(ATTACHMENT_BUNDLE_KEY)
        if bundle is not None:
            imaging.extend(bundle.imaging_studies())
            # Laboratory and antibiogram rows read from the attachments' text
            # (data/lab_results.py), after any passed in the payload itself.
            observations.extend(bundle.observations())
        raw_demographics = payload.get("demographics", {})
        raw_provenance = payload.get("provenance", {})
        provenance = dict(raw_provenance) if isinstance(raw_provenance, dict) else {}
        if bundle is not None:
            provenance["attachments"] = bundle.summary()
        if isinstance(payload.get("lab_phenotypes"), dict):
            provenance["lab_phenotypes"] = payload["lab_phenotypes"]
        return CaseContext(
            case_id=str(payload["case_id"]),
            patient_id=str(payload["patient_id"]) if payload.get("patient_id") is not None else None,
            demographics=dict(raw_demographics) if isinstance(raw_demographics, dict) else {},
            observations=observations,
            imaging=imaging,
            report_text=str(payload.get("report_text", "")),
            ehr_text=str(payload.get("ehr_text", "")),
            provenance=provenance,
        )
