"""All of a case's uploaded files, processed once, each by its own type -- step 3 of the case-attachment work.

A single case can arrive with blood tests as a PDF, an MRI as DICOM files, a
photographed X-ray as a JPG, and a typed or dictated note, all together.
Each attachment goes through ClinicalDocumentProcessor.process_document_bytes()
(format detected from the bytes, DICOM routed to dicom_handler.py) exactly
once; the results are then split two ways:

- **Text** (a PDF's content, a DICOM report, an OCR'd image) is combined into
  one block, each attachment under its own labelled header, appended AFTER
  the physician's own note -- the note comes first, nothing overwrites
  anything else, the same "nothing discarded" principle as the pending-case
  report merge.
- **Images** become ImagingStudy objects: DICOM frames grouped by series
  (StudyInstanceUID + SeriesInstanceUID), one study per series.

**A plain JPG/PNG is ambiguous, resolved by declaration, never guessed.**
The same JPG could be a photographed lab report (text to extract) or an
X-ray exported as an image (an imaging study). An attachment with a
declared `modality` ("RX", "US", "MR"...) is treated as imaging; without
one, as a document to read. The intake dashboard, when built, is the place
to ask the physician -- guessing from pixel content here would mean a lab
report silently treated as an X-ray, or the reverse.

**Filenames never enter the text.** A header reads "Allegato 2 -- PDF", not
the file's name: files are routinely named after the patient
("Rossi_Mario_emocromo.pdf"), and report_text is persisted with the case.
"""

from __future__ import annotations

import base64
import binascii
import io
from dataclasses import dataclass, field
from typing import Any

from ..types import ImagingStudy, Modality
from .document_processing import (
    FORMAT_DICOM,
    ClinicalDocumentProcessor,
    detect_document_format,
)

_FORMAT_LABELS = {"pdf": "PDF", "png": "immagine PNG", "jpeg": "immagine JPEG", "dicom": "DICOM", "text": "testo"}

# Dashboard/Italian names a physician would use, mapped to DICOM modality
# codes; anything else is passed to Modality's own alias resolution.
_DECLARED_MODALITY_ALIASES = {
    "RX": "CR",
    "RADIOGRAFIA": "CR",
    "TAC": "CT",
    "RM": "MR",
    "RMN": "MR",
    "ECOGRAFIA": "US",
    "ECO": "US",
    "MOC": "DX",  # densitometria ossea (DXA): proiettiva, codificata DX
}


@dataclass(frozen=True)
class CaseAttachment:
    """One uploaded file, held in memory. `modality` is set only when the physician declares an image as imaging."""

    filename: str
    data: bytes
    modality: str | None = None

    @classmethod
    def from_payload_item(cls, item: dict[str, Any]) -> CaseAttachment:
        """Raw bytes under "data", or base64 under "data_base64" (the form a JSON/HTTP dashboard will send)."""
        if isinstance(item.get("data"), (bytes, bytearray)):
            data = bytes(item["data"])
        elif isinstance(item.get("data_base64"), str):
            try:
                data = base64.b64decode(item["data_base64"], validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"attachment {item.get('filename', '?')!r}: invalid base64") from exc
        else:
            raise TypeError(f"attachment {item.get('filename', '?')!r}: needs 'data' bytes or 'data_base64' text")
        modality = item.get("modality")
        return cls(filename=str(item.get("filename", "")), data=data, modality=str(modality) if modality else None)


@dataclass(frozen=True)
class ProcessedAttachment:
    index: int
    document_format: str
    status: str
    reason: str | None
    text: str
    parser: str | None
    modality: str | None
    images_png: tuple[bytes, ...] = ()
    dicom_metadata: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def summary(self) -> dict[str, Any]:
        """JSON-safe, no image bytes, no filename -- what travels with the case result."""
        return {
            "index": self.index,
            "document_format": self.document_format,
            "status": self.status,
            "reason": self.reason,
            "parser": self.parser,
            "modality": self.modality,
            "text_chars": len(self.text),
            "image_count": len(self.images_png),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class AttachmentBundle:
    attachments: tuple[ProcessedAttachment, ...]

    def combined_text(self, physician_note: str = "") -> str:
        """The physician's own note first, then every attachment's text under a labelled header -- nothing overwritten."""
        parts = [physician_note.strip()] if physician_note.strip() else []
        for attachment in self.attachments:
            if not attachment.text.strip():
                continue
            label = _FORMAT_LABELS.get(attachment.document_format, attachment.document_format)
            if attachment.modality:
                label = f"{label}, {attachment.modality}"
            parts.append(f"[Allegato {attachment.index} -- {label}]\n{attachment.text.strip()}")
        return "\n\n".join(parts)

    def imaging_studies(self) -> list[ImagingStudy]:
        """One ImagingStudy per DICOM series (grouped by Study+Series UID), one per declared image."""
        grouped: dict[tuple[str, str], list[ProcessedAttachment]] = {}
        standalone: list[ProcessedAttachment] = []
        for attachment in self.attachments:
            if not attachment.images_png:
                continue
            if attachment.document_format == FORMAT_DICOM:
                key = (
                    str(attachment.dicom_metadata.get("StudyInstanceUID", "")),
                    str(attachment.dicom_metadata.get("SeriesInstanceUID", f"attachment-{attachment.index}")),
                )
                grouped.setdefault(key, []).append(attachment)
            else:
                standalone.append(attachment)

        studies: list[ImagingStudy] = []
        for number, ((_study_uid, series_uid), members) in enumerate(grouped.items(), start=1):
            members = sorted(members, key=lambda item: _safe_float(item.dicom_metadata.get("InstanceNumber")))
            first = members[0]
            studies.append(
                ImagingStudy(
                    study_id=f"attachment-series-{number}",
                    modality=_to_modality(first.modality),
                    images_png=[image for member in members for image in member.images_png],
                    metadata={
                        **{k: v for k, v in first.dicom_metadata.items() if k != "InstanceNumber"},
                        "source": "dicom_attachment",
                        "instance_count": len(members),
                        "SeriesInstanceUID": series_uid,
                    },
                )
            )
        for attachment in standalone:
            studies.append(
                ImagingStudy(
                    study_id=f"attachment-image-{attachment.index}",
                    modality=_to_modality(attachment.modality),
                    images_png=list(attachment.images_png),
                    metadata={"source": "declared_image_attachment", "Modality": _to_modality(attachment.modality).value},
                )
            )
        return studies

    def summary(self) -> list[dict[str, Any]]:
        return [attachment.summary() for attachment in self.attachments]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def _resolve_modality(declared: str | None) -> Modality | None:
    """A declared or DICOM modality as a Modality member, or None when unknown -- never raises."""
    if not declared or not declared.strip():
        return None
    normalized = declared.strip().upper()
    try:
        return Modality(_DECLARED_MODALITY_ALIASES.get(normalized, normalized))
    except ValueError:  # Modality._missing_ found no match
        return None


def _to_modality(declared: str | None) -> Modality:
    return _resolve_modality(declared) or Modality.DICOM


def _as_png(data: bytes) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    image = Image.open(io.BytesIO(data))
    image.convert("RGB" if image.mode not in ("L", "RGB") else image.mode).save(buffer, format="PNG")
    return buffer.getvalue()


def process_case_attachments(
    attachments: list[CaseAttachment], processor: ClinicalDocumentProcessor | None = None
) -> AttachmentBundle:
    """Process every attachment exactly once, by its own type -- never writes to disk."""
    processor = processor or ClinicalDocumentProcessor()
    processed: list[ProcessedAttachment] = []
    for index, attachment in enumerate(attachments, start=1):
        detected = detect_document_format(attachment.data)
        declared = _resolve_modality(attachment.modality) if attachment.modality else None
        if declared is not None and detected in ("png", "jpeg"):
            # Declared imaging: no text to read, so no OCR call at all.
            processed.append(
                ProcessedAttachment(
                    index=index, document_format=detected, status="completed", reason=None, text="",
                    parser=None, modality=declared.value, images_png=(_as_png(attachment.data),),
                )
            )
            continue

        result = processor.process_document_bytes(attachment.data, source_name=f"attachment-{index}")
        document_format = str(result.get("document_format", "unknown"))
        text = "\n".join(doc.get("text", "") for doc in result.get("documents", [])) if result.get("status") == "completed" else ""
        notes: list[str] = []

        if document_format == FORMAT_DICOM:
            dicom = result.get("dicom", {})
            processed.append(
                ProcessedAttachment(
                    index=index, document_format=document_format, status=str(result.get("status")),
                    reason=result.get("reason"), text=text, parser=result.get("parser"),
                    modality=dicom.get("modality"), images_png=tuple(result.get("dicom_images_png", [])),
                    dicom_metadata=dict(dicom.get("metadata", {})), notes=tuple(dicom.get("notes", [])),
                )
            )
            continue

        if attachment.modality and declared is None:
            notes.append(f"declared_modality_not_recognised: {attachment.modality}")
        elif attachment.modality:
            notes.append(f"declared_modality_ignored_for_{document_format}")

        processed.append(
            ProcessedAttachment(
                index=index, document_format=document_format, status=str(result.get("status")),
                reason=result.get("reason"), text=text, parser=result.get("parser"),
                modality=None, notes=tuple(notes),
            )
        )
    return AttachmentBundle(attachments=tuple(processed))
