"""Extract what Melampo needs from a DICOM file held in memory -- images, report text, and safe metadata.

Step 2 of the case-attachment work. Built on pydicom (MIT, the standard
Python DICOM library) and python-gdcm (Apache-2.0) for compressed pixel
data -- nothing reimplemented. Verified against the 193 real sample files
pydicom itself ships (CT, MR, CR, US, SR, and many transfer syntaxes)
before relying on any of it.

**Why python-gdcm and not pylibjpeg-libjpeg.** Without a decoder plugin,
JPEG Lossless (Process 14) and JPEG-LS -- among the most common
compressions for CT/MR exported from a PACS -- cannot be decoded at all
(verified on the sample files). The most-cited plugin for those,
pylibjpeg-libjpeg, is GPLv3; this project is under the Business Source
License 1.1, and a copyleft dependency would be a real distribution
problem for a medical device. python-gdcm (Apache-2.0) decodes the same
formats. With it, 88 of the sample images decode; of the remaining 6, four
are files pydicom ships deliberately malformed for its own tests, and two
are 12-bit JPEG Extended, which this GDCM build does not support -- each
such file is reported with its transfer syntax, never silently dropped.

**What a DICOM file can carry, and what happens to each**:
- Pixel data (CT, MR, RX/CR/DX, US, bone densitometry...) -> rendered to
  8-bit PNG in memory, applying the dataset's own Modality LUT
  (RescaleSlope/Intercept) and VOI LUT (the radiologist-facing window),
  via pydicom's own functions -- not a hand-rolled min/max stretch, which
  would render a CT's clinically meaningful window unreadable.
- A Structured Report (SR) -> its content tree walked into plain text.
- An encapsulated PDF report -> handed to ClinicalDocumentProcessor, the
  same path any uploaded PDF takes; an encapsulated CDA -> XML text.

**Privacy, by construction.** Metadata passes through an explicit
allowlist of clinical/technical tags only -- PatientName, PatientID,
BirthDate, physician and institution names never reach the output,
because they are never copied in the first place (an allowlist cannot
leak a tag nobody thought to block). SR person-name nodes (PNAME, e.g. the
recording observer) are skipped when building report text.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

# Clinical/technical tags only -- the allowlist. Anything not named here is
# never copied out of the dataset.
SAFE_METADATA_TAGS = (
    "Modality",
    "SOPClassUID",
    "StudyDate",
    "StudyDescription",
    "SeriesDescription",
    "BodyPartExamined",
    "ProtocolName",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "InstanceNumber",
    "Rows",
    "Columns",
    "NumberOfFrames",
    "PhotometricInterpretation",
    "SliceThickness",
    "PixelSpacing",
    "SliceLocation",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "KVP",
)

ENCAPSULATED_PDF_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.104.1"
ENCAPSULATED_CDA_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.104.2"

# A multi-frame object (a cine loop, an enhanced CT/MR) can hold hundreds of
# frames; rendering every one to PNG would multiply memory for little
# diagnostic gain at this stage. Evenly spaced frames, first and last always
# included.
MAX_RENDERED_FRAMES = 16


@dataclass(frozen=True)
class DicomExtraction:
    """What one DICOM file yielded -- status, safe metadata, rendered images, and any report text."""

    status: str
    modality: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    images_png: list[bytes] = field(default_factory=list)
    total_frames: int = 0
    report_text: str | None = None
    report_source: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Serialisable summary -- image bytes counted, not inlined."""
        return {
            "status": self.status,
            "modality": self.modality,
            "metadata": self.metadata,
            "rendered_frames": len(self.images_png),
            "total_frames": self.total_frames,
            "report_text": self.report_text,
            "report_source": self.report_source,
            "notes": self.notes,
        }


def _safe_metadata(dataset: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for keyword in SAFE_METADATA_TAGS:
        if keyword not in dataset:
            continue
        value = dataset.get(keyword)
        if keyword == "SOPClassUID":
            metadata[keyword] = str(value)
            metadata["SOPClassName"] = getattr(value, "name", str(value))
        elif hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            try:
                metadata[keyword] = [float(item) for item in value]
            except (TypeError, ValueError):
                metadata[keyword] = [str(item) for item in value]
        else:
            metadata[keyword] = str(value) if not isinstance(value, (int, float)) else value
    return metadata


def _safe_int(value: Any, default: int) -> int:
    """A DICOM integer attribute, or `default` when absent or malformed -- a dirty tag never crashes extraction."""
    try:
        return int(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _frame_indices(total: int) -> list[int]:
    if total <= MAX_RENDERED_FRAMES:
        return list(range(total))
    step = (total - 1) / (MAX_RENDERED_FRAMES - 1)
    return sorted({round(i * step) for i in range(MAX_RENDERED_FRAMES)})


def _to_png(frame: Any, dataset: Any, is_color: bool) -> bytes:
    """One frame to 8-bit PNG, with the dataset's own Modality and VOI LUTs applied for grayscale."""
    import numpy as np
    from PIL import Image
    from pydicom.pixels import apply_modality_lut, apply_voi_lut

    if is_color:
        array = np.asarray(frame)
        if array.dtype != np.uint8:
            array = _stretch_to_uint8(array.astype(np.float64))
        return _encode_png(Image.fromarray(array, mode="RGB"))

    values = apply_modality_lut(frame, dataset)
    if "WindowCenter" in dataset or "VOILUTSequence" in dataset:
        values = apply_voi_lut(values, dataset)
    values = np.asarray(values, dtype=np.float64)
    image = _stretch_to_uint8(values)
    if str(dataset.get("PhotometricInterpretation", "")) == "MONOCHROME1":
        image = 255 - image  # MONOCHROME1: low values are displayed white
    return _encode_png(Image.fromarray(image, mode="L"))


def _stretch_to_uint8(values: Any) -> Any:
    import numpy as np

    low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    return ((values - low) / (high - low) * 255.0).round().astype(np.uint8)


def _encode_png(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _render_images(dataset: Any, notes: list[str]) -> tuple[list[bytes], int]:
    try:
        pixels = dataset.pixel_array
    except Exception as exc:  # noqa: BLE001 - pydicom aggregates every decoder plugin's failure into one error type chain; reported, not swallowed
        transfer_syntax = getattr(getattr(dataset, "file_meta", None), "TransferSyntaxUID", None)
        name = getattr(transfer_syntax, "name", str(transfer_syntax))
        notes.append(f"pixel_data_not_decodable: {name}: {str(exc).splitlines()[0][:160]}")
        return [], 0

    is_color = _safe_int(dataset.get("SamplesPerPixel", 1), default=1) == 3
    # The decoded array's own shape is the authority on frame count -- a
    # declared NumberOfFrames can be malformed (pydicom ships a sample with
    # "1A") and must never crash ingestion of the whole case.
    frame_axis_present = pixels.ndim == (4 if is_color else 3)
    total = int(pixels.shape[0]) if frame_axis_present else 1
    multi_frame = frame_axis_present

    images = []
    for index in _frame_indices(total):
        frame = pixels[index] if multi_frame else pixels
        images.append(_to_png(frame, dataset, is_color))
    if total > len(images):
        notes.append(f"multi_frame_sampled: {len(images)} of {total} frames rendered")
    return images, total


def _structured_report_text(dataset: Any) -> str:
    """The SR content tree as plain text -- person-name (PNAME) nodes skipped, never copied."""
    lines: list[str] = []
    title = dataset.ConceptNameCodeSequence[0].CodeMeaning if "ConceptNameCodeSequence" in dataset else None
    if title:
        lines.append(str(title))

    def walk(sequence: Any, depth: int) -> None:
        for item in sequence:
            value_type = item.get("ValueType")
            concept = item.ConceptNameCodeSequence[0].CodeMeaning if "ConceptNameCodeSequence" in item else ""
            indent = "  " * depth
            if value_type == "TEXT" and item.get("TextValue"):
                lines.append(f"{indent}{concept}: {item.TextValue}" if concept else f"{indent}{item.TextValue}")
            elif value_type == "CODE" and "ConceptCodeSequence" in item:
                lines.append(f"{indent}{concept}: {item.ConceptCodeSequence[0].CodeMeaning}")
            elif value_type == "NUM" and "MeasuredValueSequence" in item:
                measured = item.MeasuredValueSequence[0]
                units = (
                    measured.MeasurementUnitsCodeSequence[0].CodeValue
                    if "MeasurementUnitsCodeSequence" in measured
                    else ""
                )
                lines.append(f"{indent}{concept}: {measured.NumericValue} {units}".rstrip())
            elif value_type in ("DATE", "DATETIME", "TIME"):
                raw = item.get("Date") or item.get("DateTime") or item.get("Time")
                if raw:
                    lines.append(f"{indent}{concept}: {raw}")
            elif value_type == "CONTAINER" and concept:
                lines.append(f"{indent}{concept}")
            # PNAME (a person's name), UIDREF, IMAGE, COMPOSITE: never copied
            if "ContentSequence" in item:
                walk(item.ContentSequence, depth + 1)

    if "ContentSequence" in dataset:
        walk(dataset.ContentSequence, 0)
    return "\n".join(lines).strip()


def extract_dicom(data: bytes, processor: Any = None) -> DicomExtraction:
    """Read one DICOM file from memory and extract images, report text and safe metadata -- never writes to disk.

    `processor` (a ClinicalDocumentProcessor) is used only for an
    encapsulated PDF report, so it takes exactly the path any uploaded PDF
    takes; created on demand when not supplied.
    """
    import pydicom
    from pydicom.errors import BytesLengthException, InvalidDicomError

    try:
        dataset = pydicom.dcmread(io.BytesIO(data))
    except (InvalidDicomError, BytesLengthException, NotImplementedError, EOFError, ValueError, OSError) as exc:
        return DicomExtraction(status="failed", notes=[f"not_readable_as_dicom: {str(exc)[:160]}"])

    notes: list[str] = []
    metadata = _safe_metadata(dataset)
    modality = metadata.get("Modality")
    sop_class = metadata.get("SOPClassUID")

    report_text: str | None = None
    report_source: str | None = None
    if sop_class == ENCAPSULATED_PDF_SOP_CLASS and "EncapsulatedDocument" in dataset:
        if processor is None:
            from .document_processing import ClinicalDocumentProcessor

            processor = ClinicalDocumentProcessor()
        pdf_bytes = bytes(dataset.EncapsulatedDocument).rstrip(b"\x00")
        parsed = processor.process_document_bytes(pdf_bytes, source_name="dicom_encapsulated_pdf")
        if parsed.get("status") == "completed":
            # The whole text, not the overlapping chunks re-joined (which duplicated every overlap).
            report_text = str(parsed.get("text", ""))
            report_source = "encapsulated_pdf"
        else:
            notes.append(f"encapsulated_pdf_no_text: {parsed.get('reason')}")
    elif sop_class == ENCAPSULATED_CDA_SOP_CLASS and "EncapsulatedDocument" in dataset:
        report_text = bytes(dataset.EncapsulatedDocument).rstrip(b"\x00").decode("utf-8", errors="replace")
        report_source = "encapsulated_cda"
    elif modality == "SR":
        report_text = _structured_report_text(dataset) or None
        report_source = "structured_report" if report_text else None

    images: list[bytes] = []
    total_frames = 0
    if "PixelData" in dataset:
        images, total_frames = _render_images(dataset, notes)

    if report_text or images:
        status = "completed"
    elif notes:
        status = "failed"
    else:
        status = "no_content"
        notes.append("no_pixel_data_and_no_report")
    return DicomExtraction(
        status=status, modality=modality, metadata=metadata, images_png=images, total_frames=total_frames,
        report_text=report_text, report_source=report_source, notes=notes,
    )
