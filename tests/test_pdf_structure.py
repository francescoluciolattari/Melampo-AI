"""Tests for the PDF structural classifier -- text_native / mixed /
scanned_image_only / empty / unknown (data/document_processing.py).

Regression target: a text PDF carrying an embedded raster picture (an
electrophoresis trace, an ECG snapshot alongside a typed report) previously
came back "completed" from its text layer alone via pdftotext, with the
picture silently dropped -- not reported missing, not queued for review,
gone with no trace in the result. classify_pdf_structure() plus
_list_pdf_embedded_images()/_extract_pdf_embedded_images() (via poppler's
pdfimages, entirely in memory like pdftotext already was) close that gap.

PDFs built by hand, kept local (same convention as test_dicom_handler.py
and test_case_attachments.py) -- reportlab is not a project dependency, so
a JPEG XObject (encoded with Pillow, already a dependency) is embedded
directly with /Filter/DCTDecode. Poppler logs harmless "incorrect stream
length" syntax warnings on these hand-built files to stderr (it recovers by
scanning forward); this project's code only reads stdout and the return
code, exactly as for the pre-existing _digital_pdf helper other test files
already rely on, so this is not a defect being tested around.
"""

import io

import pytest
from PIL import Image

from melampo.data.case_attachments import CaseAttachment, process_case_attachments
from melampo.data.document_processing import (
    PDF_STRUCTURE_EMPTY,
    PDF_STRUCTURE_MIXED,
    PDF_STRUCTURE_SCANNED_IMAGE_ONLY,
    PDF_STRUCTURE_TEXT_NATIVE,
    PDF_STRUCTURE_UNKNOWN,
    ClinicalDocumentProcessor,
    classify_pdf_structure,
)


def _jpeg(width: int = 150, height: int = 100) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "gray").save(buffer, format="JPEG")
    return buffer.getvalue()


def _pdf(text: str | None, image_jpeg: bytes | None, image_size: tuple[int, int] = (150, 100)) -> bytes:
    """A minimal one-page PDF with an optional text-drawing op and/or an
    optional embedded JPEG image XObject -- whichever combination the case
    needs. Kept local so this file never imports another test module."""
    content_ops = []
    if text is not None:
        content_ops.append(f"BT /F1 12 Tf 10 250 Td ({text}) Tj ET".encode("latin-1"))
    if image_jpeg is not None:
        content_ops.append(b"q 150 0 0 100 10 100 cm /Im0 Do Q")
    content = b" ".join(content_ops)

    resources = b"<</Font<</F1 5 0 R>>"
    if image_jpeg is not None:
        resources += b"/XObject<</Im0 6 0 R>>"
    resources += b">>"

    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 400 300]/Contents 4 0 R/Resources" + resources + b">>",
        b"<</Length " + str(len(content)).encode() + b">>stream\n" + content + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    if image_jpeg is not None:
        width, height = image_size
        objects.append(
            b"<</Type/XObject/Subtype/Image/Width " + str(width).encode() + b"/Height " + str(height).encode()
            + b"/ColorSpace/DeviceRGB/BitsPerComponent 8/Filter/DCTDecode/Length " + str(len(image_jpeg)).encode()
            + b">>stream\n" + image_jpeg + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj".encode() + body + b"endobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer<</Size {len(objects) + 1}/Root 1 0 R>>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)


@pytest.fixture
def processor():
    return ClinicalDocumentProcessor()


# ---------------------------------------------------------------------------
# classify_pdf_structure() as a pure function
# ---------------------------------------------------------------------------


def test_text_with_no_images_is_text_native():
    assert classify_pdf_structure("Emoglobina 13.5 g/dL", []) == PDF_STRUCTURE_TEXT_NATIVE


def test_text_with_an_embedded_image_is_mixed():
    assert classify_pdf_structure("Elettroforesi referto", [{"page": 1, "width": 150, "height": 100}]) == PDF_STRUCTURE_MIXED


def test_no_text_with_an_image_is_scanned_image_only():
    assert classify_pdf_structure("", [{"page": 1, "width": 595, "height": 842}]) == PDF_STRUCTURE_SCANNED_IMAGE_ONLY


def test_no_text_and_no_image_is_empty():
    assert classify_pdf_structure("", []) == PDF_STRUCTURE_EMPTY


def test_whitespace_only_text_counts_as_no_text():
    assert classify_pdf_structure("   \n\x0c  ", []) == PDF_STRUCTURE_EMPTY


def test_missing_text_or_images_information_is_unknown_not_empty():
    """None means pdftotext/pdfimages could not be run at all -- distinct
    from a confirmed-empty page, since nothing was actually established."""
    assert classify_pdf_structure(None, []) == PDF_STRUCTURE_UNKNOWN
    assert classify_pdf_structure("", None) == PDF_STRUCTURE_UNKNOWN
    assert classify_pdf_structure(None, None) == PDF_STRUCTURE_UNKNOWN


# ---------------------------------------------------------------------------
# End to end through ClinicalDocumentProcessor.process_document_bytes()
# ---------------------------------------------------------------------------


def test_a_pure_text_pdf_is_classified_text_native_with_no_images(processor):
    result = processor.process_document_bytes(_pdf("Emoglobina 13.5 g/dL", None), prefer_structured_parser=False)
    assert result["status"] == "completed"
    assert result["pdf_structure"] == PDF_STRUCTURE_TEXT_NATIVE
    assert result["embedded_images_png"] == []
    assert "Emoglobina" in result["text"]


def test_a_text_pdf_with_an_embedded_picture_is_mixed_and_the_picture_is_kept(processor):
    """The exact defect this closes: previously this picture vanished with no trace at all."""
    jpeg = _jpeg(150, 100)
    result = processor.process_document_bytes(_pdf("Elettroforesi referto", jpeg, (150, 100)), prefer_structured_parser=False)
    assert result["status"] == "completed"
    assert result["pdf_structure"] == PDF_STRUCTURE_MIXED
    assert "Elettroforesi" in result["text"]
    assert len(result["embedded_images_png"]) == 1
    recovered = Image.open(io.BytesIO(result["embedded_images_png"][0]))
    assert recovered.size == (150, 100)


def test_an_image_only_pdf_has_no_text_but_keeps_the_image(processor):
    """No OCR configured, so no text -- but the page's only content is not silently thrown away."""
    jpeg = _jpeg(595, 842)
    result = processor.process_document_bytes(_pdf(None, jpeg, (595, 842)), prefer_structured_parser=False)
    assert result["status"] == "no_text_extracted"
    assert result["reason"] == "pdf_has_no_text_layer_and_no_ocr_available"
    assert result["pdf_structure"] == PDF_STRUCTURE_SCANNED_IMAGE_ONLY
    assert len(result["embedded_images_png"]) == 1
    recovered = Image.open(io.BytesIO(result["embedded_images_png"][0]))
    assert recovered.size == (595, 842)


def test_a_blank_page_is_empty_with_no_text_and_no_image(processor):
    result = processor.process_document_bytes(_pdf(None, None), prefer_structured_parser=False)
    assert result["status"] == "no_text_extracted"
    assert result["pdf_structure"] == PDF_STRUCTURE_EMPTY
    assert result["embedded_images_png"] == []


# ---------------------------------------------------------------------------
# Through the whole case-attachment path (what a real upload exercises)
# ---------------------------------------------------------------------------


def test_case_attachment_summary_reports_structure_without_ever_exposing_image_bytes(processor):
    jpeg = _jpeg(150, 100)
    bundle = process_case_attachments(
        [CaseAttachment(filename="elettroforesi.pdf", data=_pdf("Elettroforesi referto", jpeg, (150, 100)))],
        processor=processor,
    )
    attachment = bundle.attachments[0]
    assert attachment.pdf_structure == PDF_STRUCTURE_MIXED
    assert len(attachment.embedded_images_png) == 1
    summary = attachment.summary()
    assert summary["pdf_structure"] == PDF_STRUCTURE_MIXED
    assert summary["embedded_image_count"] == 1
    assert all(not isinstance(value, bytes) for value in summary.values())


def test_a_mixed_pdf_attachment_never_becomes_a_spurious_imaging_study(processor):
    """embedded_images_png is deliberately separate from images_png/imaging_studies():
    an embedded picture inside a text report is not a declared or detected
    imaging attachment, and must not silently invent one."""
    jpeg = _jpeg(150, 100)
    bundle = process_case_attachments(
        [CaseAttachment(filename="elettroforesi.pdf", data=_pdf("Elettroforesi referto", jpeg, (150, 100)))],
        processor=processor,
    )
    assert bundle.imaging_studies() == []


def test_combined_text_is_unaffected_by_an_embedded_picture(processor):
    jpeg = _jpeg(150, 100)
    bundle = process_case_attachments(
        [CaseAttachment(filename="elettroforesi.pdf", data=_pdf("Elettroforesi referto", jpeg, (150, 100)))],
        processor=processor,
    )
    assert "Elettroforesi referto" in bundle.combined_text()
