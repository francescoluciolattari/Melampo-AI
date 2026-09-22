"""Tests for in-memory document processing (step 1 of the case-attachment
work): documents are parsed from bytes already in memory, never written to
disk. Format is detected from each file's own signature, never its name.
Two real defects are pinned here, both verified before fixing:

- the old plain-text fallback decoded ANY file as UTF-8 with errors
  ignored, so a JPEG with Nemotron-Parse unconfigured came back "completed"
  with its binary header ("JFIF...") presented as clinical text;
- every image was sent to Nemotron-Parse labelled image/png, so a JPEG was
  always mislabelled.
"""

import io
from unittest.mock import MagicMock, patch

from PIL import Image

from melampo.data.document_processing import (
    FORMAT_DICOM,
    FORMAT_JPEG,
    FORMAT_PDF,
    FORMAT_PNG,
    FORMAT_TEXT,
    FORMAT_UNKNOWN,
    ClinicalDocumentProcessor,
    _extract_pdf_text_layer,
    detect_document_format,
)


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _digital_pdf(text: str) -> bytes:
    """A minimal, valid PDF with a real text layer -- built by hand, no PDF library dependency."""
    content = f"BT /F1 12 Tf 10 50 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 400 100]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length " + str(len(content)).encode() + b">>stream\n" + content + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
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


def _scanned_pdf() -> bytes:
    """A PDF page with no text layer at all -- what a scanner produces."""
    buffer = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(buffer, format="PDF")
    return buffer.getvalue()


def _dicom_like() -> bytes:
    return b"\x00" * 128 + b"DICM" + b"\x02\x00\x00\x00" * 10


def _fake_nemotron_response(text: str) -> MagicMock:
    return MagicMock(json=lambda: {"choices": [{"message": {"content": text}}]}, raise_for_status=lambda: None)


def _configured() -> ClinicalDocumentProcessor:
    return ClinicalDocumentProcessor(nemotron_parse_endpoint="http://nim.example:8000", nemotron_parse_api_key="k")


# --------------------------------------------------------------------------
# detect_document_format: from the bytes, never the filename
# --------------------------------------------------------------------------


def test_each_format_is_detected_from_its_own_signature():
    assert detect_document_format(_digital_pdf("x")) == FORMAT_PDF
    assert detect_document_format(_png()) == FORMAT_PNG
    assert detect_document_format(_jpeg()) == FORMAT_JPEG
    assert detect_document_format(_dicom_like()) == FORMAT_DICOM
    assert detect_document_format(b"Paziente con febbre.") == FORMAT_TEXT


def test_undecodable_binary_is_unknown_never_text():
    assert detect_document_format(b"\x00\xff\xfe\x80\x81binary") == FORMAT_UNKNOWN


# --------------------------------------------------------------------------
# The garbage-text defect, pinned
# --------------------------------------------------------------------------


def test_an_image_without_ocr_is_no_text_extracted_never_binary_garbage():
    result = ClinicalDocumentProcessor().process_document_bytes(_jpeg(), source_name="emocromo.jpg")
    assert result["status"] == "no_text_extracted"
    assert result["reason"] == "image_requires_nemotron_parse_for_text"
    assert result["chunk_count"] == 0
    assert result["documents"] == []


def test_the_same_defect_is_fixed_through_the_path_api_too(tmp_path):
    path = tmp_path / "emocromo.jpg"
    path.write_bytes(_jpeg())
    result = ClinicalDocumentProcessor().process_document(path)
    assert result["status"] == "no_text_extracted"
    assert result["documents"] == []


def test_an_unknown_binary_is_no_text_extracted():
    result = ClinicalDocumentProcessor().process_document_bytes(b"\x00\xff\xfe\x80binary", source_name="blob")
    assert result["status"] == "no_text_extracted"
    assert result["reason"] == "unrecognised_binary_format"


def test_a_dicom_file_is_recognised_and_left_to_the_dicom_handler():
    result = ClinicalDocumentProcessor().process_document_bytes(_dicom_like(), source_name="rm_encefalo")
    assert result["document_format"] == FORMAT_DICOM
    assert result["status"] == "no_text_extracted"
    assert result["reason"] == "dicom_requires_the_dicom_handler"


# --------------------------------------------------------------------------
# Digital PDFs: the embedded text layer, entirely in memory
# --------------------------------------------------------------------------


def test_pdftotext_reads_a_digital_pdf_text_layer_from_memory():
    assert _extract_pdf_text_layer(_digital_pdf("Emoglobina 13.2 g/dL")) == "Emoglobina 13.2 g/dL"


def test_a_digital_pdf_without_ocr_completes_via_its_text_layer():
    result = ClinicalDocumentProcessor().process_document_bytes(_digital_pdf("Emoglobina 13.2 g/dL"), source_name="lab.pdf")
    assert result["status"] == "completed"
    assert result["parser"] == "pdf_text_layer"
    assert "Emoglobina 13.2" in result["documents"][0]["text"]


def test_a_scanned_pdf_without_ocr_is_honestly_no_text_extracted():
    result = ClinicalDocumentProcessor().process_document_bytes(_scanned_pdf(), source_name="scansione.pdf")
    assert result["status"] == "no_text_extracted"
    assert result["reason"] == "pdf_has_no_text_layer_and_no_ocr_available"


def test_a_missing_pdftotext_binary_is_reported_not_crashed():
    with patch("subprocess.run", side_effect=FileNotFoundError("pdftotext")):
        result = ClinicalDocumentProcessor().process_document_bytes(_digital_pdf("x"), source_name="lab.pdf")
    assert result["status"] == "no_text_extracted"
    assert result["reason"] == "pdftotext_unavailable"


# --------------------------------------------------------------------------
# Nemotron-Parse from bytes, with the real MIME type
# --------------------------------------------------------------------------


def test_a_jpeg_is_sent_to_nemotron_parse_as_image_jpeg():
    with patch("requests.post") as post:
        post.return_value = _fake_nemotron_response("Emoglobina 13.2")
        result = _configured().process_document_bytes(_jpeg(), source_name="emocromo.jpg")
    url = post.call_args.kwargs["json"]["messages"][0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert result["status"] == "completed"
    assert result["parser"] == "nemotron_parse"


def test_a_png_is_sent_as_image_png():
    with patch("requests.post") as post:
        post.return_value = _fake_nemotron_response("x")
        _configured().process_document_bytes(_png(), source_name="x.png")
    url = post.call_args.kwargs["json"]["messages"][0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


def test_a_pdf_is_rendered_from_bytes_and_each_page_sent_as_png():
    with patch("requests.post") as post:
        post.return_value = _fake_nemotron_response("page text")
        result = _configured().process_document_bytes(_digital_pdf("x"), source_name="lab.pdf")
    url = post.call_args.kwargs["json"]["messages"][0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert result["parser"] == "nemotron_parse"


def test_a_misleading_filename_does_not_change_the_detected_format():
    with patch("requests.post") as post:
        post.return_value = _fake_nemotron_response("x")
        _configured().process_document_bytes(_jpeg(), source_name="referto.pdf")
    url = post.call_args.kwargs["json"]["messages"][0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")


def test_a_nemotron_network_failure_falls_back_to_the_pdf_text_layer():
    import requests

    with patch("requests.post") as post:
        post.return_value = MagicMock(raise_for_status=MagicMock(side_effect=requests.HTTPError("down")))
        result = _configured().process_document_bytes(_digital_pdf("Leucociti 7800"), source_name="lab.pdf")
    assert result["status"] == "completed"
    assert result["parser"] == "pdf_text_layer"
    assert result["parser_result"]["status"] == "failed"


def test_a_programming_error_inside_the_parser_is_not_swallowed():
    """The narrowed except: a TypeError is a bug to surface, not an
    'unavailable parser' to degrade from."""
    processor = _configured()
    with patch.object(ClinicalDocumentProcessor, "_call_nemotron_parse_bytes", side_effect=TypeError("a real bug")):
        try:
            processor.process_document_bytes(_png(), source_name="x.png")
            raised = False
        except TypeError:
            raised = True
    assert raised


# --------------------------------------------------------------------------
# No disk writes
# --------------------------------------------------------------------------


def test_processing_bytes_never_writes_a_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ClinicalDocumentProcessor().process_document_bytes(_digital_pdf("x"), source_name="lab.pdf")
    ClinicalDocumentProcessor().process_document_bytes(_jpeg(), source_name="x.jpg")
    assert list(tmp_path.iterdir()) == []
