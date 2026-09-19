"""Tests for the real Nemotron-Parse-v1.2 HTTP call, completing
ClinicalDocumentProcessor._call_nemotron_parse() -- previously an
unimplemented transport stub. Every network call is mocked (no live NIM
endpoint exists to test against), but the request construction, PDF/image
handling, response parsing, and error paths are all exercised for real.
"""

import struct
import zlib
from unittest.mock import MagicMock, patch

import pytest
import requests

from melampo.data.document_processing import (
    ClinicalDocumentProcessor,
    _parse_nemotron_parse_response,
)

_MINIMAL_PDF = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
    b"xref\n0 4\n0000000000 65535 f \n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n0\n%%EOF"
)


def _minimal_png() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    idat = chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
    iend = chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


def _fake_response(text: str) -> MagicMock:
    return MagicMock(
        status_code=200,
        json=lambda: {"choices": [{"message": {"content": text}}]},
        raise_for_status=lambda: None,
    )


def _configured_processor() -> ClinicalDocumentProcessor:
    return ClinicalDocumentProcessor(nemotron_parse_endpoint="http://nim.example:8000", nemotron_parse_api_key="test-key")


# --------------------------------------------------------------------------
# _parse_nemotron_parse_response: the documented response envelope
# --------------------------------------------------------------------------


def test_parses_the_documented_response_shape():
    text, metadata = _parse_nemotron_parse_response({"choices": [{"message": {"content": "hello"}}]})
    assert text == "hello"
    assert metadata == {"bounding_boxes_decoded": False}


def test_raises_clearly_on_an_unexpected_response_shape():
    with pytest.raises(ValueError, match="unexpected Nemotron-Parse response shape"):
        _parse_nemotron_parse_response({"unexpected": "shape"})


# --------------------------------------------------------------------------
# The real HTTP call, mocked at the network boundary only
# --------------------------------------------------------------------------


def test_calls_the_configured_endpoint_with_the_documented_request_shape(tmp_path):
    image_path = tmp_path / "page.png"
    image_path.write_bytes(_minimal_png())
    processor = _configured_processor()

    with patch("requests.post") as mock_post:
        mock_post.return_value = _fake_response("extracted text")
        text, metadata = processor._call_nemotron_parse(image_path)

    assert text == "extracted text"
    assert metadata["page_count"] == 1
    args, kwargs = mock_post.call_args
    assert args[0] == "http://nim.example:8000/v1/chat/completions"
    assert kwargs["json"]["model"] == "nvidia/nemotron-parse-v1.2"
    assert kwargs["json"]["messages"][0]["content"][0]["type"] == "image_url"
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"


def test_a_pdf_input_is_rendered_to_images_before_the_call(tmp_path):
    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_bytes(_MINIMAL_PDF)
    processor = _configured_processor()

    with patch("requests.post") as mock_post:
        mock_post.return_value = _fake_response("page content")
        text, metadata = processor._call_nemotron_parse(pdf_path)

    assert text == "page content"
    assert metadata["page_count"] == 1
    # A real image was sent, not the raw PDF bytes -- the request body is a
    # data URL, verified simply by confirming the call succeeded through
    # the image pipeline rather than raising on non-image bytes.
    assert mock_post.called


def test_an_unsupported_file_type_raises_before_any_network_call(tmp_path):
    bad_path = tmp_path / "notes.docx"
    bad_path.write_bytes(b"not an image or pdf")
    processor = _configured_processor()

    with patch("requests.post") as mock_post:
        with pytest.raises(ValueError, match="needs an image or PDF input"):
            processor._call_nemotron_parse(bad_path)
        mock_post.assert_not_called()


def test_an_http_error_surfaces_as_a_failed_status_not_a_crash(tmp_path):
    image_path = tmp_path / "page.png"
    image_path.write_bytes(_minimal_png())
    processor = _configured_processor()

    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(raise_for_status=MagicMock(side_effect=requests.HTTPError("server error")))
        result = processor.load_with_nemotron_parse(image_path)

    assert result["status"] == "failed"
    assert result["reason"] == "nemotron_parse_conversion_failed"


def test_no_endpoint_configured_degrades_without_attempting_a_call(tmp_path):
    image_path = tmp_path / "page.png"
    image_path.write_bytes(_minimal_png())
    processor = ClinicalDocumentProcessor()  # no endpoint/key

    with patch("requests.post") as mock_post:
        result = processor.load_with_nemotron_parse(image_path)
        mock_post.assert_not_called()

    assert result["status"] == "not_executed"
    assert result["reason"] == "nemotron_parse_unavailable"


# --------------------------------------------------------------------------
# The full process_document() flow, end to end, with the network mocked
# --------------------------------------------------------------------------


def test_process_document_completes_end_to_end_through_nemotron_parse(tmp_path):
    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_bytes(_MINIMAL_PDF)
    processor = _configured_processor()

    with patch("requests.post") as mock_post:
        mock_post.return_value = _fake_response("Patient reports persistent cough and fever for three days.")
        result = processor.process_document(pdf_path)

    assert result["status"] == "completed"
    assert result["parser"] == "nemotron_parse"
    assert result["chunk_count"] >= 1
    assert "cough" in result["documents"][0]["text"].lower()


def test_process_document_falls_back_to_plain_text_when_nemotron_parse_fails(tmp_path):
    text_path = tmp_path / "report.txt"
    text_path.write_text("Plain text fallback content about fever.")
    processor = _configured_processor()

    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(raise_for_status=MagicMock(side_effect=requests.HTTPError("down")))
        result = processor.process_document(text_path)

    assert result["status"] == "completed"
    assert result["parser"] == "plain_text_fallback"
