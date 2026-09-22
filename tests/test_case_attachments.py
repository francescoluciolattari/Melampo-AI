"""Tests for steps 3 and 4 of the case-attachment work: one bundle for all
of a case's uploaded files (case_attachments.py), wired into
ClinicalIngestionPipeline and clinical_pipeline.run(). Real DICOM samples
from pydicom; PDFs built by hand.
"""

import base64
import io
from unittest.mock import patch

import pydicom
import pytest
from PIL import Image
from pydicom.data import get_testdata_file

from melampo.data.case_attachments import CaseAttachment, process_case_attachments
from melampo.data.document_processing import ClinicalDocumentProcessor
from melampo.data.ingestion import ATTACHMENT_BUNDLE_KEY, ClinicalIngestionPipeline
from melampo.types import Modality


def _digital_pdf(text: str) -> bytes:
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


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (60, 40), "gray").save(buffer, format="JPEG")
    return buffer.getvalue()


def _mr_slice(instance_number: int, brightness: int = 0) -> bytes:
    """A real MR sample, same series, distinct InstanceNumber -- and distinct pixels when brightness differs."""
    import numpy as np

    dataset = pydicom.dcmread(get_testdata_file("MR_small.dcm"))
    dataset.InstanceNumber = instance_number
    if brightness:
        pixels = dataset.pixel_array.astype(np.int32) + brightness
        dataset.PixelData = pixels.clip(0, 4000).astype(dataset.pixel_array.dtype).tobytes()
    buffer = io.BytesIO()
    dataset.save_as(buffer)
    return buffer.getvalue()


def _sr() -> bytes:
    return open(get_testdata_file("reportsi.dcm"), "rb").read()


# --------------------------------------------------------------------------
# Step 3: one bundle, each attachment by its own type
# --------------------------------------------------------------------------


def test_slices_of_one_series_become_one_imaging_study_in_instance_order():
    bundle = process_case_attachments([CaseAttachment("b", _mr_slice(2, brightness=500)), CaseAttachment("a", _mr_slice(1))])
    studies = bundle.imaging_studies()
    assert len(studies) == 1
    assert studies[0].modality is Modality.MR
    assert studies[0].metadata["instance_count"] == 2
    first_rendered = extract_frame(studies[0].images_png[0])
    expected_first = extract_frame(process_case_attachments([CaseAttachment("a", _mr_slice(1))]).imaging_studies()[0].images_png[0])
    assert first_rendered == expected_first


def extract_frame(png: bytes) -> bytes:
    return Image.open(io.BytesIO(png)).tobytes()


def test_physician_note_comes_first_then_each_attachment_under_its_own_header():
    bundle = process_case_attachments([CaseAttachment("x.pdf", _digital_pdf("Emoglobina 13.2 g/dL")), CaseAttachment("r", _sr())])
    text = bundle.combined_text("Cefalea da tre settimane.")
    assert text.startswith("Cefalea da tre settimane.")
    assert "[Allegato 1 -- PDF]\nEmoglobina 13.2 g/dL" in text
    assert "[Allegato 2 -- DICOM, SR]" in text
    assert text.index("Emoglobina") < text.index("Report Text")


def test_a_filename_never_enters_the_text():
    bundle = process_case_attachments([CaseAttachment("Rossi_Mario_emocromo.pdf", _digital_pdf("Emoglobina 13.2"))])
    assert "Rossi" not in bundle.combined_text("") and "Mario" not in bundle.combined_text("")


def test_a_declared_image_becomes_an_imaging_study_without_any_ocr_call():
    with patch.object(ClinicalDocumentProcessor, "process_document_bytes") as parse:
        bundle = process_case_attachments([CaseAttachment("torace.jpg", _jpeg(), modality="RX")])
    parse.assert_not_called()
    studies = bundle.imaging_studies()
    assert len(studies) == 1
    assert studies[0].modality is Modality.CR
    assert Image.open(io.BytesIO(studies[0].images_png[0])).format == "PNG"


@pytest.mark.parametrize("declared,expected", [("RM", Modality.MR), ("TAC", Modality.CT), ("ECOGRAFIA", Modality.US), ("MOC", Modality.DX)])
def test_italian_modality_names_are_understood(declared, expected):
    studies = process_case_attachments([CaseAttachment("x.jpg", _jpeg(), modality=declared)]).imaging_studies()
    assert studies[0].modality is expected


def test_an_undeclared_image_is_treated_as_a_document_to_read():
    bundle = process_case_attachments([CaseAttachment("foto_referto.jpg", _jpeg())])
    assert bundle.imaging_studies() == []
    assert bundle.attachments[0].reason == "image_requires_nemotron_parse_for_text"


def test_an_unrecognised_declared_modality_is_noted_not_guessed():
    bundle = process_case_attachments([CaseAttachment("x.jpg", _jpeg(), modality="XYZ")])
    assert bundle.imaging_studies() == []
    assert any("declared_modality_not_recognised" in note for note in bundle.attachments[0].notes)


def test_an_unknown_file_is_reported_and_contributes_nothing():
    bundle = process_case_attachments([CaseAttachment("x.bin", b"\x00\xff\xfe\x80binary")])
    assert bundle.attachments[0].reason == "unrecognised_binary_format"
    assert bundle.combined_text("nota") == "nota"


def test_base64_payload_items_are_decoded_and_invalid_base64_is_refused():
    item = CaseAttachment.from_payload_item({"filename": "a.pdf", "data_base64": base64.b64encode(b"%PDF-1.4").decode()})
    assert item.data == b"%PDF-1.4"
    with pytest.raises(ValueError):
        CaseAttachment.from_payload_item({"filename": "a.pdf", "data_base64": "not base64!!"})
    with pytest.raises(TypeError):
        CaseAttachment.from_payload_item({"filename": "a.pdf"})


def test_summaries_carry_no_bytes_and_no_filename():
    summary = process_case_attachments([CaseAttachment("Rossi.pdf", _digital_pdf("x"))]).summary()
    assert "Rossi" not in str(summary)
    assert all(not isinstance(value, (bytes, bytearray)) for item in summary for value in item.values())


# --------------------------------------------------------------------------
# Step 4: ClinicalIngestionPipeline
# --------------------------------------------------------------------------


def _payload(**extra):
    return {
        "case_id": "case-1",
        "report_text": "Cefalea.",
        "attachments": [
            {"filename": "a.pdf", "data": _digital_pdf("Emoglobina 13.2 g/dL")},
            {"filename": "IM1", "data": _mr_slice(1)},
        ],
        **extra,
    }


def test_prepare_payload_replaces_raw_bytes_with_combined_text_and_a_bundle():
    prepared = ClinicalIngestionPipeline().prepare_payload(_payload())
    assert "attachments" not in prepared
    assert prepared["report_text"].startswith("Cefalea.")
    assert "Emoglobina 13.2" in prepared["report_text"]
    assert ATTACHMENT_BUNDLE_KEY in prepared


def test_prepare_payload_is_idempotent_and_parses_each_file_once():
    ingestion = ClinicalIngestionPipeline()
    with patch.object(ClinicalDocumentProcessor, "process_document_bytes", wraps=ClinicalDocumentProcessor().process_document_bytes) as parse:
        prepared = ingestion.prepare_payload(_payload())
        again = ingestion.prepare_payload(prepared)
        ingestion.from_payload(again)
    assert parse.call_count == 2  # one per attachment, never repeated
    assert again["report_text"] == prepared["report_text"]


def test_from_payload_adds_the_attachment_imaging_studies_and_provenance():
    case = ClinicalIngestionPipeline().from_payload(_payload(imaging=[{"study_id": "s0", "modality": "CR", "series_paths": ["/x.png"]}]))
    assert [study.study_id for study in case.imaging] == ["s0", "attachment-series-1"]
    assert case.imaging[1].modality is Modality.MR
    assert len(case.imaging[1].images_png) == 1
    assert len(case.provenance["attachments"]) == 2


def test_a_payload_without_attachments_is_unchanged():
    case = ClinicalIngestionPipeline().from_payload({"case_id": "c", "report_text": "solo testo"})
    assert case.report_text == "solo testo"
    assert case.imaging == []


def test_from_env_reads_the_nemotron_parse_configuration(monkeypatch):
    monkeypatch.setenv("NEMOTRON_PARSE_ENDPOINT", "http://nim.example:8000")
    monkeypatch.setenv("NEMOTRON_PARSE_API_KEY", "k")
    processor = ClinicalDocumentProcessor.from_env()
    assert processor.nemotron_parse_endpoint == "http://nim.example:8000"
    assert processor.nemotron_parse_api_key == "k"


def test_from_env_with_nothing_set_leaves_the_parser_unconfigured(monkeypatch):
    monkeypatch.delenv("NEMOTRON_PARSE_ENDPOINT", raising=False)
    monkeypatch.delenv("NEMOTRON_PARSE_API_KEY", raising=False)
    assert ClinicalDocumentProcessor.from_env().nemotron_parse_endpoint is None


# --------------------------------------------------------------------------
# Step 4: through the real clinical pipeline
# --------------------------------------------------------------------------


def _non_json(obj, found=None):
    found = [] if found is None else found
    if isinstance(obj, dict):
        for value in obj.values():
            _non_json(value, found)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _non_json(value, found)
    elif not isinstance(obj, (str, int, float, bool, type(None))):
        found.append(type(obj).__name__)
    return found


def test_the_pipeline_sees_attachment_text_and_counts_in_memory_images(monkeypatch, tmp_path):
    from melampo.app import build_default_runtime

    monkeypatch.chdir(tmp_path)
    result = build_default_runtime().pipeline.run(_payload())
    assert result["volume_features"]["image_count"] == 1
    assert result["volume_features"]["has_local_images"] is True
    assert result["volume_features"]["local_features"]["local_readiness"] == "ready_in_memory"


def test_attachments_introduce_no_bytes_or_bundle_objects_into_the_result(monkeypatch, tmp_path):
    """PipelineState is a pre-existing non-JSON object in every result, with
    or without attachments -- tracked separately; attachments must add none."""
    from melampo.app import build_default_runtime

    monkeypatch.chdir(tmp_path)
    kinds = set(_non_json(build_default_runtime().pipeline.run(_payload()))) - {"PipelineState"}
    assert kinds == set()


def test_attachment_text_survives_merge_and_rerun(monkeypatch, tmp_path):
    """The reason prepare_payload() runs before pending-case routing."""
    from melampo.app import build_default_runtime

    monkeypatch.setenv("DB_PASSWORD", "test-secret")
    monkeypatch.chdir(tmp_path)
    pipeline = build_default_runtime().pipeline
    first = pipeline.run(_payload())
    pipeline._nexus_scheduler_instance().run_once(activity={"active_requests": 0, "idle_seconds": 100})

    second = pipeline.run({
        "case_id": first["pending_case_routing"]["case_id"],
        "report_text": "Controllo.",
        "attachments": [{"filename": "pcr.pdf", "data": _digital_pdf("PCR 12 mg/L")}],
    })
    merged = second["pending_case_routing"]["merged_report_text"]
    assert second["pending_case_routing"]["action"] == "merge_and_rerun"
    assert "PCR 12 mg/L" in merged
    assert "Emoglobina 13.2" in merged
