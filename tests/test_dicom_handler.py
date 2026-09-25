"""Tests for dicom_handler.py, run against the real sample DICOM files that
pydicom itself ships -- CT, MR, CR, US, a JPEG-Lossless-compressed image, a
multi-frame object and a Structured Report -- not synthetic files built to
pass. Also pins the privacy guarantee: no identifying tag ever reaches the
output.
"""

import io
import json

import pydicom
import pytest
from PIL import Image
from pydicom.data import get_testdata_file, get_testdata_files

from melampo.data.dicom_handler import MAX_RENDERED_FRAMES, extract_dicom
from melampo.data.document_processing import ClinicalDocumentProcessor

IDENTIFYING_TAGS = ("PatientName", "PatientID", "PatientBirthDate", "ReferringPhysicianName", "InstitutionName", "OperatorsName")


def _bytes(name):
    return open(get_testdata_file(name), "rb").read()


def _read_header(path):
    """A sample's header, or None -- pydicom ships some samples deliberately malformed."""
    from pydicom.errors import BytesLengthException, InvalidDicomError

    try:
        return pydicom.dcmread(path, stop_before_pixels=True)
    except (InvalidDicomError, BytesLengthException, NotImplementedError, ValueError, OSError):
        return None


def _first_sample(predicate):
    for path in get_testdata_files():
        dataset = _read_header(path)
        if dataset is not None and predicate(dataset):
            return open(path, "rb").read()
    pytest.skip("no matching pydicom sample available")


def _frames(dataset):
    try:
        return int(dataset.get("NumberOfFrames", 1) or 1)
    except (TypeError, ValueError):  # pydicom ships a sample with NumberOfFrames "1A"
        return 1


def _digital_pdf(text: str) -> bytes:
    """A minimal PDF with a real text layer, built by hand -- kept local so this file never imports another test module."""
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


def _png_ok(png_bytes):
    image = Image.open(io.BytesIO(png_bytes))
    return image.format == "PNG" and image.size[0] > 0


@pytest.mark.parametrize("name,modality", [("CT_small.dcm", "CT"), ("MR_small.dcm", "MR")])
def test_ct_and_mr_render_to_png(name, modality):
    result = extract_dicom(_bytes(name))
    assert result.status == "completed"
    assert result.modality == modality
    assert len(result.images_png) == 1
    assert _png_ok(result.images_png[0])


def test_ct_windowing_uses_the_full_8bit_range():
    """Modality + VOI LUTs applied, then stretched -- not a flat image."""
    import numpy as np

    image = np.asarray(Image.open(io.BytesIO(extract_dicom(_bytes("CT_small.dcm")).images_png[0])))
    assert image.min() == 0 and image.max() == 255
    assert len(np.unique(image)) > 100


def test_a_radiograph_renders():
    result = extract_dicom(_first_sample(lambda d: d.get("Modality") == "CR"))
    assert result.status == "completed"
    assert result.modality == "CR"
    assert _png_ok(result.images_png[0])


def test_a_colour_ultrasound_renders_as_rgb():
    result = extract_dicom(_bytes("examples_rgb_color.dcm"))
    assert result.status == "completed"
    assert Image.open(io.BytesIO(result.images_png[0])).mode == "RGB"


def test_jpeg_lossless_process_14_decodes_via_gdcm():
    """The compression most common from a PACS, undecodable without a plugin;
    python-gdcm (Apache-2.0) chosen over GPLv3 pylibjpeg-libjpeg."""
    result = extract_dicom(_first_sample(lambda d: "Process 14" in d.file_meta.TransferSyntaxUID.name))
    assert result.status == "completed"
    assert result.images_png


def test_multi_frame_is_sampled_not_fully_rendered():
    result = extract_dicom(_first_sample(lambda d: _frames(d) > MAX_RENDERED_FRAMES))
    assert len(result.images_png) == MAX_RENDERED_FRAMES
    assert result.total_frames > MAX_RENDERED_FRAMES
    assert any(note.startswith("multi_frame_sampled") for note in result.notes)


def test_a_structured_report_becomes_text():
    result = extract_dicom(_bytes("reportsi.dcm"))
    assert result.status == "completed"
    assert result.report_source == "structured_report"
    assert "Report Text" in result.report_text
    assert result.images_png == []


def test_sr_person_name_nodes_are_never_copied_into_report_text():
    """The sample's own observer name is the placeholder "Enter text", which
    legitimately appears in other TEXT nodes too -- so a distinctive name is
    set on the real PNAME node first, then the report is re-extracted."""
    dataset = pydicom.dcmread(get_testdata_file("reportsi.dcm"))
    set_names = 0

    def rename(sequence):
        nonlocal set_names
        for item in sequence:
            if item.get("ValueType") == "PNAME":
                item.PersonName = "Rossi^Mario"
                set_names += 1
            if "ContentSequence" in item:
                rename(item.ContentSequence)

    rename(dataset.ContentSequence)
    assert set_names >= 1
    buffer = io.BytesIO()
    dataset.save_as(buffer)
    text = extract_dicom(buffer.getvalue()).report_text
    assert "Rossi" not in text and "Mario" not in text
    assert "Recording Observer's Name" not in text


def test_a_malformed_frame_count_never_crashes_extraction():
    """pydicom ships badVR.dcm with NumberOfFrames "1A": reported, not raised."""
    result = extract_dicom(_bytes("badVR.dcm"))
    assert result.status == "failed"
    assert result.notes


@pytest.mark.parametrize("name", ["CT_small.dcm", "MR_small.dcm", "reportsi.dcm", "examples_rgb_color.dcm"])
def test_no_identifying_tag_value_ever_reaches_the_output(name):
    dataset = pydicom.dcmread(get_testdata_file(name), stop_before_pixels=True)
    output = json.dumps(extract_dicom(_bytes(name)).as_dict(), default=str)
    for tag in IDENTIFYING_TAGS:
        value = str(dataset.get(tag, "") or "")
        if value:
            assert value not in output, f"{tag} leaked"


def test_metadata_is_allowlisted_clinical_and_technical_only():
    metadata = extract_dicom(_bytes("CT_small.dcm")).metadata
    assert metadata["Modality"] == "CT"
    assert "SOPClassName" in metadata
    for tag in IDENTIFYING_TAGS:
        assert tag not in metadata


def test_bytes_that_are_not_dicom_fail_cleanly():
    result = extract_dicom(b"\x00" * 128 + b"DICM" + b"\x02\x00\x00\x00" * 4)
    assert result.status == "failed"
    assert result.notes


def test_an_encapsulated_pdf_report_goes_through_the_normal_pdf_path():
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.104.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.104.1"
    dataset.Modality = "DOC"
    dataset.MIMETypeOfEncapsulatedDocument = "application/pdf"
    dataset.EncapsulatedDocument = _digital_pdf("Referto RM encefalo: nessuna lesione")
    buffer = io.BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)

    result = extract_dicom(buffer.getvalue())
    assert result.report_source == "encapsulated_pdf"
    assert "nessuna lesione" in result.report_text


def test_an_encapsulated_pdf_report_is_not_duplicated_by_overlapping_chunks():
    """Regression: report_text was rebuilt by joining chunks, which overlap by
    design, so every overlap appeared twice. Tiny chunks force several overlaps
    on a one-line report; each phrase must still appear exactly once."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.104.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.104.1"
    dataset.Modality = "DOC"
    dataset.MIMETypeOfEncapsulatedDocument = "application/pdf"
    dataset.EncapsulatedDocument = _digital_pdf("Referto RM encefalo: nessuna lesione focale")
    buffer = io.BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)

    processor = ClinicalDocumentProcessor(chunk_size=20, chunk_overlap=8)
    result = extract_dicom(buffer.getvalue(), processor=processor)
    assert result.report_text.strip() == "Referto RM encefalo: nessuna lesione focale"


def test_process_document_bytes_routes_a_dicom_image_and_carries_its_images():
    result = ClinicalDocumentProcessor().process_document_bytes(_bytes("CT_small.dcm"), source_name="tac")
    assert result["document_format"] == "dicom"
    assert result["status"] == "no_text_extracted"
    assert result["reason"] == "dicom_image_only"
    assert len(result["dicom_images_png"]) == 1
    assert result["dicom"]["modality"] == "CT"


def test_process_document_bytes_turns_a_dicom_sr_into_document_text():
    result = ClinicalDocumentProcessor().process_document_bytes(_bytes("reportsi.dcm"), source_name="referto_sr")
    assert result["status"] == "completed"
    assert result["parser"] == "dicom_structured_report"
    assert "Report Text" in result["documents"][0]["text"]


def test_a_dicom_file_with_no_extension_is_still_recognised():
    """Files exported from a PACS often have no extension at all."""
    result = ClinicalDocumentProcessor().process_document_bytes(_bytes("MR_small.dcm"), source_name="IM000001")
    assert result["dicom"]["modality"] == "MR"
