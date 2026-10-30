"""The narrative reader and the imaging interpreter in the ingestion (30 October 2026): the case's reports read by
the linker, the imaging report's findings, and the exam's lesion isolated when the exam is paired with the report.
The segmenters are replaced by known masks; the DICOM series is pydicom's real CT sample (test_dicom_volume)."""

import json

import numpy as np
import pytest

from melampo.data.ingestion import ClinicalIngestionPipeline
from melampo.data.report_reading import ReportReaders, build_linker
from melampo.imaging.segmenters import MaskSegmenter, SegmenterUnavailable

REPORT = ("TC del torace con mdc. Nel lobo superiore destro nodulo solido di 12 mm a margini spiculati. "
          "Fegato nei limiti. Non versamento pleurico.")
CLAUSE = "Nel lobo superiore destro nodulo solido di 12 mm a margini spiculati"


@pytest.fixture(scope="module")
def linker():
    return build_linker()[0]


def _reading(case):
    reading = case.provenance["report_reading"]
    json.dumps(reading)                       # travels with the case: JSON-safe, no pixels
    return reading


def test_the_physician_note_is_read_sites_and_findings_on_the_text(linker):
    ingestion = ClinicalIngestionPipeline(report_readers=ReportReaders(linker))
    reading = _reading(ingestion.from_payload({"case_id": "c1", "report_text": REPORT}))
    [text] = reading["texts"]
    assert text["read"] and text["kinds"] == ["imaging_report"] and text["language"] == "it"
    sites = {s["cid"]: s for s in text["sites"]}
    assert "lung_upper_lobe_right" in sites and sites["liver"]["polarity"] == "negated"
    [finding] = reading["findings"]           # the liver "nei limiti" is a site, not a finding
    assert finding["structure"] == "lung_upper_lobe_right" and finding["text_sizes_mm"] == [12.0]
    assert "margin:spiculated" in finding["character"]
    assert finding["image_status"] == "no_image" and "no_volume_in_the_attachments" in finding["image_notes"]


def test_without_a_linker_the_case_says_so_and_reading_can_be_left_out():
    case = ClinicalIngestionPipeline(report_readers=ReportReaders(None, notes=("narrative_reader_off",))) \
        .from_payload({"case_id": "c1", "report_text": REPORT})
    assert case.provenance["report_reading"]["status"] == "narrative_reader_not_available"
    case = ClinicalIngestionPipeline(read_reports=False).from_payload({"case_id": "c1", "report_text": REPORT})
    assert "report_reading" not in case.provenance


def test_a_laboratory_text_is_not_read_by_the_narrative_reader(linker):
    reading = _reading(ClinicalIngestionPipeline(report_readers=ReportReaders(linker)).from_payload(
        {"case_id": "c1", "report_text": "Emoglobina 10.2 g/dL (13.5-17.5)\nCreatinina 1.9 mg/dL (0.7-1.2)\n"
                                         "Sodio 139 mmol/L (135-145)"}))
    assert reading["texts"] and not reading["texts"][0]["read"] and reading["findings"] == []


class _Atlas:
    """The patient's atlas: one lobe over the whole volume (a stand-in for TotalSegmentator)."""

    def __init__(self):
        self.calls = 0

    def all_structures(self, volume, affine, modality="CT"):
        self.calls += 1
        return {"lung_upper_lobe_right": np.ones(volume.shape, bool)}

    def segment(self, volume, affine, prompts):
        raise SegmenterUnavailable("not needed")


def _series_payload(note):
    from test_dicom_volume import _ct_series

    return {"case_id": "c1", "report_text": note,
            "attachments": [{"filename": f"IM{i}", "data": item} for i, item in enumerate(_ct_series(20))]}


def test_the_exam_paired_with_its_report_isolates_the_lesion(linker):
    pytest.importorskip("pydicom")
    from melampo.data.dicom_volume import assess_series
    from melampo.data.document_processing import ClinicalDocumentProcessor

    payload = _series_payload(REPORT)
    probe = ClinicalIngestionPipeline(document_processor=ClinicalDocumentProcessor(), read_reports=False)
    study = probe.from_payload(payload).imaging[0]
    assessment = assess_series(study.dicom_instances)
    shape = (assessment.slice_count, assessment.rows, assessment.columns)
    sp = assessment.pixel_spacing[0]
    z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    radius = 6.0 / sp                                            # a 12 mm nodule in the slice plane
    lesion = ((y - 64) ** 2 + (x - 64) ** 2 <= radius ** 2) & (abs(z - 10) <= 1)
    atlas = _Atlas()
    readers = ReportReaders(linker, MaskSegmenter({CLAUSE: lesion}, "test"), atlas)
    ingestion = ClinicalIngestionPipeline(document_processor=ClinicalDocumentProcessor(), report_readers=readers)
    reading = _reading(ingestion.from_payload(_series_payload(REPORT)))
    [finding] = reading["findings"]
    assert finding["image_provider"] == "given_mask" and finding["image_status"] == "agree"
    assert abs(finding["longest_axial_mm"] - 12.0) < 1.5 and finding["size_check"] == "consistent"
    assert atlas.calls == 1 and not finding["needs_review"]


def test_a_report_is_never_read_on_an_exam_it_may_not_describe():
    from melampo.data.report_reading import _pair

    pair, why = _pair([("note", REPORT, {}), ("attachment-1", REPORT, {})], ["series"])
    assert pair is None and why == "exam_not_paired:2_imaging_reports,1_volumes"
    pair, why = _pair([("note", REPORT, {})], ["series", "series"])
    assert pair is None and why == "exam_not_paired:1_imaging_reports,2_volumes"
    pair, why = _pair([("note", REPORT, {})], [])
    assert pair is None and why == "no_volume_in_the_attachments"
    assert _pair([("note", REPORT, {})], ["series"]) == (("note", "series"), "")


def test_without_segmenters_the_paired_exam_says_why_there_is_no_image(linker):
    pytest.importorskip("pydicom")
    from melampo.data.document_processing import ClinicalDocumentProcessor

    ingestion = ClinicalIngestionPipeline(document_processor=ClinicalDocumentProcessor(),
                                          report_readers=ReportReaders(linker))
    [finding] = _reading(ingestion.from_payload(_series_payload(REPORT)))["findings"]
    assert finding["image_status"] == "no_image" and "segmenters_not_configured" in finding["image_notes"]


def test_a_remote_model_is_refused_for_the_linker():
    _, notes = build_linker(chat_endpoint="https://api.example.com/v1", chat_models=("m",))
    assert any(n.startswith("models_refused") for n in notes)
