"""Tests for data/dicom_volume.py -- step 5 of the document-processing plan:
a DICOM series kept whole and de-identified, assessed as a volume, and
checked against the released Pillar-0 checkpoints.

Series are built from pydicom's own real CT sample (CT_small.dcm: rescale
intercept -1024, a patient name, 179 private tags), cloned slice by slice
with distinct positions and distinct pixels, so ordering, Hounsfield
conversion and de-identification are all checked on real data, not on
hand-written headers alone.
"""

import copy
import io
import json
import random
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.data import get_testdata_file, get_testdata_files
from pydicom.uid import generate_uid

from melampo.data.case_attachments import CaseAttachment, process_case_attachments
from melampo.data.dicom_volume import (
    CHECKPOINT_ABDOMEN_CT,
    CHECKPOINT_BREAST_MRI,
    CHECKPOINT_CHEST_CT,
    CHECKPOINT_HEAD_CT,
    INSTANCE_ALLOWLIST,
    MIN_SLICES_FOR_VOLUME,
    assess_series,
    deidentify_instance,
    export_series_directory,
    load_volume_hu,
    pillar0_eligibility,
    preferred_series,
)
from melampo.data.document_processing import ClinicalDocumentProcessor
from melampo.data.ingestion import ClinicalIngestionPipeline

IDENTIFYING = ("PatientName", "PatientID", "PatientBirthDate", "PatientSex", "InstitutionName",
               "ReferringPhysicianName", "OperatorsName", "AccessionNumber", "StudyDate", "SeriesDate",
               "AcquisitionDate", "ContentDate", "StudyTime")


def _write(dataset) -> bytes:
    buffer = io.BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def _ct_series(count=20, *, spacing=5.0, body_part="CHEST", description="", orientation=None,
               positions=None, sizes=None, seed=1, series_uid=None, thickness=5.0, burned_in=False, vary="z"):
    """Real CT slices, one per position, pixels offset by slice index, returned shuffled."""
    base = pydicom.dcmread(get_testdata_file("CT_small.dcm"))
    base.PatientName = "Rossi^Mario"
    base.PatientID = "RSSMRA80A01H501U"
    base.InstitutionName = "Ospedale Esempio"
    series_uid = series_uid or generate_uid()
    positions = positions if positions is not None else [-75.7 + spacing * index for index in range(count)]
    slices = []
    for index, z in enumerate(positions):
        # deepcopy, not Dataset.copy(): a shallow copy shares its elements
        # with `base`, so each slice's PixelData would leak into the next.
        dataset = copy.deepcopy(base)
        dataset.SOPInstanceUID = generate_uid()
        dataset.SeriesInstanceUID = series_uid
        dataset.InstanceNumber = 1000 - index  # deliberately the reverse of spatial order
        # Positions step along the slice normal: z for axial, y for coronal.
        dataset.ImagePositionPatient = [-158.135803, z, -75.7] if vary == "y" else [-158.135803, -179.035797, z]
        if orientation is not None:
            dataset.ImageOrientationPatient = orientation
        if body_part is not None:
            dataset.BodyPartExamined = body_part
        elif "BodyPartExamined" in dataset:
            del dataset.BodyPartExamined
        dataset.StudyDescription = description
        dataset.SliceThickness = thickness
        if burned_in:
            dataset.BurnedInAnnotation = "YES"
        pixels = base.pixel_array.copy() + index
        if sizes and index in sizes:
            pixels = np.zeros(sizes[index], dtype=pixels.dtype)
            dataset.Rows, dataset.Columns = sizes[index]
        dataset.PixelData = pixels.tobytes()
        slices.append(_write(dataset))
    random.Random(seed).shuffle(slices)
    return slices


def _deid(raw):
    return [deidentify_instance(item).data for item in raw]


# ---------------------------------------------------------------------------
# De-identification by allowlist
# ---------------------------------------------------------------------------


def test_identity_dates_and_private_tags_are_gone():
    original = _ct_series(1)[0]
    before = pydicom.dcmread(io.BytesIO(original))
    assert "PatientName" in before and any(element.tag.is_private for element in before)
    after = pydicom.dcmread(io.BytesIO(deidentify_instance(original).data))
    assert [keyword for keyword in IDENTIFYING if keyword in after] == []
    assert not any(element.tag.is_private for element in after)
    assert after.PatientIdentityRemoved == "YES"


def test_only_allowlisted_attributes_survive():
    after = pydicom.dcmread(io.BytesIO(deidentify_instance(_ct_series(1)[0]).data))
    kept = {element.keyword for element in after}
    assert kept <= set(INSTANCE_ALLOWLIST) | {"PatientIdentityRemoved", "DeidentificationMethod"}


@pytest.mark.parametrize("sample", ["CT_small.dcm", "MR_small_jpeg_ls_lossless.dcm", "MR_small_RLE.dcm"])
def test_pixel_data_and_transfer_syntax_are_kept_byte_for_byte(sample):
    raw = Path(get_testdata_file(sample)).read_bytes()
    before = pydicom.dcmread(io.BytesIO(raw))
    after = pydicom.dcmread(io.BytesIO(deidentify_instance(raw).data))
    assert after.PixelData == before.PixelData
    assert after.file_meta.TransferSyntaxUID == before.file_meta.TransferSyntaxUID


def test_a_declared_burned_in_annotation_is_reported():
    assert deidentify_instance(_ct_series(1, burned_in=True)[0]).notes == ("burned_in_annotation_declared",)


def test_bytes_that_are_not_dicom_give_nothing():
    assert deidentify_instance(b"%PDF-1.4 not dicom") is None


# ---------------------------------------------------------------------------
# Volume assessment and Hounsfield units
# ---------------------------------------------------------------------------


def test_slices_are_ordered_by_position_not_instance_number_and_read_in_hounsfield_units():
    instances = _deid(_ct_series(20))
    assessment = assess_series(instances)
    assert assessment.usable and assessment.problems == ()
    assert (assessment.slice_count, assessment.slice_spacing, assessment.orientation, assessment.anatomy) == (20, 5.0, "axial", "chest")
    volume = load_volume_hu(instances, assessment)
    base = pydicom.dcmread(get_testdata_file("CT_small.dcm")).pixel_array.astype(np.int32)
    assert volume.shape == (20, 128, 128) and volume.dtype == np.int16
    # Slice k (from the lowest position up) has pixels offset by k; stored -> HU is +(-1024).
    for k in (0, 7, 19):
        assert np.array_equal(volume[k], base + k - 1024)


@pytest.mark.parametrize(
    ("options", "problem"),
    [
        ({"positions": [0.0, 5.0, 5.0] + [10.0 + 5.0 * i for i in range(17)]}, "duplicate_slice_positions"),
        ({"positions": [5.0 * i for i in range(10)] + [5.0 * i for i in range(11, 21)]}, "uneven_slice_spacing"),
        ({"count": MIN_SLICES_FOR_VOLUME - 1}, "too_few_slices"),
        ({"sizes": {3: (64, 64)}}, "inconsistent_image_size"),
    ],
)
def test_a_series_that_is_not_a_clean_volume_says_why(options, problem):
    assessment = assess_series(_deid(_ct_series(**options)))
    assert not assessment.usable
    assert problem in assessment.problems


def test_mixed_orientations_are_refused():
    axial = _ct_series(10, series_uid="1.2.3.4")
    coronal = _ct_series(10, series_uid="1.2.3.4", orientation=[1, 0, 0, 0, 0, -1], positions=[100.0 + 5 * i for i in range(10)], vary="y")
    assert "inconsistent_orientation" in assess_series(_deid(axial + coronal)).problems


def test_an_unusable_series_is_never_stacked_anyway():
    instances = _deid(_ct_series(5))
    with pytest.raises(ValueError, match="too_few_slices"):
        load_volume_hu(instances, assess_series(instances))


# ---------------------------------------------------------------------------
# Pillar-0 eligibility -- the released checkpoints only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body_part", "description", "expected"),
    [
        ("CHEST", "", CHECKPOINT_CHEST_CT),
        (None, "TC TORACE SENZA MDC", CHECKPOINT_CHEST_CT),
        ("ABDOMEN", "", CHECKPOINT_ABDOMEN_CT),
        (None, "TC ADDOME COMPLETO", CHECKPOINT_ABDOMEN_CT),
        ("HEAD", "", CHECKPOINT_HEAD_CT),
        (None, "TC ENCEFALO", CHECKPOINT_HEAD_CT),
    ],
)
def test_covered_ct_anatomies_map_to_their_checkpoint(body_part, description, expected):
    eligibility = pillar0_eligibility(assess_series(_deid(_ct_series(20, body_part=body_part, description=description))))
    assert (eligibility.eligible, eligibility.checkpoint) == (True, expected)


@pytest.mark.parametrize(
    ("body_part", "description", "reason"),
    [
        (None, "", "anatomy_not_stated_in_dicom"),
        ("KNEE", "", "anatomy_not_stated_in_dicom"),
        (None, "TC TORACE ADDOME", "anatomy_ambiguous:abdomen+chest"),
    ],
)
def test_ct_without_a_single_covered_anatomy_is_not_sent(body_part, description, reason):
    eligibility = pillar0_eligibility(assess_series(_deid(_ct_series(20, body_part=body_part, description=description))))
    assert (eligibility.eligible, eligibility.reason) == (False, reason)


def test_a_non_axial_ct_is_not_sent():
    coronal = _ct_series(20, orientation=[1, 0, 0, 0, 0, -1], vary="y")
    assert assess_series(_deid(coronal)).usable
    assert pillar0_eligibility(assess_series(_deid(coronal))).reason == "ct_series_not_axial"


def test_other_modalities_have_no_pillar0_model():
    from melampo.data.dicom_volume import VolumeAssessment

    for modality in ("CR", "DX", "US", "MG"):
        assessment = VolumeAssessment(usable=False, problems=("too_few_slices",), modality=modality, slice_count=1)
        assert pillar0_eligibility(assessment).reason == f"modality_not_covered_by_pillar0:{modality}"


def test_breast_mri_is_eligible_other_mri_is_not():
    from melampo.data.dicom_volume import VolumeAssessment

    breast = VolumeAssessment(usable=True, problems=(), modality="MR", slice_count=120, orientation="axial", anatomy="breast")
    brain = VolumeAssessment(usable=True, problems=(), modality="MR", slice_count=120, orientation="axial", anatomy="head")
    assert pillar0_eligibility(breast).checkpoint == CHECKPOINT_BREAST_MRI
    assert pillar0_eligibility(brain).reason == "anatomy_not_covered_by_pillar0:mr_head"


def test_the_thinnest_eligible_series_is_the_one_sent():
    thick = assess_series(_deid(_ct_series(20, thickness=5.0)))
    thin = assess_series(_deid(_ct_series(20, thickness=1.25)))
    choice = preferred_series([("thick", thick, pillar0_eligibility(thick)), ("thin", thin, pillar0_eligibility(thin))])
    assert choice == {CHECKPOINT_CHEST_CT: "thin"}


# ---------------------------------------------------------------------------
# Through the case attachments
# ---------------------------------------------------------------------------


def _structured_report() -> bytes:
    from pydicom.errors import InvalidDicomError

    for path in get_testdata_files():
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
        except (InvalidDicomError, ValueError, OSError, NotImplementedError):  # pydicom ships deliberately malformed samples
            dataset = None
        if dataset is not None and dataset.get("Modality") == "SR":
            return Path(path).read_bytes()
    pytest.skip("no SR sample available")


def test_a_ct_series_upload_becomes_one_study_with_its_whole_deidentified_series():
    raw = _ct_series(20)
    attachments = [CaseAttachment(filename=f"IM{index}", data=item) for index, item in enumerate(raw)]
    attachments.append(CaseAttachment(filename="report.dcm", data=_structured_report()))
    bundle = process_case_attachments(attachments, processor=ClinicalDocumentProcessor())
    [study] = bundle.imaging_studies()  # the structured report is not an imaging study
    assert len(study.dicom_instances) == 20
    assert study.metadata["volume"]["usable"] is True
    assert study.metadata["pillar0"] == {"eligible": True, "checkpoint": CHECKPOINT_CHEST_CT, "reason": None, "preferred_for_checkpoint": True}
    assert all("Rossi" not in pydicom.dcmread(io.BytesIO(item)).get("PatientName", "") for item in study.dicom_instances)


def test_case_provenance_carries_the_assessment_as_json_without_pixels_or_identity():
    pipeline = ClinicalIngestionPipeline(document_processor=ClinicalDocumentProcessor())
    case = pipeline.from_payload({
        "case_id": "c1",
        "attachments": [{"filename": f"IM{index}", "data": item} for index, item in enumerate(_ct_series(20))],
    })
    serialised = json.dumps(case.provenance)
    assert "Rossi" not in serialised and "RSSMRA" not in serialised
    [volume] = case.provenance["imaging_volumes"]
    assert (volume["instance_count"], volume["pillar0"]["checkpoint"]) == (20, CHECKPOINT_CHEST_CT)
    assert len(case.imaging[0].dicom_instances) == 20


def test_export_writes_the_series_directory_rave_expects_in_slice_order(tmp_path):
    instances = _deid(_ct_series(20))
    assessment = assess_series(instances)
    written = export_series_directory(instances, assessment, tmp_path / "series")
    positions = [float(pydicom.dcmread(path).ImagePositionPatient[2]) for path in written]
    assert positions == sorted(positions) and len(written) == 20
    assert all("PatientName" not in pydicom.dcmread(path) for path in written)
