"""Tests for the measurement geometry in data/dicom_volume.py: voxel -> patient
millimetres, gantry tilt, slice thickness against spacing, frame of reference,
declared measurement precision and acquisition comparability.

Series are synthetic -- pydicom's real CT sample (CT_small.dcm) cloned slice by
slice with positions, orientation, spacing and tags set per test -- so each
geometric condition is produced exactly and its effect checked against numbers
computed by hand, not against the code under test.
"""

import copy
import io
import math
import random

import numpy as np
import pydicom
import pytest
from pydicom.data import get_testdata_file
from pydicom.uid import generate_uid

from melampo.data.dicom_volume import (
    PRECISION_HIGH,
    PRECISION_LOW,
    PRECISION_NONE,
    PRECISION_REDUCED,
    assess_series,
    compare_acquisitions,
    deidentify_instance,
    index_to_patient_mm,
    load_volume_hu,
    measurement_precision,
    share_frame_of_reference,
)

ORIGIN = (-158.0, -179.0, -75.0)


def _write(dataset) -> bytes:
    buffer = io.BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def _series(positions, *, orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0), pixel_spacing=(0.661468, 0.661468),
            thickness=5.0, thicknesses=None, spacing_tag=None, frame_uid="1.2.826.0.1.3680043.8.498.1",
            frame_uids=None, kernel="STANDARD", model="LightSpeed", modality=None, seed=3):
    """One de-identified slice per position (3-vectors), shuffled; each slice's pixels offset by its index."""
    base = pydicom.dcmread(get_testdata_file("CT_small.dcm"))
    series_uid = generate_uid()
    slices = []
    for index, position in enumerate(positions):
        dataset = copy.deepcopy(base)
        dataset.SOPInstanceUID = generate_uid()
        dataset.SeriesInstanceUID = series_uid
        dataset.InstanceNumber = index + 1
        dataset.ImagePositionPatient = list(position)
        dataset.ImageOrientationPatient = list(orientation)
        dataset.PixelSpacing = list(pixel_spacing)
        dataset.SliceThickness = thicknesses[index] if thicknesses else thickness
        if spacing_tag is not None:
            dataset.SpacingBetweenSlices = spacing_tag
        uid = frame_uids[index] if frame_uids else frame_uid
        if uid is None:
            if "FrameOfReferenceUID" in dataset:
                del dataset.FrameOfReferenceUID
        else:
            dataset.FrameOfReferenceUID = uid
        dataset.ConvolutionKernel = kernel
        dataset.ManufacturerModelName = model
        if modality:
            dataset.Modality = modality
        dataset.PixelData = (base.pixel_array.copy() + index).tobytes()
        slices.append(deidentify_instance(_write(dataset)).data)
    random.Random(seed).shuffle(slices)
    return slices


def _axial(count=20, spacing=5.0, **options):
    return _series([(ORIGIN[0], ORIGIN[1], ORIGIN[2] + spacing * k) for k in range(count)], **options)


# ---------------------------------------------------------------------------
# Voxel -> patient millimetres
# ---------------------------------------------------------------------------


def test_the_affine_maps_voxels_to_patient_millimetres_with_row_spacing_first():
    # Non-square pixels: rows 0.5 mm apart, columns 0.8 mm apart. Swapping the
    # two PixelSpacing values is the classic silent error this must catch.
    assessment = assess_series(_axial(pixel_spacing=(0.5, 0.8)))
    assert assessment.usable
    assert index_to_patient_mm(assessment, 0, 0, 0) == pytest.approx(ORIGIN)
    # One column step moves along the row direction (x) by the column spacing.
    assert index_to_patient_mm(assessment, 0, 0, 10) == pytest.approx((ORIGIN[0] + 8.0, ORIGIN[1], ORIGIN[2]))
    # One row step moves along the column direction (y) by the row spacing.
    assert index_to_patient_mm(assessment, 0, 10, 0) == pytest.approx((ORIGIN[0], ORIGIN[1] + 5.0, ORIGIN[2]))
    assert index_to_patient_mm(assessment, 19, 0, 0) == pytest.approx((ORIGIN[0], ORIGIN[1], ORIGIN[2] + 95.0))


def test_fractional_indices_give_sub_voxel_positions():
    assessment = assess_series(_axial(pixel_spacing=(0.5, 0.8)))
    assert index_to_patient_mm(assessment, 2.5, 1.5, 0.25) == pytest.approx((ORIGIN[0] + 0.2, ORIGIN[1] + 0.75, ORIGIN[2] + 12.5))


def test_the_affine_matches_the_array_load_volume_hu_returns():
    instances = _axial()
    assessment = assess_series(instances)
    volume = load_volume_hu(instances, assessment)
    base = pydicom.dcmread(get_testdata_file("CT_small.dcm")).pixel_array.astype(np.int32)
    # Slice k of the array is the slice whose pixels were offset by k, and the affine puts it at z0 + 5k.
    for k in (0, 11, 19):
        assert np.array_equal(volume[k], base + k - 1024)
        assert index_to_patient_mm(assessment, k, 0, 0)[2] == pytest.approx(ORIGIN[2] + 5.0 * k)


def test_an_oblique_orientation_is_carried_by_the_affine():
    angle = math.radians(30)
    row = (math.cos(angle), math.sin(angle), 0.0)
    column = (-math.sin(angle), math.cos(angle), 0.0)
    assessment = assess_series(_axial(orientation=(*row, *column), pixel_spacing=(1.0, 1.0)))
    x, y, z = index_to_patient_mm(assessment, 0, 0, 10)
    assert (x, y, z) == pytest.approx((ORIGIN[0] + 10 * row[0], ORIGIN[1] + 10 * row[1], ORIGIN[2]))


@pytest.mark.parametrize(
    "positions",
    [
        [(ORIGIN[0], ORIGIN[1], ORIGIN[2] + z) for z in [5.0 * i for i in range(10)] + [5.0 * i for i in range(11, 21)]],
        [(ORIGIN[0] + (2.0 if k == 7 else 0.0), ORIGIN[1], ORIGIN[2] + 5.0 * k) for k in range(20)],
    ],
    ids=["uneven_spacing", "off_line_slice"],
)
def test_an_irregular_grid_gets_no_affine(positions):
    # One step vector cannot describe these slices: an affine would misplace the inner ones.
    assessment = assess_series(_series(positions))
    assert assessment.affine is None
    with pytest.raises(ValueError, match="no patient geometry"):
        index_to_patient_mm(assessment, 3, 0, 0)


def test_a_reversed_column_direction_keeps_array_and_affine_on_the_same_slice():
    # IOP with the column direction pointing to -y gives a slice normal along -z:
    # slices are then ordered from the highest z down, and slice 0 must be that one.
    assessment = assess_series(_axial(orientation=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)))
    assert assessment.usable
    assert index_to_patient_mm(assessment, 0, 0, 0)[2] == pytest.approx(ORIGIN[2] + 95.0)
    assert index_to_patient_mm(assessment, 0, 10, 0)[1] == pytest.approx(ORIGIN[1] - 6.61468)
    assert index_to_patient_mm(assessment, 19, 0, 0)[2] == pytest.approx(ORIGIN[2])


def test_a_multi_valued_kernel_is_read_whole():
    assessment = assess_series(_axial(kernel=["FC13", "AIDR 3D"]))
    assert assessment.convolution_kernel == "FC13\\AIDR 3D"


def test_a_kernel_changing_inside_a_series_is_reported():
    # One series whose second half was reconstructed with another kernel.
    positions = [(ORIGIN[0], ORIGIN[1], ORIGIN[2] + 5.0 * k) for k in range(20)]
    first = _series(positions[:10], kernel="STANDARD", seed=1)
    base_uid = pydicom.dcmread(io.BytesIO(first[0])).SeriesInstanceUID
    second = []
    for item in _series(positions[10:], kernel="BONE", seed=2):
        dataset = pydicom.dcmread(io.BytesIO(item))
        dataset.SeriesInstanceUID = base_uid
        second.append(_write(dataset))
    assert "inconsistent_convolution_kernel" in assess_series(first + second).notes


def test_no_geometry_means_no_measurement():
    assessment = assess_series(_axial(count=20, orientation=(1, 0, 0, 0, 1, 0))[:1])
    with pytest.raises(ValueError, match="no patient geometry"):
        index_to_patient_mm(assessment, 0, 0, 0)


# ---------------------------------------------------------------------------
# Gantry tilt, collinearity, thickness against spacing
# ---------------------------------------------------------------------------


def test_a_tilted_gantry_is_measured_from_positions_and_kept_in_the_affine():
    tilt = 15.0
    shift = 5.0 * math.tan(math.radians(tilt))  # y drift per slice for 5 mm along the normal
    positions = [(ORIGIN[0], ORIGIN[1] + shift * k, ORIGIN[2] + 5.0 * k) for k in range(20)]
    assessment = assess_series(_series(positions))
    assert assessment.usable, assessment.problems
    assert "sheared_volume_gantry_tilt" in assessment.notes
    assert assessment.tilt_degrees == pytest.approx(tilt, abs=1e-3)
    assert assessment.slice_spacing == pytest.approx(5.0)  # distance between planes, along the normal
    # The last slice lands exactly where the scanner put it, drift included.
    assert index_to_patient_mm(assessment, 19, 0, 0) == pytest.approx(positions[19])
    assert "sheared_volume_measure_through_affine_only" in measurement_precision(assessment).reasons


def test_an_untilted_series_has_zero_tilt_and_no_note():
    assessment = assess_series(_axial())
    assert assessment.tilt_degrees == pytest.approx(0.0, abs=1e-6)
    assert "sheared_volume_gantry_tilt" not in assessment.notes


def test_a_slice_displaced_sideways_is_refused():
    positions = [(ORIGIN[0], ORIGIN[1], ORIGIN[2] + 5.0 * k) for k in range(20)]
    positions[7] = (ORIGIN[0] + 2.0, ORIGIN[1], positions[7][2])
    assessment = assess_series(_series(positions))
    assert not assessment.usable
    assert "slice_positions_not_collinear" in assessment.problems


@pytest.mark.parametrize(
    ("thickness", "spacing", "note"),
    [
        (1.0, 5.0, "gaps_between_slices"),
        (2.5, 1.25, "overlapping_slices"),
    ],
)
def test_thickness_is_compared_with_the_measured_spacing(thickness, spacing, note):
    assessment = assess_series(_axial(spacing=spacing, thickness=thickness))
    assert assessment.usable
    assert note in assessment.notes


def test_contiguous_slices_have_neither_gaps_nor_overlap():
    notes = assess_series(_axial(spacing=1.25, thickness=1.25)).notes
    assert "gaps_between_slices" not in notes and "overlapping_slices" not in notes


def test_mixed_slice_thickness_and_a_wrong_spacing_tag_are_reported():
    thicknesses = [1.25] * 10 + [2.5] * 10
    assessment = assess_series(_axial(spacing=1.25, thicknesses=thicknesses, spacing_tag=2.0))
    assert "inconsistent_slice_thickness" in assessment.notes
    assert "spacing_between_slices_tag_disagrees" in assessment.notes


# ---------------------------------------------------------------------------
# Frame of reference
# ---------------------------------------------------------------------------


def test_a_series_mixing_frames_of_reference_is_refused():
    uids = ["1.2.3.1"] * 10 + ["1.2.3.2"] * 10
    assessment = assess_series(_axial(frame_uids=uids))
    assert not assessment.usable
    assert "mixed_frame_of_reference" in assessment.problems


def test_series_are_spatially_related_only_when_they_share_a_known_frame():
    first = assess_series(_axial(frame_uid="1.2.3.9"))
    same = assess_series(_axial(frame_uid="1.2.3.9", seed=5))
    other = assess_series(_axial(frame_uid="1.2.3.10"))
    unknown = assess_series(_axial(frame_uid=None))
    assert share_frame_of_reference(first, same)
    assert not share_frame_of_reference(first, other)
    assert unknown.frame_of_reference_uid is None
    assert not share_frame_of_reference(unknown, unknown)


# ---------------------------------------------------------------------------
# Declared precision and comparability of two exams
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("thickness", "spacing", "level"),
    [
        (1.0, 1.0, PRECISION_HIGH),
        (1.25, 1.25, PRECISION_HIGH),
        (2.5, 2.5, PRECISION_REDUCED),
        (5.0, 5.0, PRECISION_LOW),
        (1.0, 2.0, PRECISION_LOW),  # thin slices with gaps: anatomy between them is never sampled
    ],
)
def test_the_declared_precision_follows_the_acquisition(thickness, spacing, level):
    assert measurement_precision(assess_series(_axial(spacing=spacing, thickness=thickness))).level == level


def test_an_unusable_series_supports_no_measurement():
    precision = measurement_precision(assess_series(_axial(count=5)))
    assert precision.level == PRECISION_NONE
    assert "too_few_slices" in precision.reasons


def test_mr_is_flagged_for_non_absolute_intensities():
    precision = measurement_precision(assess_series(_axial(spacing=1.0, thickness=1.0, modality="MR")))
    assert precision.level == PRECISION_HIGH
    assert "mr_intensities_not_absolute" in precision.reasons


def test_two_identical_acquisitions_are_comparable():
    comparison = compare_acquisitions(assess_series(_axial()), assess_series(_axial(seed=9)))
    assert comparison.qiba_comparable and comparison.differences == ()


def test_a_different_kernel_or_thickness_breaks_comparability_and_says_which():
    earlier = assess_series(_axial(thickness=1.25, spacing=1.25, kernel="STANDARD"))
    later = assess_series(_axial(thickness=2.5, spacing=2.5, kernel="BONE"))
    comparison = compare_acquisitions(earlier, later)
    assert not comparison.qiba_comparable
    assert "different_slice_thickness:1.25->2.5" in comparison.differences
    assert "different_convolution_kernel:STANDARD->BONE" in comparison.differences


def test_an_unknown_scanner_model_is_not_assumed_comparable():
    earlier = assess_series(_axial(model=""))
    comparison = compare_acquisitions(earlier, assess_series(_axial()))
    assert not comparison.qiba_comparable
    assert "manufacturer_model_unknown" in comparison.differences


def test_the_assessment_serialises_its_geometry():
    data = assess_series(_axial(pixel_spacing=(0.5, 0.8))).as_dict()
    assert data["affine"][0][0] == pytest.approx(0.8) and data["affine"][1][1] == pytest.approx(0.5)
    assert data["frame_of_reference_uid"] == "1.2.826.0.1.3680043.8.498.1"
    assert data["convolution_kernel"] == "STANDARD" and data["manufacturer_model"] == "LightSpeed"
