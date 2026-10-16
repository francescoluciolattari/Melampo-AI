"""The imaging probe end to end on a synthetic chest CT, with the segmenters replaced by given masks
(16 October 2026). Needs nibabel (the imaging extra)."""

import json

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

import imaging_probe as probe  # noqa: E402
from melampo.imaging.segmenters import MaskSegmenter  # noqa: E402

AFFINE_RAS = np.diag([0.8, 0.8, 2.5, 1.0])
SHAPE_XYZ = (96, 96, 40)


def _ball(c, r):
    x, y, z = np.mgrid[0:SHAPE_XYZ[0], 0:SHAPE_XYZ[1], 0:SHAPE_XYZ[2]]
    return ((x - c[0]) * 0.8) ** 2 + ((y - c[1]) * 0.8) ** 2 + ((z - c[2]) * 2.5) ** 2 <= r ** 2


def test_probe_reports_the_far_speck_case_and_the_review_reasons(tmp_path, monkeypatch):
    lesion = _ball((48, 48, 20), 10)
    ct = np.full(SHAPE_XYZ, -850.0)
    ct[lesion] = 35.0
    nib.save(nib.Nifti1Image(ct, AFFINE_RAS), str(tmp_path / "img.nii.gz"))
    nib.save(nib.Nifti1Image(lesion.astype(np.uint8), AFFINE_RAS), str(tmp_path / "ref.nii.gz"))
    zyx = lambda m: m.transpose(2, 1, 0)  # noqa: E731
    found = lesion.copy()
    found[90, 90, 20] = True                                    # the lung_006 pattern: a speck far away
    lung = _ball((48, 48, 20), 40)

    class Lesions(MaskSegmenter):
        def __init__(self, **_):
            super().__init__({"tumore polmonare": zyx(found)}, "fake")

    class Organs(MaskSegmenter):
        def __init__(self, **_):
            super().__init__({"lung_upper_lobe_right": zyx(lung)}, "fake")

    monkeypatch.setattr(probe, "VoxTellSegmenter", Lesions)
    monkeypatch.setattr(probe, "TotalSegmentatorSegmenter", Organs)
    cases = [{"id": "syn", "image": str(tmp_path / "img.nii.gz"), "lesion": str(tmp_path / "ref.nii.gz"),
              "structure": ["lung_upper_lobe_right"], "prompt": "tumore polmonare"}]
    row = probe.run(cases, "cpu", True)[0]
    got = row["prompt"]
    assert row["report_is_synthetic"] and "cm" in row["report_clause"]
    assert abs(got["measured_longest_axial_mm"] - row["reference_longest_axial_mm"]) < 1.5
    assert got["whole_region_longest_axial_mm"] > 40 and got["parts"] == 2
    assert got["second_reading"]["agree"] and got["dice_second_reading"] > 0.9
    assert got["status"] == "agree" and not got["needs_review"]
    assert got["measure_support"] == ["report", "second_reading"] and got["density"]["composition"] == "solid"
    assert row["reference_density"]["units"] == "HU" and "solida" not in row["report_clause"]
    json.dumps(row, default=float)


def test_a_failing_case_is_recorded_and_the_others_go_on(tmp_path, monkeypatch):
    class Broken(MaskSegmenter):
        def __init__(self, **_):
            super().__init__({})

    monkeypatch.setattr(probe, "VoxTellSegmenter", Broken)
    monkeypatch.setattr(probe, "TotalSegmentatorSegmenter", Broken)
    out = tmp_path / "probe.json"
    rows = probe.run([{"id": "missing", "image": str(tmp_path / "nope.nii.gz"), "lesion": "x", "structure": "lung"}],
                     "cpu", True, out)
    assert "error" in rows[0] and json.loads(out.read_text())[0]["id"] == "missing"
