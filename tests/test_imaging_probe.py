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


def test_size_diagnostics_say_where_the_missed_voxels_are(tmp_path, monkeypatch):
    pytest.importorskip("scipy")
    from melampo.imaging.segmenters import Segmentation

    zyx = lambda m: m.transpose(2, 1, 0)  # noqa: E731
    ref = zyx(_ball((48, 48, 20), 12))
    found = zyx(_ball((48, 48, 20), 10.5))                       # the border placed 1.5 mm inside
    ct = np.where(ref, 30.0, -850.0)
    affine = np.array([[0.8, 0, 0, 0], [0, 0.8, 0, 0], [0, 0, 2.5, 0], [0, 0, 0, 1.0]])

    class Fake:
        def probabilities(self, volume, aff, prompts):
            return [np.where(ref, 0.4, 0.0) + np.where(found, 0.5, 0.0)]

        def segment(self, volume, aff, prompts):
            return [Segmentation(np.ones(volume.shape, bool)[:0] if False else None, "fake")]

    d = probe.size_diagnostics(ct, affine, ref, found, found, Fake(), "tumore polmonare")
    assert d["volume_ratio"] < 0.8 and d["missed_within_2mm_share"] > 0.75 and d["missed_median_hu"] == 30.0
    assert d["cut_0.35"]["dice"] > 0.95 > d["cut_0.5"]["dice"]
    assert d["resampled_1_5mm"]["dice"] == 0.0 and "nninteractive" not in d


def test_shape_diagnostics_find_a_missing_tail():
    pytest.importorskip("scipy")
    zyx = lambda m: m.transpose(2, 1, 0)  # noqa: E731
    body = _ball((60, 40, 20), 9)
    tail = np.zeros_like(body)
    tail[30:56, 18:23, 8:13] = True                              # a thin tail along x the region lacks
    ref, found = zyx(body | tail), zyx(body)
    ct = np.where(ref, 30.0, -850.0)
    affine = np.array([[0.8, 0, 0, 0], [0, 0.8, 0, 0], [0, 0, 2.5, 0], [0, 0, 0, 1.0]])
    s = probe._shape_diagnostics(found, ref, ct, affine)
    assert s["extent_reference_mm"][0] > s["extent_region_mm"][0] + 5
    assert s["missed_at_ends_share"] > 0.7 and s["missed_blobs"] == 1 and s["missed_largest_blob_share"] == 1.0


def test_isolated_run_turns_a_dead_process_into_a_row(tmp_path):
    import argparse

    args = argparse.Namespace(device="cpu", fast_organs=False, diagnose_size=False, nninteractive=False)
    rows = probe.run_isolated([{"id": "x", "image": str(tmp_path / "none.nii.gz")}], args, tmp_path / "out.json")
    assert rows[0]["id"] == "x" and "error" in rows[0]
    assert (tmp_path / "out.json").exists()


def test_anatomy_records_the_zones_the_position_and_the_margins(tmp_path, monkeypatch):
    lesion = _ball((48, 48, 20), 10)
    ct = np.full(SHAPE_XYZ, -850.0)
    ct[lesion] = 35.0
    nib.save(nib.Nifti1Image(ct, AFFINE_RAS), str(tmp_path / "img.nii.gz"))
    nib.save(nib.Nifti1Image(lesion.astype(np.uint8), AFFINE_RAS), str(tmp_path / "ref.nii.gz"))
    zyx = lambda m: m.transpose(2, 1, 0)  # noqa: E731
    lobe = _ball((48, 48, 20), 40) & ~lesion

    class Lesions(MaskSegmenter):
        def __init__(self, **_):
            super().__init__({"tumore polmonare": zyx(lesion)}, "fake")

    class Organs(MaskSegmenter):
        def __init__(self, **_):
            super().__init__({}, "fake")

        def all_structures(self, volume, affine):
            return {"lung_upper_lobe_right": zyx(lobe)}

    monkeypatch.setattr(probe, "VoxTellSegmenter", Lesions)
    monkeypatch.setattr(probe, "TotalSegmentatorSegmenter", Organs)
    cases = [{"id": "syn", "image": str(tmp_path / "img.nii.gz"), "lesion": str(tmp_path / "ref.nii.gz"),
              "structure": ["lung_upper_lobe_right"], "prompt": "tumore polmonare"}]
    row = probe.run(cases, "cpu", True, anatomy=True)[0]
    got = row["prompt"]["anatomy"]
    assert got["position"]["host"] == "lung_upper_lobe_right" and got["position"]["zone"] == "central"
    assert got["involvement"][0]["structure"] == "lung_upper_lobe_right"
    assert got["margins"]["sharpness"] == "sharp" and row["reference_anatomy"]["position"] == got["position"]
    rad = row["prompt"]["radiologist"]
    assert rad["window"]["window"] == "lung" and rad["window"]["dice"] > 0.9
    assert abs(rad["window"]["axial_long_mm"] - rad["reference"]["axial_long_mm"]) < 2.0
    assert rad["segmenter"]["axial_short_mm"] is not None and "mask" not in rad["window"]
    json.dumps(row, default=float)


def test_overlay_png_draws_both_contours_where_the_measure_is_off(tmp_path, monkeypatch):
    import struct
    import zlib

    lesion = _ball((48, 48, 20), 10)
    found = _ball((48, 48, 20), 6)                                 # a region well short of the reference
    ct = np.full(SHAPE_XYZ, -850.0)
    ct[lesion] = 35.0
    zyx = lambda m: m.transpose(2, 1, 0)  # noqa: E731
    affine = np.diag([-0.8, -0.8, 2.5, 1.0])
    png = probe.overlay_png(zyx(ct), affine, zyx(lesion), zyx(found))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", png[16:24])
    at, idat = 8, b""
    while at < len(png):
        n = struct.unpack(">I", png[at:at + 4])[0]
        if png[at + 4:at + 8] == b"IDAT":
            idat += png[at + 8:at + 8 + n]
        at += 12 + n
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + 3 * w)
    assert (raw[:, 0] == 0).all()
    px = raw[:, 1:].reshape(h, w, 3)
    green = ((px == (0, 255, 0)).all(-1)).sum()
    red = ((px == (255, 0, 0)).all(-1)).sum()
    assert green > 50 and red > 30 and green > red      # the reference contour is the longer one

    nib.save(nib.Nifti1Image(ct, AFFINE_RAS), str(tmp_path / "img.nii.gz"))
    nib.save(nib.Nifti1Image(lesion.astype(np.uint8), AFFINE_RAS), str(tmp_path / "ref.nii.gz"))

    class Lesions(MaskSegmenter):
        def __init__(self, **_):
            super().__init__({"tumore polmonare": zyx(found)}, "fake")

    class Organs(MaskSegmenter):
        def __init__(self, **_):
            super().__init__({"lung_upper_lobe_right": zyx(_ball((48, 48, 20), 40))}, "fake")

    monkeypatch.setattr(probe, "VoxTellSegmenter", Lesions)
    monkeypatch.setattr(probe, "TotalSegmentatorSegmenter", Organs)
    cases = [{"id": "syn", "image": str(tmp_path / "img.nii.gz"), "lesion": str(tmp_path / "ref.nii.gz"),
              "structure": ["lung_upper_lobe_right"], "prompt": "tumore polmonare"}]
    row = probe.run(cases, "cpu", True, overlays=tmp_path / "ov")[0]
    assert row["prompt"]["within_strict"] is False and row["prompt"]["overlay"] == "syn.png"
    assert (tmp_path / "ov" / "syn.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("anatomy", [True, False])
def test_a_run_saved_and_replayed_on_cpu_gives_the_same_row(tmp_path, monkeypatch, anatomy):
    lesion = _ball((48, 48, 20), 10)
    ct = np.full(SHAPE_XYZ, -850.0)
    ct[lesion] = 35.0
    nib.save(nib.Nifti1Image(ct, AFFINE_RAS), str(tmp_path / "img.nii.gz"))
    nib.save(nib.Nifti1Image(lesion.astype(np.uint8), AFFINE_RAS), str(tmp_path / "ref.nii.gz"))
    zyx = lambda m: m.transpose(2, 1, 0)  # noqa: E731
    found = _ball((48, 48, 20), 8)
    lobe = zyx(_ball((48, 48, 20), 40) & ~lesion)

    class Lesions(MaskSegmenter):
        def __init__(self, **_):
            super().__init__({"tumore polmonare": zyx(found), "lung tumor": zyx(found)}, "fake")

    class Organs(MaskSegmenter):
        def __init__(self, **_):
            super().__init__({"lung_upper_lobe_right": lobe}, "fake")

        def all_structures(self, volume, affine):
            from melampo.imaging.involvement import StructureLabels
            return StructureLabels(lobe.astype(np.int16) * 3, {3: "lung_upper_lobe_right"},
                                   regions={"thoracic_cavity": np.ones_like(lobe)})

    monkeypatch.setattr(probe, "VoxTellSegmenter", Lesions)
    monkeypatch.setattr(probe, "TotalSegmentatorSegmenter", Organs)
    cases = [{"id": "syn", "image": str(tmp_path / "img.nii.gz"), "lesion": str(tmp_path / "ref.nii.gz"),
              "structure": ["lung_upper_lobe_right"], "prompt": "tumore polmonare", "prompt_en": "lung tumor"}]
    first = probe.run(cases, "cpu", True, anatomy=anatomy, save_masks=tmp_path / "masks")[0]
    assert first["saved_masks"] == "syn.npz"

    class Broken:   # replay must not touch a segmenter
        def __init__(self, **_):
            raise AssertionError("a segmenter was built in a replay")

    monkeypatch.setattr(probe, "VoxTellSegmenter", Broken)
    monkeypatch.setattr(probe, "TotalSegmentatorSegmenter", Broken)
    again = probe.run(cases, "cpu", True, anatomy=anatomy, from_masks=tmp_path / "masks")[0]
    for key in ("prompt", "prompt_en"):
        a, b = first[key], again[key]
        assert a["measured_longest_axial_mm"] == b["measured_longest_axial_mm"] and a["dice"] == b["dice"]
        assert a["status"] == b["status"] and a["review"] == b["review"]
        if anatomy:
            assert a["anatomy"]["position"] == b["anatomy"]["position"]
    if anatomy:
        assert first["prompt"]["radiologist"]["window"]["axial_long_mm"] == \
            again["prompt"]["radiologist"]["window"]["axial_long_mm"]
    assert "replayed" in again["prompt"]["segmenter_notes"]
    if anatomy:   # the cavities of the atlas come back from the saved file
        assert first["atlas"]["regions"] == again["atlas"]["regions"] == ["thoracic_cavity"]
