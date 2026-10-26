"""The patient's atlas from TotalSegmentator's main and open tasks (25 October 2026), with TotalSegmentator
replaced by a fake that writes known label maps."""

import sys
import types

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from melampo.imaging.segmenters import TotalSegmentatorSegmenter  # noqa: E402

CLASS_MAP = {
    "total": {1: "lung_upper_lobe_right", 2: "aorta"},
    "lung_vessels": {1: "lung_airways", 2: "lung_airways_wall", 3: "lung_arteries", 4: "lung_veins"},
    "pleural_pericard_effusion": {1: "lung_pleural", 2: "pleural_effusion", 3: "pericardial_effusion"},
    "abdominal_muscles": {1: "pectoralis_major_right"},
    "trunk_cavities": {1: "abdominal_cavity", 2: "thoracic_cavity", 3: "pericardium", 4: "mediastinum"},
}


@pytest.fixture
def fake_ts(monkeypatch):
    calls = []

    def totalsegmentator(src, out, ml=True, device="gpu", task="total", fast=False, nr_thr_saving=6,
                         force_split=False):
        calls.append((task, fast))
        if force_split:
            calls.append(("split", task))
        if task == "abdominal_muscles":
            raise RuntimeError("weights not downloaded")
        img = nib.load(src)
        x, y, z = img.shape
        lab = np.zeros(img.shape, np.uint8)
        if task == "total":
            lab[: x // 2] = 1                      # a lobe over half the volume
            lab[x // 2:, :2] = 2                   # the aorta in a corner
        elif task == "lung_vessels":
            lab[:4, :4] = 3                        # an artery inside the lobe
        elif task == "pleural_pericard_effusion":
            lab[:] = 1                             # the auxiliary lung class everywhere
            lab[-2:, -2:] = 2
        elif task == "trunk_cavities":
            lab[: x // 2] = 2
            lab[x // 2:] = 4
        nib.save(nib.Nifti1Image(lab, img.affine), out)

    api = types.ModuleType("totalsegmentator.python_api")
    api.totalsegmentator = totalsegmentator
    mapping = types.ModuleType("totalsegmentator.map_to_binary")
    mapping.class_map = CLASS_MAP
    pkg = types.ModuleType("totalsegmentator")
    monkeypatch.setitem(sys.modules, "totalsegmentator", pkg)
    monkeypatch.setitem(sys.modules, "totalsegmentator.python_api", api)
    monkeypatch.setitem(sys.modules, "totalsegmentator.map_to_binary", mapping)
    return calls


def test_the_atlas_merges_the_finer_tasks_and_keeps_the_cavities_apart(fake_ts):
    volume = np.zeros((6, 10, 10), np.float32)                     # (slice, row, column)
    atlas = TotalSegmentatorSegmenter(fast=True, device="cpu").all_structures(volume, np.eye(4))
    assert ("total", True) in fake_ts and ("lung_vessels", False) in fake_ts   # fast only for the main task
    assert {"lung_upper_lobe_right", "aorta", "lung_arteries", "pleural_effusion"} <= set(atlas)
    assert "lung_pleural" not in atlas                              # a helper class, not a structure
    art = atlas["lung_arteries"]
    assert art.any() and not (atlas["lung_upper_lobe_right"] & art).any()   # the finer task wins
    assert set(atlas.regions) == {"thoracic_cavity", "mediastinum"}
    assert atlas.notes and atlas.notes[0].startswith("abdominal_muscles:RuntimeError")


def test_on_mr_only_the_main_mr_task(fake_ts):
    CLASS_MAP["total_mr"] = {1: "lung_right"}
    atlas = TotalSegmentatorSegmenter(fast=False, device="cpu").all_structures(
        np.zeros((6, 10, 10), np.float32), np.eye(4), modality="MR")
    assert [t for t, _ in fake_ts] == ["total_mr"] and atlas.regions == {}


def test_a_long_series_is_segmented_in_parts(fake_ts):
    TotalSegmentatorSegmenter(fast=True, device="cpu").all_structures(np.zeros((450, 4, 4), np.float32), np.eye(4))
    assert ("split", "total") in fake_ts and ("split", "lung_vessels") in fake_ts
