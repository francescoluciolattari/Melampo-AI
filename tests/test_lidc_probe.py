"""The LIDC-IDRI probe (28 October 2026): series reading, the readers' masks on the series grid, the readers
among themselves, and our reading against them. A synthetic DICOM series with the geometry of a real LIDC scan;
the four readers' annotations come from pylidc's own database."""

import json
from pathlib import Path

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

import lidc_probe as lp  # noqa: E402


def _write_series(folder: Path, volume: np.ndarray, zs, spacing: float, shuffle=True):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    folder.mkdir(parents=True, exist_ok=True)
    order = list(range(len(zs)))
    if shuffle:
        order = order[::-1]                                     # files not in z order
    for n, k in enumerate(order):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(folder / f"{n}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.Modality = "CT"
        ds.Rows, ds.Columns = volume.shape[1:]
        ds.PixelSpacing = [spacing, spacing]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [-150.0, -160.0, float(zs[k])]
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
        ds.PixelData = (volume[k] + 1024).astype(np.int16).tobytes()
        ds.save_as(str(folder / f"{n}.dcm"), enforce_file_format=True)


def test_a_series_is_read_in_z_order_with_its_geometry(tmp_path):
    vol = np.zeros((5, 8, 6), np.float32)
    for k in range(5):
        vol[k] = k * 10 - 1000
    _write_series(tmp_path / "s", vol, [10.0, 12.5, 15.0, 17.5, 20.0], 0.7)
    got, affine, zs, meta = lp.load_series(tmp_path / "s")
    assert abs(meta["slice_step_mm"] - 2.5) < 1e-6
    assert got.shape == (5, 8, 6) and np.allclose(got[:, 0, 0], [-1000, -990, -980, -970, -960])
    assert np.allclose(zs, [10, 12.5, 15, 17.5, 20]) and np.allclose(np.diag(affine)[:3], [0.7, 0.7, 2.5])


def test_readers_among_themselves_and_a_fifth_reader():
    aff = np.diag([1.0, 1.0, 1.0, 1.0])
    z, y, x = np.mgrid[0:20, 0:40, 0:40]
    masks = [((x - 20) ** 2 + (y - 20) ** 2 <= r * r) & (abs(z - 10) <= 3) for r in (5, 6, 6, 7)]
    stats, consensus = lp.reader_stats(masks, aff)
    assert stats["long_min_mm"] < stats["long_max_mm"] and abs(stats["long_spread_mm"] - 4) < 1.1
    assert len(stats["leave_one_out_mm"]) == 4 and consensus.sum() > 0
    mine = ((x - 20) ** 2 + (y - 20) ** 2 <= 36) & (abs(z - 10) <= 3)
    got = lp.against_readers(lp._axes(mine, aff), mine, stats, consensus)
    assert got["within_readers_range"] and got["error_vs_worst_reader"] and got["dice_consensus"] > 0.9


def test_the_part_of_the_region_that_is_the_nodule():
    aff = np.eye(4)
    z, y, x = np.mgrid[0:10, 0:30, 0:30]
    a = (x - 8) ** 2 + (y - 8) ** 2 <= 9
    b = (x - 22) ** 2 + (y - 22) ** 2 <= 9
    comp, info = lp.pick_component(a | b, b & (z > 2), aff)
    assert info["found"] and info["parts"] == 2 and (comp == b).all()
    none, info = lp.pick_component(a, b, aff)
    assert none is None and not info["found"]


def test_one_real_lidc_nodule_on_a_synthetic_series(tmp_path):
    pl = pytest.importorskip("pylidc")
    sel = json.loads(lp.SELECTION.read_text())
    item = sel["misura"][0]
    anns = lp.reader_annotations(item["annotation_ids"])
    scan = anns[0].scan
    zs = [float(z) for z in scan.slice_zvals]
    sp = float(scan.pixel_spacing)
    vol = np.full((len(zs), 512, 512), -850.0, np.float32)
    smap = list(range(len(zs)))
    masks = [lp.place_mask(a.boolean_mask(), a.bbox(), smap, vol.shape) for a in anns]
    union = np.any(masks, axis=0)
    vol[union] = 30.0
    _write_series(tmp_path / item["series"], vol, zs, sp)

    from imaging_probe import ReplaySegmenters  # noqa: F401  (the probe imports it)
    from melampo.imaging.involvement import StructureLabels
    from melampo.imaging.segmenters import MaskSegmenter

    class Lesions(MaskSegmenter):
        def __init__(self):
            super().__init__({p: union for p in lp.PROMPTS.values()}, "fake")

    class Organs:
        def all_structures(self, volume, affine):
            lab = np.zeros(volume.shape, np.int16)             # a lobe around the nodule (small: memory)
            idx = np.argwhere(union)
            lo, hi = np.maximum(idx.min(0) - (4, 30, 30), 0), idx.max(0) + (5, 31, 31)
            lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 1
            lab[union] = 0
            return StructureLabels(lab, {1: "lung_upper_lobe_right"})

    row = lp.probe_nodule(item, tmp_path, Lesions(), Organs())
    r = row["readers"]
    assert len(r["readers"]) == 4 and r["long_spread_mm"] >= 0 and r["consensus"]["axial_long_mm"] > 0
    p = row["prompt"]
    assert p["found"]["found"] and p["segmenter"]["dice_consensus"] > 0.6
    assert p["segmenter"]["within_readers_range"] or abs(p["segmenter"]["error_on_readers_mean_mm"]) < 5
    assert row["report_site"] == "lung_upper_lobe_right" and "lobo superiore destro" in row["report_clause"]
    assert "edge" in p and p["edge"]["axial_long_mm"] is not None and p["edge"].get("caliper_long_mm")
    assert set(p["size_check_by_policy"]) == {"fixed", "readers"}
    assert set(p["sweep"]) == {"0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"}
    assert row["acquisition"]["pixel_spacing_mm"] and "small_for_the_slices" in row
    assert len(row["ratings"]) == 4 and all(x["texture"] for x in row["ratings"])
    json.dumps(row, default=float)
