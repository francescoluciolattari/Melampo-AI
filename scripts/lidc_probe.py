#!/usr/bin/env python
"""How radiologists draw a lung nodule, and where our readings fall among them (LIDC-IDRI, 28 October 2026).

MSD gives one reference contour per lesion: an error against it mixes our mistakes with that annotator's
convention. LIDC-IDRI (TCIA, CC BY 3.0) gives, for each nodule, the contours of four thoracic radiologists drawn
without forced consensus. For each nodule of a selection fixed before any run
(``data/lidc/selection_2026-10-28.json``, built from pylidc's annotation database: all four readers, slices
<= 2.5 mm, stratified by the readers' texture and size) the probe writes:

* **the readers among themselves**: each reader's longest axial diameter, short axis and mean (measured by the same
  code as ours, ``lesion_measures``), the spread (largest minus smallest), each reader's leave-one-out error (his
  diameter minus the mean of the other three), the Dice between readers, the 50% consensus (at least two readers)
  and the readers' own ratings (texture, margin, spiculation, lobulation, sphericity, calcification, subtlety);
* **our readings** -- VoxTell's region (the part that overlaps the readers' nodule: the report names where the
  lesion is), and the radiologist's readings on the atlas (``radiologist_reading``) -- with their diameters, Dice
  against the consensus, error against the readers' mean, whether they fall within the readers' range, and how
  their error compares with a reader's leave-one-out error ("a fifth reader");
* the image-side check against a synthetic report naming the lobe and the consensus size (as the MSD probe).

Nothing here is tuned: the numbers are a measure of what radiologists do. A rule chosen on the set "misura" is
verified on the set "verifica" of the same file, never seen while choosing.

    python scripts/lidc_probe.py --set misura --dicom-root DIR --out lidc.json [--device cuda] [--limit N]
        [--save-masks DIR | --from-masks DIR] [--isolate]

The DICOM series are read from ``DIR/<SeriesInstanceUID>/`` (as ``tcia_utils.nbia.downloadSeries`` writes them).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from melampo.imaging import findings as fd  # noqa: E402
from melampo.imaging.lesion_measures import label_parts, measure_region  # noqa: E402
from melampo.imaging.segmenters import Segmentation  # noqa: E402

SELECTION = Path(__file__).resolve().parents[1] / "data" / "lidc" / "selection_2026-10-28.json"
PROMPTS = {"prompt": "nodulo polmonare", "prompt_en": "lung nodule"}
RATINGS = ("subtlety", "internalStructure", "calcification", "sphericity", "margin", "lobulation", "spiculation",
           "texture", "malignancy")
READINGS = ("window_band", "edge", "edge_wide", "window", "solid", "total")


# --- the series ------------------------------------------------------------------------------------------------

def load_series(folder: Path):
    """The CT series in ``folder``: (volume (slice, row, column) in HU float32, affine (column, row, slice) ->
    patient LPS mm, the slices' z positions in ascending order)."""
    import pydicom

    slices = []
    for f in sorted(Path(folder).rglob("*")):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f))
        except Exception:  # noqa: BLE001 -- not a DICOM file
            continue
        if "ImagePositionPatient" not in ds or "PixelData" not in ds:
            continue
        slices.append(ds)
    if not slices:
        raise FileNotFoundError(f"no CT slice in {folder}")
    slices.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    zs, kept = [], []
    for d in slices:                        # duplicated positions (repeated slices) are kept once
        z = float(d.ImagePositionPatient[2])
        if zs and abs(z - zs[-1]) < 1e-3:
            continue
        zs.append(z)
        kept.append(d)
    vol = np.stack([d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1))
                    + float(getattr(d, "RescaleIntercept", 0)) for d in kept])
    iop = [float(x) for x in kept[0].ImageOrientationPatient]
    row_sp, col_sp = (float(x) for x in kept[0].PixelSpacing)
    first, last = np.array(kept[0].ImagePositionPatient, float), np.array(kept[-1].ImagePositionPatient, float)
    step = (last - first) / max(len(kept) - 1, 1) if len(kept) > 1 else np.array([0, 0, 1.0])
    affine = np.eye(4)
    affine[:3, 0] = np.array(iop[:3]) * col_sp
    affine[:3, 1] = np.array(iop[3:]) * row_sp
    affine[:3, 2] = step
    affine[:3, 3] = first
    return vol, affine, np.array(zs)


def place_mask(mask_rcs: np.ndarray, bbox, slice_index: list[int], shape) -> np.ndarray:
    """A pylidc annotation mask (row, column, slice inside its bounding box) on the series' grid (slice, row,
    column). ``slice_index``: for each pylidc slice index, ours (pylidc sorts by z too, but a series with a
    repeated slice has one more there)."""
    full = np.zeros(shape, dtype=bool)
    rows, cols, sls = bbox
    for k in range(mask_rcs.shape[2]):
        ours = slice_index[sls.start + k]
        if ours is None or not mask_rcs[:, :, k].any():
            continue
        full[ours, rows.start:rows.stop, cols.start:cols.stop] |= mask_rcs[:, :, k]
    return full


def slice_map(pylidc_z: list[float], our_z: np.ndarray, tol: float = 0.6) -> list[int | None]:
    out = []
    for z in pylidc_z:
        k = int(np.argmin(np.abs(our_z - z)))
        out.append(k if abs(our_z[k] - z) <= tol else None)
    return out


def reader_annotations(ids: list[int]):
    """The four readers' annotations of one nodule from pylidc's database (numpy aliases restored for it)."""
    for name, kind in (("int", int), ("float", float), ("bool", bool)):
        if not hasattr(np, name):
            setattr(np, name, kind)
    import pylidc as pl

    anns = pl.query(pl.Annotation).filter(pl.Annotation.id.in_(ids)).all()
    return sorted(anns, key=lambda a: a.id)


# --- the readers among themselves -------------------------------------------------------------------------------

def _axes(mask, affine) -> dict:
    if not mask.any():
        return {"axial_long_mm": None, "axial_short_mm": None, "axial_mean_mm": None, "volume_mm3": 0.0}
    m = measure_region(mask, affine)
    return {"axial_long_mm": m.longest_axial_diameter_mm, "axial_short_mm": m.axial_short_mm,
            "axial_mean_mm": m.axial_mean_mm, "volume_mm3": m.volume_mm3}


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = int(a.sum()) + int(b.sum())
    return float(2 * (a & b).sum() / s) if s else 1.0


def reader_stats(masks: list[np.ndarray], affine) -> dict:
    """The readers' measures, their spread and each one's leave-one-out error, the Dice between them, and the
    50% consensus (voxels outlined by at least half of them)."""
    per = [_axes(m, affine) for m in masks]
    longs = [p["axial_long_mm"] for p in per]
    means = [p["axial_mean_mm"] for p in per]
    loo = [longs[i] - st.mean(longs[:i] + longs[i + 1:]) for i in range(len(longs))]
    loo_mean = [means[i] - st.mean(means[:i] + means[i + 1:]) for i in range(len(means))]
    pair = [dice(masks[i], masks[j]) for i in range(len(masks)) for j in range(i + 1, len(masks))]
    votes = np.sum(masks, axis=0)
    consensus = votes >= max(2, int(np.ceil(len(masks) / 2)))
    return {"readers": per, "long_mean_mm": st.mean(longs), "long_min_mm": min(longs), "long_max_mm": max(longs),
            "long_spread_mm": max(longs) - min(longs), "long_sd_mm": st.pstdev(longs),
            "leave_one_out_mm": loo, "leave_one_out_abs_median_mm": st.median(abs(x) for x in loo),
            "mean_axis_mean_mm": st.mean(means), "mean_axis_leave_one_out_mm": loo_mean,
            "dice_between_readers": pair, "dice_between_readers_median": st.median(pair) if pair else None,
            "consensus": _axes(consensus, affine)}, consensus


def against_readers(measure: dict, mask: np.ndarray | None, stats: dict, consensus: np.ndarray) -> dict:
    """One of our readings against the four readers: error on the readers' mean, within their range, and how its
    error compares with a reader's leave-one-out error (is it a fifth reader?)."""
    d = measure.get("axial_long_mm")
    if d is None:
        return {"axial_long_mm": None}
    err = d - stats["long_mean_mm"]
    loo = sorted(abs(x) for x in stats["leave_one_out_mm"])
    out = {"axial_long_mm": d, "axial_short_mm": measure.get("axial_short_mm"),
           "axial_mean_mm": measure.get("axial_mean_mm"),
           "error_on_readers_mean_mm": err,
           "within_readers_range": stats["long_min_mm"] - 1e-6 <= d <= stats["long_max_mm"] + 1e-6,
           "error_vs_worst_reader": abs(err) <= loo[-1] + 1e-6,
           "error_vs_median_reader": abs(err) <= st.median(loo) + 1e-6,
           "dice_consensus": dice(mask, consensus) if mask is not None else None}
    if measure.get("axial_mean_mm") is not None:
        out["mean_axis_error_on_readers_mm"] = measure["axial_mean_mm"] - stats["mean_axis_mean_mm"]
    return out


def pick_component(mask: np.ndarray | None, target: np.ndarray, affine) -> tuple[np.ndarray | None, dict]:
    """The part of the segmenter's region that is this nodule: the connected part overlapping the readers'
    nodule most (the report names where the lesion is); else nothing -- the nodule was not found."""
    if mask is None or not mask.any():
        return None, {"found": False, "parts": 0}
    labels, sizes = label_parts(mask)
    over = np.bincount(labels[target].ravel(), minlength=len(sizes) + 1)[1:]
    if not over.any():
        return None, {"found": False, "parts": len(sizes)}
    k = int(np.argmax(over)) + 1
    return labels == k, {"found": True, "parts": len(sizes), "overlapping_parts": int((over > 0).sum())}


# --- one nodule --------------------------------------------------------------------------------------------------

def probe_nodule(item: dict, dicom_root: Path, lesion_seg, organ_seg, save_masks: Path | None = None,
                 annotations=reader_annotations) -> dict:
    from melampo.imaging.involvement import involvement
    from melampo.imaging.radiologist_reading import radiologist_readings, readings_summary
    from imaging_probe import LOBES_IT, save_case_masks, synthetic_clause

    case_id = f"{item['patient']}_{item['annotation_ids'][0]}"
    row = {"id": case_id, "patient": item["patient"], "series": item["series"], "stratum": item.get("stratum"),
           "slice_thickness": item.get("slice_thickness")}
    for seg_obj in {id(lesion_seg): lesion_seg, id(organ_seg): organ_seg}.values():
        if hasattr(seg_obj, "set_case"):
            seg_obj.set_case(case_id)
    volume, affine, zs = load_series(Path(dicom_root) / item["series"])
    row["slices"] = int(volume.shape[0])
    anns = annotations(item["annotation_ids"])
    scan_z = [float(z) for z in anns[0].scan.slice_zvals] if hasattr(anns[0], "scan") else list(zs)
    smap = slice_map(scan_z, zs)
    masks = [place_mask(a.boolean_mask(), a.bbox(), smap, volume.shape) for a in anns]
    row["ratings"] = [{k: getattr(a, k, None) for k in RATINGS} | {"pylidc_diameter_mm": float(a.diameter)}
                      for a in anns]
    stats, consensus = reader_stats(masks, affine)
    row["readers"] = stats

    structures = organ_seg.all_structures(volume, affine) if hasattr(organ_seg, "all_structures") else None
    site = None
    if structures is not None:
        lobes = {n: structures[n] for n in LOBES_IT if n in structures}
        _, pos = involvement(consensus, lobes, affine)
        site = pos.host if pos is not None else None
    row["report_site"] = site
    clause = synthetic_clause(stats["consensus"]["axial_long_mm"], site)
    row["report_clause"] = clause
    text_size = max((s["largest_mm"] for s in fd.sizes_in_text(clause)), default=None)

    keys = list(PROMPTS)
    t0 = time.time()
    segs = lesion_seg.segment(volume, affine, [PROMPTS[k] for k in keys])
    row["seconds"] = round(time.time() - t0, 1)
    if save_masks is not None:
        try:
            row["saved_masks"] = save_case_masks(save_masks, case_id, [PROMPTS[k] for k in keys], segs,
                                                 structures).name
        except Exception as error:  # noqa: BLE001
            row["save_masks_error"] = f"{type(error).__name__}: {error}"[:300]
    organ = structures[site] if site and structures is not None and site in structures else None
    for key, seg in zip(keys, segs):
        comp, found = pick_component(seg.mask, consensus, affine)
        out = {"found": found}
        if comp is None:
            row[key] = out
            continue
        one = Segmentation(comp, seg.provider, seg.version, seg.prompt, seg.notes)
        ev = fd.image_evidence(one, affine, volume, organ, text_size, modality="CT", structure=site,
                               structures=structures)
        out.update(status=ev.status, needs_review=ev.needs_review, review=list(ev.review))
        out["segmenter"] = against_readers(_axes(comp, affine), comp, stats, consensus)
        if key == keys[0]:
            r = radiologist_readings(volume, comp, affine, "CT", site, structures)
            summ = readings_summary(r)
            out["readings_notes"] = summ.get("notes")
            out["edge_rule"] = (summ.get("edge") or {}).get("rule")
            for name in READINGS:
                reading = r.get(name)
                if reading is None:
                    continue
                out[name] = against_readers(reading, reading.get("mask"), stats, consensus)
                out[name]["notes"] = reading.get("notes")
        row[key] = out
    return row


def run(items: list[dict], dicom_root: Path, device: str, out_path: Path | None, save_masks: Path | None = None,
        from_masks: Path | None = None) -> list[dict]:
    from imaging_probe import ReplaySegmenters, _free_memory

    if from_masks is not None:
        lesion_seg = organ_seg = ReplaySegmenters(from_masks)
    else:
        from melampo.imaging.segmenters import TotalSegmentatorSegmenter, VoxTellSegmenter

        lesion_seg = VoxTellSegmenter(device=device)
        organ_seg = TotalSegmentatorSegmenter(fast=True, device="gpu" if device == "cuda" else "cpu")
    rows = []
    for item in items:
        try:
            row = probe_nodule(item, dicom_root, lesion_seg, organ_seg, save_masks)
        except Exception as error:  # noqa: BLE001 -- one nodule failing must not lose the others
            row = {"id": f"{item.get('patient')}_{(item.get('annotation_ids') or ['?'])[0]}",
                   "error": f"{type(error).__name__}: {error}"[:500]}
        rows.append(row)
        print(json.dumps(row, default=float), flush=True)
        if out_path is not None:
            out_path.write_text(json.dumps(rows, indent=1, default=float))
        _free_memory()
    return rows


def run_isolated(items, args, out_path: Path) -> list[dict]:
    import subprocess
    import tempfile

    rows = []
    for item in items:
        with tempfile.TemporaryDirectory() as tmp:
            sel, res = Path(tmp) / "one.json", Path(tmp) / "row.json"
            sel.write_text(json.dumps({args.set: [item]}))
            cmd = [sys.executable, str(Path(__file__).resolve()), "--selection", str(sel), "--set", args.set,
                   "--dicom-root", args.dicom_root, "--out", str(res), "--device", args.device]
            for flag, attr in (("--save-masks", "save_masks"), ("--from-masks", "from_masks")):
                if getattr(args, attr, None):
                    cmd += [flag, getattr(args, attr)]
            done = subprocess.run(cmd, check=False)
            try:
                row = json.loads(res.read_text())[0]
            except Exception:  # noqa: BLE001
                row = {"id": f"{item.get('patient')}_{item['annotation_ids'][0]}",
                       "error": f"process ended with code {done.returncode} (killed: -9 is the system out of memory)"}
        rows.append(row)
        print(json.dumps(row, default=float), flush=True)
        out_path.write_text(json.dumps(rows, indent=1, default=float))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selection", default=str(SELECTION))
    ap.add_argument("--set", default="misura", choices=("misura", "verifica"))
    ap.add_argument("--dicom-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-masks", metavar="DIR")
    ap.add_argument("--from-masks", metavar="DIR")
    ap.add_argument("--isolate", action="store_true")
    args = ap.parse_args(argv)
    items = json.loads(Path(args.selection).read_text())[args.set]
    if args.limit:
        items = items[: args.limit]
    out = Path(args.out)
    if args.isolate and len(items) > 1:
        rows = run_isolated(items, args, out)
    else:
        rows = run(items, Path(args.dicom_root), args.device, out,
                   Path(args.save_masks) if args.save_masks else None,
                   Path(args.from_masks) if args.from_masks else None)
    out.write_text(json.dumps(rows, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
