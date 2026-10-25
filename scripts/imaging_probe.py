#!/usr/bin/env python
"""Measure the imaging interpreter on public CT/MR with reference lesion masks (16 October 2026).

For each case: the image (NIfTI), the reference lesion mask (NIfTI, same grid), the structure the lesion is in
(a TotalSegmentator class) and a prompt (the words a report would use). VoxTell isolates the region from the
prompt, TotalSegmentator gives the organ, ``findings.image_evidence`` runs the checks of decision D19 (organ,
size written in the report, detached parts, the threshold reading of ``threshold_check`` on lung CT), and the
probe writes per case: Dice of the region, of its largest part and of the threshold reading against the
reference, the status, whether a person must look and why, the sizes, the run time.

Where it runs: Kaggle (free GPU, public data only) or a CPU runner for one or two small cases. Public data only:
pseudonymised patient data is still personal data and these services have no data-processing agreement.

    python scripts/imaging_probe.py --cases cases.json --out probe.json [--device cuda|cpu] [--fast-organs]

``cases.json``: [{"id": "msd_liver_001", "image": "...nii.gz", "lesion": "...nii.gz", "structure": "liver",
"prompt": "lesione ipodensa nel fegato", "prompt_en": "hypodense liver lesion", "modality": "CT",
"report": optional clause of the real report (else a synthetic one with the reference size, flagged as such)}]
Datasets with permissive licences for this: Medical Segmentation Decathlon (CC BY-SA 4.0: liver, lung, pancreas,
colon tumours), TotalSegmentator CT (CC BY 4.0: organs, no lesions).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from melampo.imaging import findings as fd  # noqa: E402
from melampo.imaging import threshold_check as tc  # noqa: E402
from melampo.imaging.density import density_profile  # noqa: E402
from melampo.imaging.lesion_measures import measure_region, split_main_part  # noqa: E402
from melampo.imaging.segmenters import (  # noqa: E402
    MaskSegmenter,
    TotalSegmentatorSegmenter,
    VoxTellSegmenter,
    read_nifti_mask,
)


def load_case(path: str):
    import nibabel as nib

    img = nib.load(path)
    ras = np.asarray(img.affine, dtype=float)
    lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ ras
    # float32: a 512x512x567 series is 0.6 GB instead of 1.2 GB (the second probe ran out of memory on it)
    return np.asarray(img.get_fdata(dtype=np.float32)).transpose(2, 1, 0), lps


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0


def synthetic_clause(reference_mm: float | None) -> str:
    """MSD has no reports: the probe writes the sentence a report would carry, with the reference size and no
    other character (nothing the data does not say), and the finding's size is read from it by the same reader
    used on real reports (``findings.sizes_in_text``). The sentence is synthetic and labelled so in the output."""
    if not reference_mm:
        return "Formazione polmonare."
    return f"Formazione polmonare di {reference_mm / 10:.1f} cm.".replace(".", ",", 1)


def _free_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _density_summary(d) -> dict | None:
    if d is None:
        return None
    return {"units": d.units, "median": d.percentiles["p50"], "relative": d.relative,
            "relative_value": d.relative_value, "composition": d.composition, "solid_axial_mm": d.solid_axial_mm,
            "flags": list(d.flags), "border_gradient": d.border_gradient,
            "core_to_rim": [round(r.get("mean", float("nan")), 1) for r in d.depth_profile][::-1]}


def _axial(mask, affine):
    return measure_region(mask, affine).longest_axial_diameter_mm if mask is not None and mask.any() else None


def _merge_near(mask, affine, gap_mm: float = 5.0):
    """The largest part with the parts closer than ``gap_mm`` to it (a large tumour split by a thin gap)."""
    main, frags = split_main_part(mask, affine)
    if not frags:
        return main
    from melampo.imaging.lesion_measures import label_parts

    labels, _ = label_parts(mask)
    keep = main.copy()
    for k, f in enumerate(frags, start=2):
        if f.distance_mm <= gap_mm:
            keep |= labels == k
    return keep


def _shape_diagnostics(main, ref, volume, affine) -> dict:
    """Volume right but diameter short (fifth probe): is the miss a tail at the two ends of the reference's long
    axis, one detached blob, or spread around? Extents along the principal axes (1st-99th percentile) of region
    and reference, the share of missed voxels in the outer fifth of each end, the missed blobs, and the
    sphericity and elongation of both."""
    from melampo.imaging.lesion_measures import label_parts, to_mm

    def extent(mask):
        pts = to_mm(np.argwhere(mask), affine)
        pts = pts - pts.mean(axis=0)
        vec = np.linalg.eigh(np.cov(pts.T))[1][:, ::-1]
        proj = pts @ vec
        return [round(float(x), 1) for x in np.percentile(proj, 99, axis=0) - np.percentile(proj, 1, axis=0)], vec[:, 0]

    out: dict = {}
    out["extent_reference_mm"], axis = extent(ref)
    out["extent_region_mm"], _ = extent(main)
    missed = ref & ~main
    if missed.any():
        p_ref = to_mm(np.argwhere(ref), affine) @ axis
        p_mis = to_mm(np.argwhere(missed), affine) @ axis
        lo, hi = np.percentile(p_ref, 1), np.percentile(p_ref, 99)
        span = hi - lo
        out["missed_at_ends_share"] = float(((p_mis < lo + 0.2 * span) | (p_mis > hi - 0.2 * span)).mean())
        labels, sizes = label_parts(missed)
        out["missed_blobs"] = len(sizes)
        if sizes:
            out["missed_largest_blob_share"] = float(sizes[0] / missed.sum())
            out["missed_largest_blob_median_hu"] = float(np.median(volume[labels == 1]))
    for name, m in (("region", main), ("reference", ref)):
        r = measure_region(m, affine)
        out[f"sphericity_{name}"] = r.sphericity
        out[f"elongation_{name}"] = r.elongation
    return out


def size_diagnostics(volume, affine, reference, main, full, lesion_seg, prompt, refiner=None) -> dict:
    """Why a measure is short of the reference (fifth probe: under-measures growing with size). Nothing here
    decides: each line is a candidate cause, measured.

    * the region vs the reference: volume ratio, share of the reference missed, and whether the missed voxels are
      a rim around the region (within 2 mm of it: the border placed inside) or whole lobes (farther: parts not
      found), with their density;
    * the 0.5 cut on VoxTell's probabilities: the region at 0.35 and 0.2, and the soft volume;
    * the scale: VoxTell on the series resampled to 1.5 mm (its README's suggestion);
    * parts split by a thin gap merged with the main one;
    * a second network, nnInteractive, refining the region from a box."""
    from melampo.imaging.density import distance_mm, spacing_zyx
    from melampo.imaging.segmenters import mask_back, resample_to

    out: dict = {}
    sp = spacing_zyx(affine)
    vox = float(np.prod(sp))
    ref = reference.astype(bool)
    if not main.any() or not ref.any():
        return {"note": "nothing to compare"}
    out["volume_ratio"] = float(main.sum() / ref.sum())
    missed = ref & ~main
    out["missed_share_of_reference"] = float(missed.sum() / ref.sum())
    out["extra_share_of_region"] = float((main & ~ref).sum() / main.sum())
    if missed.any():
        box = tuple(slice(max(0, int(a.min()) - 5), int(a.max()) + 6) for a in np.nonzero(ref | main))
        d, _ = distance_mm(~main[box], sp)
        md = d[missed[box]]
        out["missed_within_2mm_share"] = float((md <= 2.0).mean())
        out["missed_distance_p90_mm"] = float(np.percentile(md, 90))
        out["missed_median_hu"] = float(np.median(volume[missed]))
        out["rim_thickness_mm"] = float(missed.sum() * vox / max(measure_region(main, affine).surface_mm2 or 1, 1))
    try:   # shape: same volume but a shorter diameter means the region is rounder than the reference
        out["shape"] = _shape_diagnostics(main, ref, volume, affine)
    except Exception as error:  # noqa: BLE001
        out["shape_error"] = f"{type(error).__name__}: {error}"[:200]
    out["merged_near_parts_axial_mm"] = None
    try:
        merged = _merge_near(full, affine)
        out["merged_near_parts_axial_mm"] = _axial(merged, affine)
    except Exception as error:  # noqa: BLE001
        out["merged_error"] = str(error)[:200]
    try:
        prob = lesion_seg.probabilities(volume, affine, [prompt])[0]
        out["soft_volume_ratio"] = float(prob.sum() / ref.sum())
        for cut in (0.5, 0.35, 0.2):
            m = split_main_part(prob >= cut, affine)[0] if (prob >= cut).any() else prob >= cut
            out[f"cut_{cut}"] = {"axial_mm": _axial(m, affine), "dice": dice(m, ref)}
    except Exception as error:  # noqa: BLE001
        out["probabilities_error"] = f"{type(error).__name__}: {error}"[:300]
    try:
        small, small_aff, _ = resample_to(volume, affine, 1.5)
        seg = lesion_seg.segment(small, small_aff, [prompt])[0]
        if seg.mask is not None:
            back = mask_back(seg.mask, volume.shape)
            m = split_main_part(back, affine)[0] if back.any() else back
            out["resampled_1_5mm"] = {"axial_mm": _axial(m, affine), "dice": dice(m, ref)}
        else:
            out["resampled_1_5mm"] = {"axial_mm": None, "dice": 0.0}
    except Exception as error:  # noqa: BLE001
        out["resampled_error"] = f"{type(error).__name__}: {error}"[:300]
    if refiner is not None:
        try:
            r = refiner.refine(volume, main)
            m = split_main_part(r, affine)[0] if r.any() else r
            out["nninteractive"] = {"axial_mm": _axial(m, affine), "dice": dice(m, ref)}
        except Exception as error:  # noqa: BLE001
            out["nninteractive_error"] = f"{type(error).__name__}: {error}"[:300]
    return out


def _pack(mask: np.ndarray) -> dict:
    m = np.asarray(mask, dtype=bool)
    return {"bits": np.packbits(m.ravel()), "shape": np.array(m.shape)}


def _unpack(bits: np.ndarray, shape: np.ndarray) -> np.ndarray:
    shape = tuple(int(x) for x in shape)
    return np.unpackbits(bits)[: int(np.prod(shape))].reshape(shape).astype(bool)


def save_case_masks(folder: Path, case_id: str, prompts: list[str], segs, structures=None,
                    organs: dict | None = None) -> Path:
    """What the GPU computed for one case -- the segmenter's region per prompt and the whole-body structures (or
    the named organs) -- so that the analysis can be run again on a CPU without segmenting again."""
    folder.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"prompts": np.array(json.dumps(prompts))}
    for i, seg in enumerate(segs):
        if seg.mask is not None:
            for k, v in _pack(seg.mask).items():
                arrays[f"lesion{i}_{k}"] = v
        arrays[f"lesion{i}_meta"] = np.array(json.dumps({"provider": seg.provider, "version": seg.version,
                                                         "notes": list(seg.notes)}))
    if structures is not None and hasattr(structures, "labels"):
        arrays["labels"] = np.asarray(structures.labels, dtype=np.int16)
        for name, m in (getattr(structures, "regions", None) or {}).items():   # the cavities (containers)
            for k, v in _pack(m).items():
                arrays[f"region__{name}__{k}"] = v
        arrays["label_names"] = np.array(json.dumps({str(v): k for k, v in structures.code.items()}))
    elif structures:
        organs = {**(organs or {}), **dict(structures.items())}
    for name, m in (organs or {}).items():
        if m is not None:
            for k, v in _pack(m).items():
                arrays[f"organ__{name}__{k}"] = v
    target = folder / f"{case_id}.npz"
    np.savez_compressed(target, **arrays)
    return target


class ReplaySegmenters:
    """The segmenters of a GPU run read back from ``save_case_masks`` files: the same regions, no GPU. One object
    serves as both the lesion and the organ segmenter; ``set_case`` picks the case."""

    name = "replay"

    def __init__(self, folder: Path):
        self.folder, self.data, self.case = Path(folder), None, None

    def set_case(self, case_id: str) -> None:
        f = self.folder / f"{case_id}.npz"
        if not f.exists():
            raise FileNotFoundError(f"no saved masks for {case_id} in {self.folder}")
        self.data, self.case = dict(np.load(f, allow_pickle=False)), case_id

    def segment(self, volume, affine, prompts):
        from melampo.imaging.segmenters import Segmentation

        saved = json.loads(str(self.data["prompts"]))
        out = []
        for p in prompts:
            if p in saved:   # a lesion prompt
                i = saved.index(p)
                meta = json.loads(str(self.data[f"lesion{i}_meta"]))
                m = _unpack(self.data[f"lesion{i}_bits"], self.data[f"lesion{i}_shape"]) \
                    if f"lesion{i}_bits" in self.data else None
                out.append(Segmentation(m, f"replay:{meta['provider']}", meta["version"], p,
                                        notes=tuple(meta["notes"]) + ("replayed",)))
            else:            # an organ
                m = self._organ(p)
                out.append(Segmentation(m if m is not None and m.any() else None, "replay:organs", "", p))
        return out

    def _organ(self, name):
        if "labels" in self.data:
            names = {v: int(k) for k, v in json.loads(str(self.data["label_names"])).items()}
            return self.data["labels"] == names[name] if name in names else None
        key = f"organ__{name}__bits"
        return _unpack(self.data[key], self.data[f"organ__{name}__shape"]) if key in self.data else None

    def all_structures(self, volume, affine, **_):
        from melampo.imaging.involvement import StructureLabels
        from melampo.imaging.segmenters import SegmenterUnavailable

        if "labels" in self.data:
            names = {int(k): v for k, v in json.loads(str(self.data["label_names"])).items()}
            regions = {k.split("__")[1]: _unpack(self.data[k], self.data[k.replace("__bits", "__shape")])
                       for k in self.data if k.startswith("region__") and k.endswith("__bits")}
            return StructureLabels(self.data["labels"], names, regions)
        organs = {k.split("__")[1]: self._organ(k.split("__")[1]) for k in self.data if k.startswith("organ__")
                  and k.endswith("__bits")}
        if not organs:
            raise SegmenterUnavailable("the saved run has no whole-body structures (run it with --anatomy)")
        return organs


def _png(rgb: np.ndarray) -> bytes:
    """A PNG from an (h, w, 3) uint8 array, with the standard library only (no matplotlib on a bare runner)."""
    import struct
    import zlib

    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _edge2d(m: np.ndarray) -> np.ndarray:
    """The border pixels of a 2D mask (inside pixels with a 4-neighbour outside)."""
    p = np.pad(m, 1)
    inner = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m & ~inner


def overlay_png(volume, affine, reference, region, margin_mm: float = 25.0, pixel_mm: float = 0.5) -> bytes:
    """Two views of one lesion -- the axial slice and the coronal slice of its largest cross-section -- with the
    reference contour in green and the found region's in red (yellow where they coincide), grey levels from the
    1st to the 99th percentile of the values around the lesion (no window chosen for an organ)."""
    from melampo.imaging.density import spacing_zyx

    sp = spacing_zyx(affine)
    both = reference | region
    idx = np.argwhere(both)
    lo = np.maximum(idx.min(0) - np.ceil(margin_mm / sp).astype(int), 0)
    hi = np.minimum(idx.max(0) + np.ceil(margin_mm / sp).astype(int) + 1, both.shape)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    img, ref, reg = np.asarray(volume[box], dtype=np.float32), reference[box], region[box]
    vlo, vhi = np.percentile(img, [1, 99])
    gray = np.clip((img - vlo) / max(vhi - vlo, 1e-6), 0, 1)
    z = int(np.argmax((ref | reg).sum((1, 2))))
    y = int(np.argmax((ref | reg).sum((0, 2))))
    views = []
    for name, sl, step in (("axial", (z, slice(None), slice(None)), (sp[1], sp[2])),
                           ("coronal", (slice(None), y, slice(None)), (sp[0], sp[2]))):
        g, a, b = gray[sl], ref[sl], reg[sl]
        ry, rx = max(1, round(step[0] / pixel_mm)), max(1, round(step[1] / pixel_mm))
        rgb = np.repeat((np.stack([g, g, g], -1) * 255).astype(np.uint8), 1, axis=2)
        ea, eb = _edge2d(a), _edge2d(b)
        rgb[ea] = (0, 255, 0)
        rgb[eb] = (255, 0, 0)
        rgb[ea & eb] = (255, 255, 0)
        views.append(np.repeat(np.repeat(rgb, ry, axis=0), rx, axis=1))
    h = max(v.shape[0] for v in views)
    pad = [np.pad(v, ((0, h - v.shape[0]), (0, 4), (0, 0))) for v in views]
    return _png(np.concatenate(pad, axis=1))


def _radiologist_summary(volume, main, ref_main, affine, modality, host, structures) -> dict:
    """The lesion read as a radiologist reads it (window, solid part, total with its ground glass) beside the
    segmenter's region, each against the reference contour: which edge does the reference follow?"""
    from melampo.imaging.radiologist_reading import radiologist_readings, readings_summary

    try:
        r = radiologist_readings(volume, main, affine, modality, host, structures)
    except Exception as error:  # noqa: BLE001 -- a diagnostic failing must not lose the measures
        return {"error": f"{type(error).__name__}: {error}"[:300]}
    out = readings_summary(r)
    ref = measure_region(ref_main, affine) if ref_main.any() else None
    seg = measure_region(main, affine)
    out["reference"] = None if ref is None else {"axial_long_mm": ref.longest_axial_diameter_mm,
                                                 "axial_short_mm": ref.axial_short_mm, "axial_mean_mm": ref.axial_mean_mm}
    out["segmenter"] = {"axial_long_mm": seg.longest_axial_diameter_mm, "axial_short_mm": seg.axial_short_mm,
                        "axial_mean_mm": seg.axial_mean_mm, "dice": dice(main, ref_main)}
    for k in ("window", "window_band", "solid", "total"):
        if r.get(k) is not None and r[k].get("mask") is not None:
            out[k]["dice"] = dice(r[k]["mask"], ref_main)
    return out


def _anatomy_summary(region, affine, structures, density=None) -> dict:
    """The structures a region involves, its position in its host and its margins (18 Oct 2026)."""
    from melampo.imaging.involvement import involvement
    from melampo.imaging.margins import margin_profile

    got, pos = involvement(region, structures, affine) if structures else ([], None)
    return {"involvement": [i.as_dict() for i in got[:8]], "position": pos.as_dict() if pos else None,
            "margins": margin_profile(region, affine, density).as_dict()}


def probe_case(case: dict, lesion_seg, organ_seg, refiner=None, diagnose: bool = False,
               anatomy: bool = False, overlays: Path | None = None, overlays_all: bool = False,
               save_masks: Path | None = None) -> dict:
    volume, affine = load_case(case["image"])
    reference = read_nifti_mask(Path(case["lesion"]))
    modality = case.get("modality", "CT")
    names = case["structure"] if isinstance(case["structure"], list) else [case["structure"]]
    row = {"id": case["id"], "structure": case["structure"], "modality": modality, "slices": int(volume.shape[0])}
    for seg_obj in {id(lesion_seg): lesion_seg, id(organ_seg): organ_seg}.values():
        if hasattr(seg_obj, "set_case"):   # a replay of a saved GPU run
            seg_obj.set_case(case["id"])
    # one class or several (the lobes of a lung): the organ check uses their union
    organ, parts, named = None, [], {}
    structures = organ_seg.all_structures(volume, affine) if anatomy else None
    if structures is not None:
        row["atlas"] = {"structures": len(structures), "regions": sorted(getattr(structures, "regions", {}) or {}),
                        "notes": list(getattr(structures, "notes", ()) or ())}
    if structures is not None:   # one whole-body run gives the named structures too
        for name in names:
            if name in structures:
                parts.append(structures[name])
                organ = parts[-1] if organ is None else (organ | parts[-1])
    else:
        for name, got in zip(names, organ_seg.segment(volume, affine, names)):
            named[name] = got.mask
            if got.mask is not None:
                parts.append(got.mask)
                organ = got.mask if organ is None else (organ | got.mask)
    ref_main, _ = split_main_part(reference, affine) if reference.any() else (reference, [])
    ref_size = measure_region(ref_main, affine).longest_axial_diameter_mm if reference.any() else None
    clause = case.get("report") or synthetic_clause(ref_size)
    sizes = fd.sizes_in_text(clause)
    text_size = max((s["largest_mm"] for s in sizes), default=None)
    character = case.get("report_character", [])
    row.update(report_clause=clause, report_is_synthetic=not case.get("report"), reference_longest_axial_mm=ref_size)
    if reference.any():
        ref_density = density_profile(volume, ref_main, affine, modality, organ)
        row["reference_density"] = _density_summary(ref_density)
        if anatomy:   # the same reading on the reference contour: what the image reading should find
            row["reference_anatomy"] = _anatomy_summary(ref_main, affine, structures, ref_density)
    keys = [k for k in ("prompt", "prompt_en") if case.get(k)]
    t0 = time.time()
    segs = lesion_seg.segment(volume, affine, [case[k] for k in keys])   # one run for both prompts
    row["seconds"] = round(time.time() - t0, 1)
    if save_masks is not None:
        try:
            row["saved_masks"] = save_case_masks(save_masks, case["id"], [case[k] for k in keys], segs,
                                                 structures, named).name
        except Exception as error:  # noqa: BLE001 -- saving must not lose the measures
            row["save_masks_error"] = f"{type(error).__name__}: {error}"[:300]
    for key, seg in zip(keys, segs):
        ev = fd.image_evidence(seg, affine, volume, organ, text_size, modality=modality, structure=names,
                               character=character, contrast=case.get("contrast", "unknown"), organ_parts=parts,
                               structures=structures)
        got = seg.mask if seg.mask is not None else np.zeros_like(reference)
        main = split_main_part(got, affine)[0] if got.any() else got
        ext, ext_d = tc.ground_glass_extent(main, volume, affine, modality, organ) if main.any() else (None, {})
        measured = ev.measures.longest_axial_diameter_mm if ev.measures else None
        err = measured - ref_size if measured is not None and ref_size else None
        second_dice = None
        if ev.second.get("status") == tc.APPLIED and main.any():
            region, _ = tc.threshold_region(volume, main, affine, organ)
            second_dice = dice(region, reference) if region is not None else 0.0
        row[key] = {"status": ev.status, "needs_review": ev.needs_review, "review": list(ev.review),
                    "character_differences": list(ev.character_differences),
                    "measure_support": list(ev.measure_support) or ["single_reading"],
                    "dice": dice(got, reference), "dice_main_part": dice(main, reference),
                    "dice_second_reading": second_dice, "segmenter_notes": list(seg.notes),
                    "size_check": ev.size_check, "in_organ_share": ev.in_organ_share, "parts": ev.components,
                    "other_regions": ev.other_regions, "fragments": list(ev.fragments)[:5],
                    "measured_longest_axial_mm": ev.measures.longest_axial_diameter_mm if ev.measures else None,
                    "whole_region_longest_axial_mm": ev.whole_axial_mm, "second_reading": ev.second,
                    "density": _density_summary(ev.density), "text_size_mm": text_size,
                    "image_notes": [n for n in ev.notes if n != "api"],
                    "measured_on": ev.measured_on, "with_ground_glass_axial_mm": ev.with_ground_glass_axial_mm,
                    "dice_with_ground_glass": dice(ext, reference) if ext is not None else None,
                    "ground_glass_details": {k: v for k, v in ext_d.items() if k != "notes"} | {"notes": list(ext_d.get("notes", ()))},
                    # against the reference contour: the probe's own truth, never seen by the checks
                    "reference_error_mm": err,
                    "within_strict": None if err is None else abs(err) <= max(2.0, 0.10 * ref_size)}
        if anatomy and key == keys[0] and main.any():
            row[key]["radiologist"] = _radiologist_summary(volume, main, ref_main, affine, modality,
                                                           (ev.position or {}).get("host") or names[0], structures)
        if anatomy:
            row[key]["anatomy"] = {"involvement": list(ev.involvement)[:8], "position": ev.position or None,
                                   "margins": ev.margins.as_dict() if ev.margins is not None else None}
        if overlays is not None and key == keys[0] and main.any() and reference.any() \
                and (overlays_all or not row[key]["within_strict"]):
            try:   # the lesion as the reference and the found region draw it, where the measure is off
                overlays.mkdir(parents=True, exist_ok=True)
                target = overlays / f"{case['id']}.png"
                target.write_bytes(overlay_png(volume, affine, ref_main, main))
                row[key]["overlay"] = target.name
            except Exception as error:  # noqa: BLE001 -- a picture failing must not lose the measures
                row[key]["overlay_error"] = f"{type(error).__name__}: {error}"[:200]
        if diagnose and key == keys[0]:
            row[key]["size_diagnostics"] = size_diagnostics(volume, affine, reference, main, got, lesion_seg, case[key],
                                                            refiner)
    return row


def run(cases: list[dict], device: str, fast: bool, out_path: Path | None = None, diagnose: bool = False,
        refiner: bool = False, anatomy: bool = False, overlays: Path | None = None,
        overlays_all: bool = False, save_masks: Path | None = None, from_masks: Path | None = None) -> list[dict]:
    if from_masks is not None:   # the regions of a saved GPU run: no segmenter runs, no GPU
        lesion_seg = organ_seg = ReplaySegmenters(from_masks)
        if diagnose or refiner:
            print("replay: --diagnose-size and --nninteractive need the segmenters, skipped", flush=True)
        diagnose, refiner = False, False
    else:
        lesion_seg = VoxTellSegmenter(device=device)
        organ_seg = TotalSegmentatorSegmenter(fast=fast, device="gpu" if device == "cuda" else "cpu")
    ref_obj = None
    if refiner:
        from melampo.imaging.segmenters import NnInteractiveRefiner
        ref_obj = NnInteractiveRefiner(device=device)
    out = []
    for case in cases:
        try:
            row = probe_case(case, lesion_seg, organ_seg, ref_obj, diagnose, anatomy, overlays, overlays_all,
                             save_masks)
        except Exception as error:  # noqa: BLE001 -- one case failing must not lose the others
            row = {"id": case.get("id"), "error": f"{type(error).__name__}: {error}"[:500]}
        out.append(row)
        print(json.dumps(row, default=float), flush=True)
        if out_path is not None:   # written after every case: a crash keeps what was measured
            out_path.write_text(json.dumps(out, indent=1, default=float))
        _free_memory()
    return out


def run_isolated(cases: list[dict], args, out_path: Path) -> list[dict]:
    """One process per case. The sixth and seventh probes died silently after 21 and 15 cases (no traceback, the
    log ends inside a progress bar: the host ran out of memory, which VoxTell, TotalSegmentator and nnInteractive
    keep across cases). A process per case gives the memory back each time, and a case that is killed becomes a
    row with the exit code instead of ending the whole run."""
    import subprocess
    import tempfile

    out: list[dict] = []
    for case in cases:
        with tempfile.TemporaryDirectory() as tmp:
            one, res = Path(tmp) / "case.json", Path(tmp) / "row.json"
            one.write_text(json.dumps([case]))
            cmd = [sys.executable, str(Path(__file__).resolve()), "--cases", str(one), "--out", str(res),
                   "--device", args.device]
            cmd += [f for f, on in (("--fast-organs", args.fast_organs), ("--diagnose-size", args.diagnose_size),
                                    ("--nninteractive", args.nninteractive), ("--anatomy", getattr(args, "anatomy", False))) if on]
            for flag, attr in (("--save-masks", "save_masks"), ("--from-masks", "from_masks")):
                if getattr(args, attr, None):
                    cmd += [flag, getattr(args, attr)]
            if getattr(args, "overlays", None):
                cmd += ["--overlays", args.overlays] + (["--overlays-all"] if getattr(args, "overlays_all", False) else [])
            done = subprocess.run(cmd, check=False)
            try:
                row = json.loads(res.read_text())[0]
            except Exception:  # noqa: BLE001 -- killed before writing: say so
                row = {"id": case.get("id"), "error": f"process ended with code {done.returncode} "
                       "(killed: -9 is the system out of memory)"}
        out.append(row)
        print(json.dumps(row, default=float), flush=True)
        out_path.write_text(json.dumps(out, indent=1, default=float))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fast-organs", action="store_true")
    ap.add_argument("--self-test", action="store_true", help="run the checks with the reference masks only")
    ap.add_argument("--diagnose-size", action="store_true", help="measure the candidate causes of short measures")
    ap.add_argument("--nninteractive", action="store_true", help="also refine with nnInteractive (diagnosis)")
    ap.add_argument("--isolate", action="store_true", help="one process per case (memory is given back each time)")
    ap.add_argument("--overlays", metavar="DIR", help="a PNG per case whose measure is off (axial + coronal, "
                    "reference green, found region red) written in DIR")
    ap.add_argument("--overlays-all", action="store_true", help="a PNG for every case, not only the off ones")
    ap.add_argument("--save-masks", metavar="DIR", help="save each case's regions and structures (npz) in DIR, "
                    "to run the analysis again later on a CPU (--from-masks)")
    ap.add_argument("--from-masks", metavar="DIR", help="no segmenter: read the regions and structures saved "
                    "by a run with --save-masks (CPU only)")
    ap.add_argument("--anatomy", action="store_true",
                    help="whole-body structures: the zones each lesion involves, its position and its margins")
    args = ap.parse_args(argv)
    cases = json.loads(Path(args.cases).read_text())
    if args.self_test:
        rows = []
        for case in cases:
            volume, affine = load_case(case["image"])
            ref = read_nifti_mask(Path(case["lesion"]))
            seg = MaskSegmenter({"p": ref}).segment(volume, affine, ["p"])[0]
            rows.append({"id": case["id"], "status": fd.image_evidence(seg, affine, volume).status})
    elif args.isolate and len(cases) > 1:
        rows = run_isolated(cases, args, Path(args.out))
    else:
        rows = run(cases, args.device, args.fast_organs, Path(args.out), args.diagnose_size, args.nninteractive,
                   args.anatomy, Path(args.overlays) if args.overlays else None, args.overlays_all,
                   Path(args.save_masks) if args.save_masks else None,
                   Path(args.from_masks) if args.from_masks else None)
    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
