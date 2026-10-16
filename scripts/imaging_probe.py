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


def probe_case(case: dict, lesion_seg, organ_seg) -> dict:
    volume, affine = load_case(case["image"])
    reference = read_nifti_mask(Path(case["lesion"]))
    modality = case.get("modality", "CT")
    names = case["structure"] if isinstance(case["structure"], list) else [case["structure"]]
    row = {"id": case["id"], "structure": case["structure"], "modality": modality, "slices": int(volume.shape[0])}
    # one class or several (the lobes of a lung): the organ check uses their union
    organ = None
    for got in organ_seg.segment(volume, affine, names):
        if got.mask is not None:
            organ = got.mask if organ is None else (organ | got.mask)
    ref_main, _ = split_main_part(reference, affine) if reference.any() else (reference, [])
    ref_size = measure_region(ref_main, affine).longest_axial_diameter_mm if reference.any() else None
    clause = case.get("report") or synthetic_clause(ref_size)
    sizes = fd.sizes_in_text(clause)
    text_size = max((s["largest_mm"] for s in sizes), default=None)
    character = case.get("report_character", [])
    row.update(report_clause=clause, report_is_synthetic=not case.get("report"), reference_longest_axial_mm=ref_size)
    if reference.any():
        row["reference_density"] = _density_summary(density_profile(volume, ref_main, affine, modality, organ))
    keys = [k for k in ("prompt", "prompt_en") if case.get(k)]
    t0 = time.time()
    segs = lesion_seg.segment(volume, affine, [case[k] for k in keys])   # one run for both prompts
    row["seconds"] = round(time.time() - t0, 1)
    for key, seg in zip(keys, segs):
        ev = fd.image_evidence(seg, affine, volume, organ, text_size, modality=modality, structure=names,
                               character=character)
        got = seg.mask if seg.mask is not None else np.zeros_like(reference)
        main = split_main_part(got, affine)[0] if got.any() else got
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
                    "density": _density_summary(ev.density), "text_size_mm": text_size}
    return row


def run(cases: list[dict], device: str, fast: bool, out_path: Path | None = None) -> list[dict]:
    lesion_seg = VoxTellSegmenter(device=device)
    organ_seg = TotalSegmentatorSegmenter(fast=fast, device="gpu" if device == "cuda" else "cpu")
    out = []
    for case in cases:
        try:
            row = probe_case(case, lesion_seg, organ_seg)
        except Exception as error:  # noqa: BLE001 -- one case failing must not lose the others
            row = {"id": case.get("id"), "error": f"{type(error).__name__}: {error}"[:500]}
        out.append(row)
        print(json.dumps(row, default=float), flush=True)
        if out_path is not None:   # written after every case: a crash keeps what was measured
            out_path.write_text(json.dumps(out, indent=1, default=float))
        _free_memory()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fast-organs", action="store_true")
    ap.add_argument("--self-test", action="store_true", help="run the checks with the reference masks only")
    args = ap.parse_args(argv)
    cases = json.loads(Path(args.cases).read_text())
    if args.self_test:
        rows = []
        for case in cases:
            volume, affine = load_case(case["image"])
            ref = read_nifti_mask(Path(case["lesion"]))
            seg = MaskSegmenter({"p": ref}).segment(volume, affine, ["p"])[0]
            rows.append({"id": case["id"], "status": fd.image_evidence(seg, affine, volume).status})
    else:
        rows = run(cases, args.device, args.fast_organs, Path(args.out))
    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
