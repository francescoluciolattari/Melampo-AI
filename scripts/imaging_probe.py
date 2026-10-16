#!/usr/bin/env python
"""Measure the imaging interpreter on public CT/MR with reference lesion masks (16 October 2026).

For each case: the image (NIfTI), the reference lesion mask (NIfTI, same grid), the structure the lesion is in
(a TotalSegmentator class) and a prompt (the words a report would use). VoxTell isolates the region from the
prompt, TotalSegmentator gives the organ, ``findings.image_evidence`` runs the checks of decision D19, and the
probe writes per case: Dice and surface distance against the reference, the status (agree / disagree /
not_found / ambiguous), the measured vs reference size and volume, the run time and the device.

Where it runs: Kaggle (free GPU, public data only) or a CPU runner for one or two small cases. Public data only:
pseudonymised patient data is still personal data and these services have no data-processing agreement.

    python scripts/imaging_probe.py --cases cases.json --out probe.json [--device cuda|cpu] [--fast-organs]

``cases.json``: [{"id": "msd_liver_001", "image": "...nii.gz", "lesion": "...nii.gz", "structure": "liver",
"prompt": "lesione ipodensa nel fegato", "prompt_en": "hypodense liver lesion"}]
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
from melampo.imaging.lesion_measures import measure_region  # noqa: E402
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
    return np.asarray(img.get_fdata()).transpose(2, 1, 0), lps


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0


def run(cases: list[dict], device: str, fast: bool) -> list[dict]:
    lesion_seg = VoxTellSegmenter(device=device)
    organ_seg = TotalSegmentatorSegmenter(fast=fast, device="gpu" if device == "cuda" else "cpu")
    out = []
    for case in cases:
        volume, affine = load_case(case["image"])
        reference = read_nifti_mask(Path(case["lesion"]))
        row = {"id": case["id"], "structure": case["structure"]}
        # one class or several (the lobes of a lung): the organ check uses their union
        names = case["structure"] if isinstance(case["structure"], list) else [case["structure"]]
        organ = None
        for got in organ_seg.segment(volume, affine, names):
            if got.mask is not None:
                organ = got.mask if organ is None else (organ | got.mask)
        for key in ("prompt", "prompt_en"):
            if not case.get(key):
                continue
            t0 = time.time()
            seg = lesion_seg.segment(volume, affine, [case[key]])[0]
            seconds = time.time() - t0
            ref_size = measure_region(reference, affine).longest_axial_diameter_mm if reference.any() else None
            ev = fd.image_evidence(seg, affine, volume, organ, ref_size)
            got = seg.mask if seg.mask is not None else np.zeros_like(reference)
            row[key] = {"status": ev.status, "dice": dice(got, reference), "seconds": round(seconds, 1),
                        "size_check": ev.size_check, "in_organ_share": ev.in_organ_share,
                        "measured_longest_axial_mm": ev.measures.longest_axial_diameter_mm if ev.measures else None,
                        "reference_longest_axial_mm": ref_size}
        out.append(row)
        print(json.dumps(row), flush=True)
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
        rows = run(cases, args.device, args.fast_organs)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
