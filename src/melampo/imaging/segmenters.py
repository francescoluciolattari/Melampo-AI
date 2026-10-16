"""The segmenters behind one interface: the region a report names, and the organ it should be in (16 October 2026).

* ``VoxTellSegmenter`` -- free text -> 3D region (CT, MR, PET). Weights CC BY-NC-SA 4.0 (research and tests; the
  non-commercial clause is open for the product). Runs through its CLI on a NIfTI written from the series.
* ``TotalSegmentatorSegmenter`` -- the organ masks, as the check of decision D1/D19 (open tasks Apache-2.0).
* ``MaskSegmenter`` -- a mask already known (a test, a reference annotation, a physician's contour).

Two-dimensional modalities (ultrasound, radiographs) get their own provider once the weights' licence is settled
(BiomedParse: CC BY-NC-SA 4.0, gated, excluded as a medical device; SonoBase, UltraSAM3: licence not stated).

Nothing here decides anything clinical: a segmenter returns a mask or nothing, and ``findings.image_evidence``
checks it against the organ and the size written in the report.

Geometry: a series' array is (slice, row, column) with an affine (column, row, slice) -> LPS mm
(``dicom_volume``); NIfTI wants (i, j, k) = (column, row, slice) and an affine to RAS. ``write_nifti`` and
``read_nifti_mask`` convert both ways, so a mask comes back on the series' own grid.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from .lesion_measures import lps_to_ras_affine


class SegmenterUnavailable(RuntimeError):
    """The segmenter is not installed or cannot run here (no weights, no GPU and no CPU fallback)."""


@dataclass
class Segmentation:
    mask: np.ndarray | None          # (slice, row, column) bool, on the series' grid; None: nothing found
    provider: str
    version: str = ""
    prompt: str = ""
    notes: tuple[str, ...] = ()
    extra: dict = field(default_factory=dict)


class Segmenter(Protocol):
    name: str

    def segment(self, volume: np.ndarray, affine, prompts: list[str]) -> list[Segmentation]: ...


def write_nifti(volume_zyx: np.ndarray, affine_lps, path: Path) -> None:
    import nibabel as nib  # optional dependency of the imaging extra

    data = np.ascontiguousarray(np.asarray(volume_zyx).transpose(2, 1, 0))
    nib.save(nib.Nifti1Image(data, lps_to_ras_affine(affine_lps)), str(path))


def read_nifti_mask(path: Path) -> np.ndarray:
    import nibabel as nib

    data = np.asarray(nib.load(str(path)).get_fdata()) > 0.5
    return data.transpose(2, 1, 0)


class MaskSegmenter:
    """Returns the masks it was given, by prompt (tests, reference annotations, a physician's contour)."""

    name = "given_mask"

    def __init__(self, masks: dict[str, np.ndarray | None], version: str = "reference"):
        self.masks, self.version = masks, version

    def segment(self, volume, affine, prompts):
        out = []
        for p in prompts:
            m = self.masks.get(p)
            out.append(Segmentation(None if m is None or not np.any(m) else np.asarray(m, bool), self.name,
                                    self.version, p))
        return out


class VoxTellSegmenter:
    """VoxTell: one mask per prompt. Through its Python API when installed (the model is loaded once and the
    prompts embedded once -- the first Kaggle probe spent 70-200 s per call reloading the model through the CLI,
    and an Italian prompt, not in VoxTell's bank of precomputed English embeddings, also loads the text encoder,
    Qwen3-Embedding-4B), else through its CLI (``voxtell-predict``)."""

    name = "voxtell"

    def __init__(self, model_dir: str | None = None, device: str = "cuda", version: str = "voxtell_v1.1",
                 use_api: bool = True):
        self.model_dir, self.device, self.version, self.use_api = model_dir, device, version, use_api
        self._predictor = None
        self._embeddings: dict[tuple[str, ...], object] = {}

    def _api(self):
        if self._predictor is None:
            import torch
            from voxtell.inference.predictor import VoxTellPredictor

            kwargs = {"device": torch.device("cuda:0" if self.device == "cuda" else "cpu")}
            if self.model_dir:
                kwargs["model_dir"] = self.model_dir
            self._predictor = VoxTellPredictor(**kwargs)
        return self._predictor

    def _segment_api(self, volume, affine, prompts):
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

        predictor = self._api()
        key = tuple(prompts)
        if key not in self._embeddings:
            self._embeddings[key] = predictor.embed_text_prompts(list(prompts))
        io = NibabelIOWithReorient()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "case.nii.gz"
            write_nifti(volume, affine, src)
            img, props = io.read_images([str(src)])
            seg = np.asarray(predictor.predict_single_image(img, text_embeddings=self._embeddings[key]))
            results = []
            for i, p in enumerate(prompts):
                out = Path(tmp) / f"mask_{i}.nii.gz"
                io.write_seg((seg[i] > 0).astype(np.uint8), str(out), props)   # back to the input's orientation
                m = read_nifti_mask(out)
                results.append(Segmentation(m if m.any() else None, self.name, self.version, p, notes=("api",)))
            return results

    def segment(self, volume, affine, prompts):
        if self.use_api:
            try:
                import voxtell.inference.predictor  # noqa: F401
                import nnunetv2.imageio.nibabel_reader_writer  # noqa: F401
            except ImportError:
                pass
            else:
                try:
                    return self._segment_api(volume, affine, prompts)
                except Exception as error:  # noqa: BLE001 -- the API changed: the CLI still works, and says so
                    self.use_api, self.api_error = False, f"api_failed:{type(error).__name__}:{error}"[:300]
        if shutil.which("voxtell-predict") is None:
            raise SegmenterUnavailable("voxtell-predict not installed (pip install voxtell)")
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "case.nii.gz", Path(tmp) / "out"
            write_nifti(volume, affine, src)
            cmd = ["voxtell-predict", "-i", str(src), "-o", str(out), "--device", self.device, "-p", *prompts]
            if self.model_dir:
                cmd += ["-m", self.model_dir]
            subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603 -- fixed executable
            files = sorted(out.rglob("*.nii*"))
            results = []
            for i, p in enumerate(prompts):
                # one file per prompt, in prompt order; a missing file is "nothing found", never an error
                f = files[i] if i < len(files) else None
                m = read_nifti_mask(f) if f else None
                notes = ("cli",) + ((self.api_error,) if getattr(self, "api_error", "") else ())
                results.append(Segmentation(m if m is not None and m.any() else None, self.name, self.version, p,
                                            notes=notes if f else notes + ("no_output_file",)))
            return results


class TotalSegmentatorSegmenter:
    """TotalSegmentator for the organ masks (``roi_subset`` of TotalSegmentator class names)."""

    name = "totalsegmentator"

    def __init__(self, fast: bool = True, device: str = "gpu", version: str = "2"):
        self.fast, self.device, self.version = fast, device, version

    def segment(self, volume, affine, prompts):
        try:
            from totalsegmentator.python_api import totalsegmentator
        except ImportError as error:
            raise SegmenterUnavailable("TotalSegmentator not installed (pip install TotalSegmentator)") from error
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "case.nii.gz", Path(tmp) / "seg"
            write_nifti(volume, affine, src)
            totalsegmentator(str(src), str(out), roi_subset=list(prompts), fast=self.fast, device=self.device)
            results = []
            for p in prompts:
                f = out / f"{p}.nii.gz"
                m = read_nifti_mask(f) if f.exists() else None
                results.append(Segmentation(m if m is not None and m.any() else None, self.name, self.version, p))
            return results


__all__ = ["MaskSegmenter", "Segmentation", "Segmenter", "SegmenterUnavailable", "TotalSegmentatorSegmenter",
           "VoxTellSegmenter", "read_nifti_mask", "write_nifti"]
