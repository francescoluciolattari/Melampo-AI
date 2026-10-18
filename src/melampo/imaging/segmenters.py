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

    def probabilities(self, volume, affine, prompts) -> list[np.ndarray]:
        """The voxel probabilities (sigmoid of the logits) per prompt, on the series' grid, in [0, 1] with 1/255
        steps -- before the 0.5 cut that ``segment`` applies. For the measurement of how the cut shapes the
        region (fifth probe: under-measures growing with the lesion's size)."""
        from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
        import nibabel as nib
        import torch

        predictor = self._api()
        key = tuple(prompts)
        if key not in self._embeddings:
            self._embeddings[key] = predictor.embed_text_prompts(list(prompts))
        io = NibabelIOWithReorient()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "case.nii.gz"
            write_nifti(volume, affine, src)
            img, props = io.read_images([str(src)])
            data, bbox, shape = predictor.preprocess(img)
            with torch.no_grad():
                logits = predictor.predict_sliding_window_return_logits(data, self._embeddings[key])
                prob = (torch.sigmoid(logits.float()) * 255).round().to(torch.uint8).cpu().numpy()
            del logits
            full = insert_crop_into_image(np.zeros([prob.shape[0], *shape], dtype=np.uint8), prob, bbox)
            out = []
            for i in range(len(prompts)):
                f = Path(tmp) / f"prob_{i}.nii.gz"
                io.write_seg(full[i], str(f), props)            # back to the input's orientation, values kept
                data_i = np.asarray(nib.load(str(f)).dataobj, dtype=np.float32) / 255.0
                out.append(data_i.transpose(2, 1, 0))
            return out

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


def resample_to(volume: np.ndarray, affine, spacing_mm: float, order: int = 1):
    """The volume on an isotropic grid of ``spacing_mm`` (VoxTell does not resample; its README suggests a
    typical clinical spacing such as 1.5 mm when the native one is unusual). Returns (volume, affine, factors)."""
    from scipy.ndimage import zoom

    a = np.asarray(affine, dtype=float)
    sp = np.array([np.linalg.norm(a[:3, 2]), np.linalg.norm(a[:3, 1]), np.linalg.norm(a[:3, 0])])  # z, y, x
    factors = sp / spacing_mm
    out = zoom(np.asarray(volume, dtype=np.float32), factors, order=order)
    new = a.copy()
    for col, f in zip((2, 1, 0), factors):
        new[:3, col] = a[:3, col] / f
    return out, new, factors


def mask_back(mask: np.ndarray, shape) -> np.ndarray:
    """A mask from a resampled grid back onto the original one (linear, cut at 0.5; exact shape)."""
    from scipy.ndimage import zoom

    f = np.array(shape, dtype=float) / np.array(mask.shape, dtype=float)
    back = zoom(mask.astype(np.float32), f, order=1) > 0.5
    out = np.zeros(shape, dtype=bool)
    sl = tuple(slice(0, min(a, b)) for a, b in zip(shape, back.shape))
    out[sl] = back[sl]
    return out


class NnInteractiveRefiner:
    """nnInteractive (MIC-DKFZ, 2025; code Apache-2.0, weights CC BY-NC-SA 4.0): a second, different network
    that refines a region from a prompt. Given the first region as the initial segmentation and a 2D box on its
    largest axial slice (enlarged by ``box_margin``), it returns its own 3D region -- an independent reader of the
    lesion's extent, not a threshold."""

    name = "nninteractive"

    def __init__(self, device: str = "cuda", box_margin: float = 0.2):
        self.device, self.box_margin = device, box_margin
        self._session = None

    def _get(self):
        if self._session is None:
            import torch
            from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
            from nnInteractive.model_management import ensure_model_available, get_default_model_id

            session = nnInteractiveInferenceSession(device=torch.device("cuda:0" if self.device == "cuda" else "cpu"),
                                                    use_torch_compile=False, verbose=False, do_autozoom=True)
            session.initialize_from_trained_model_folder(str(ensure_model_available(get_default_model_id())))
            self._session = session
        return self._session

    def refine(self, volume: np.ndarray, region: np.ndarray, use_initial: bool = True) -> np.ndarray:
        import torch

        session = self._get()
        # transposed views are not C-contiguous and nnInteractive refuses them (fifth probe: 3 cases failed)
        img = np.ascontiguousarray(np.asarray(volume, dtype=np.float32).transpose(2, 1, 0)[None])   # (1, x, y, z)
        reg = np.ascontiguousarray(np.asarray(region, dtype=bool).transpose(2, 1, 0))
        session.set_image(img)
        target = torch.zeros(img.shape[1:], dtype=torch.uint8)
        session.set_target_buffer(target)
        if use_initial:
            session.add_initial_seg_interaction(reg.astype(np.uint8), run_prediction=False)
        z = int(np.argmax(reg.sum(axis=(0, 1))))
        xs, ys = np.where(reg[:, :, z])
        mx, my = self.box_margin * (xs.max() - xs.min() + 1), self.box_margin * (ys.max() - ys.min() + 1)
        box = [[max(0, int(xs.min() - mx)), min(img.shape[1], int(xs.max() + mx) + 1)],
               [max(0, int(ys.min() - my)), min(img.shape[2], int(ys.max() + my) + 1)], [z, z + 1]]
        session.add_bbox_interaction(box, include_interaction=True)
        out = np.asarray(session.target_buffer.cpu().numpy() if hasattr(session.target_buffer, "cpu")
                         else session.target_buffer) > 0
        return np.ascontiguousarray(out.transpose(2, 1, 0))


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
            kwargs = dict(roi_subset=list(prompts), fast=self.fast, device=self.device)
            try:   # one saving thread: the second Kaggle probe ran out of memory saving a 567-slice series
                totalsegmentator(str(src), str(out), nr_thr_saving=1, **kwargs)
            except TypeError:
                totalsegmentator(str(src), str(out), **kwargs)
            results = []
            for p in prompts:
                f = out / f"{p}.nii.gz"
                m = read_nifti_mask(f) if f.exists() else None
                results.append(Segmentation(m if m is not None and m.any() else None, self.name, self.version, p))
            return results

    def all_structures(self, volume, affine):
        """Every structure of the task ``total`` (``involvement.StructureLabels``: name -> mask on the series'
        grid, built on demand from one label map; empty structures left out), from one multi-label run: the structures a lesion involves, not only the one the report names."""
        try:
            from totalsegmentator.python_api import totalsegmentator
            from totalsegmentator.map_to_binary import class_map
        except ImportError as error:
            raise SegmenterUnavailable("TotalSegmentator not installed (pip install TotalSegmentator)") from error
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "case.nii.gz", Path(tmp) / "seg.nii.gz"
            write_nifti(volume, affine, src)
            kwargs = dict(ml=True, fast=self.fast, device=self.device)
            try:
                totalsegmentator(str(src), str(out), nr_thr_saving=1, **kwargs)
            except TypeError:
                totalsegmentator(str(src), str(out), **kwargs)
            import nibabel as nib

            labels = np.asarray(nib.load(str(out)).dataobj).astype(np.int16).transpose(2, 1, 0)
        from .involvement import StructureLabels

        return StructureLabels(labels, {int(k): v for k, v in class_map["total"].items()})


__all__ = ["MaskSegmenter", "NnInteractiveRefiner", "Segmentation", "mask_back", "resample_to", "Segmenter", "SegmenterUnavailable", "TotalSegmentatorSegmenter",
           "VoxTellSegmenter", "read_nifti_mask", "write_nifti"]
