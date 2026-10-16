"""A second, independent reading of a lesion's extent on CT: the method of the scanners' own software
(16 October 2026).

The nodule volumetry shipped with CT workstations is semi-automatic and deterministic: the radiologist clicks the
nodule, the software keeps the voxels above a density threshold connected to the click, cuts the thin vessels
with a morphological opening, and measures. In the lung this works because a solid lesion (soft tissue, about
-100..+100 HU) sits in air (about -850 HU): the border is where the density crosses a threshold, not a guess.

Here the "click" is the region VoxTell found from the report's words. Inside that region grown by a margin, the
voxels at or above ``SOLID_HU`` connected to it are the second reading. The two readings are compared
(``compare``): when they agree, the size has two independent sources; when they do not, or the threshold region
runs to the edge of the search zone (it leaked into the chest wall or the mediastinum), the finding goes to a
person. Nothing is averaged and nothing is chosen: disagreement is reported, not resolved.

Where the method holds, stated: CT, lung structures, solid lesions. A ground-glass or part-solid lesion
(-750..-300 HU) is under-read by a solid threshold, so the readings disagree and the finding goes to a person:
that is the intended outcome, not an error to tune away. Elsewhere (liver, pancreas, MR, ultrasound) a lesion is
not separated from its organ by a fixed density, and the method answers ``not_applicable``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lesion_measures import label_parts, measure_region

SOLID_HU = -400.0          # solid tissue vs aerated lung
MARGIN_MM = 6.0            # how far beyond the first reading the threshold may extend
DICE_AGREE = 0.70          # overlap of the two readings
SIZE_AGREE_MM = 2.0        # longest axial diameter: within max(2 mm, 15%) -- the order of the variability
SIZE_AGREE_SHARE = 0.15    # between radiologists measuring the same nodule
LUNG_CLASSES = frozenset({"lung", "lung_left", "lung_right", "lung_upper_lobe_left", "lung_lower_lobe_left",
                          "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"})

APPLIED, NOT_APPLICABLE = "applied", "not_applicable"


@dataclass
class SecondReading:
    status: str                                # "applied" or "not_applicable"
    method: str = f"ct_threshold_{int(SOLID_HU)}HU"
    mask: np.ndarray | None = None
    agree: bool | None = None                  # None when not applicable or nothing to compare
    dice: float | None = None
    first_axial_mm: float | None = None
    second_axial_mm: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "mask"}


def applicable(modality: str, structures, image: np.ndarray | None) -> bool:
    names = {structures} if isinstance(structures, str) else set(structures or ())
    if image is None or (modality or "").upper() != "CT" or not names & LUNG_CLASSES:
        return False
    return float(np.min(image)) < -500.0   # the values are Hounsfield units with air in the field


def _spacing(affine) -> np.ndarray:
    a = np.asarray(affine, dtype=float)[:3, :3]
    return np.array([np.linalg.norm(a[:, 2]), np.linalg.norm(a[:, 1]), np.linalg.norm(a[:, 0])])  # slice,row,col


def _shift(mask: np.ndarray, axis: int, step: int) -> np.ndarray:
    out = np.zeros_like(mask)
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    if step > 0:
        src[axis], dst[axis] = slice(0, -step), slice(step, None)
    else:
        src[axis], dst[axis] = slice(-step, None), slice(0, step)
    out[tuple(dst)] = mask[tuple(src)]
    return out


def grow(mask: np.ndarray, radius_vox) -> np.ndarray:
    """Grow the region by a number of voxels per axis (a box; no wrap-around at the edges)."""
    out = np.asarray(mask, dtype=bool).copy()
    for axis, r in enumerate(radius_vox):
        for _ in range(int(r)):
            out = out | _shift(out, axis, 1) | _shift(out, axis, -1)
    return out


def _erode(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for axis in range(3):
        out &= _shift(mask, axis, 1) & _shift(mask, axis, -1)
    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for axis in range(3):
        out |= _shift(mask, axis, 1) | _shift(mask, axis, -1)
    return out


def threshold_region(image: np.ndarray, seed: np.ndarray, affine, threshold: float = SOLID_HU,
                     margin_mm: float = MARGIN_MM) -> tuple[np.ndarray | None, tuple[str, ...]]:
    """The voxels >= threshold connected to the seed, within the seed grown by ``margin_mm``, after a one-voxel
    opening (thin vessels leave the lesion). Returns the mask (None: nothing) and notes."""
    seed = np.asarray(seed, dtype=bool)
    notes: list[str] = []
    radius = np.ceil(margin_mm / _spacing(affine)).astype(int)
    zone = grow(seed, radius)
    dense = (np.asarray(image) >= threshold) & zone
    opened = _dilate(_erode(dense)) & dense
    if (opened & seed).any():
        dense = opened
    else:
        notes.append("opening_skipped_small_lesion")
    labels, sizes = label_parts(dense)
    if not sizes:
        return None, tuple(notes + ["nothing_above_threshold"])
    overlap = np.bincount(labels[seed].ravel(), minlength=len(sizes) + 1)[1:]
    if not overlap.any():
        return None, tuple(notes + ["nothing_above_threshold_in_the_first_reading"])
    region = labels == int(np.argmax(overlap)) + 1
    border = zone & ~_erode(zone)
    if (region & border).any():
        notes.append("reaches_the_search_zone_edge")   # leaked along the pleura, a vessel or the mediastinum
    return region, tuple(notes)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = int(a.sum()) + int(b.sum())
    return float(2 * int((a & b).sum()) / s) if s else 1.0


def second_reading(first: np.ndarray, image: np.ndarray | None, affine, modality: str, structures) -> SecondReading:
    """The threshold reading of the region ``first`` (the largest part of the segmenter's region) and whether
    the two readings agree."""
    if not applicable(modality, structures, image) or first is None or not np.any(first):
        return SecondReading(NOT_APPLICABLE)
    region, notes = threshold_region(image, first, affine)
    first_axial = measure_region(first, affine).longest_axial_diameter_mm
    if region is None:
        return SecondReading(APPLIED, agree=False, first_axial_mm=first_axial, notes=notes)
    second_axial = measure_region(region, affine).longest_axial_diameter_mm
    d = dice(first, region)
    tol = max(SIZE_AGREE_MM, SIZE_AGREE_SHARE * max(first_axial, second_axial))
    agree = d >= DICE_AGREE and abs(first_axial - second_axial) <= tol and "reaches_the_search_zone_edge" not in notes
    return SecondReading(APPLIED, mask=region, agree=agree, dice=d, first_axial_mm=first_axial,
                         second_axial_mm=second_axial, notes=notes)


__all__ = ["APPLIED", "LUNG_CLASSES", "NOT_APPLICABLE", "SOLID_HU", "SecondReading", "applicable", "dice", "grow",
           "second_reading", "threshold_region"]
