"""A second, independent reading of a lesion's extent: the half-way threshold of volumetry software
(16 October 2026, revised after the second Kaggle probe).

Volumetry software on CT and PET workstations is semi-automatic and deterministic: from a click on the lesion
it keeps the voxels on the lesion's side of a threshold, connected to the click, and measures. The first version
here used a fixed -400 HU: it only fits a solid lesion in aerated lung, and on 5 real MSD lung tumours it ran
along vessels and the pleura every time (all five reached the edge of the search zone).

Now the threshold comes from the lesion itself, as PET volumetry does with a share of the maximum:

* the "click" is the region the segmenter found from the report's words (its largest part);
* ``core``: the median value in the deeper half of that region; ``surrounding``: the median of the tissue
  2-10 mm around it (inside the organ when the organ mask is given); the threshold is half-way between them, on
  the lesion's side (above it for a lesion denser than its surroundings, below for a cyst in the liver);
* the search is limited to the region grown by ``MARGIN_MM`` and, when the organ mask is given, to the organ
  (grown by two voxels) plus the region: a tumour touching the chest wall does not run into the wall;
* a one-voxel opening detaches thin vessels; the connected part overlapping the click is kept.

It applies wherever the lesion differs from its surroundings by more than the noise (contrast-to-noise >=
``MIN_CNR``): CT in any organ, MR and PET too, since half-way between two values of the same image does not need
absolute units. A ground-glass nodule (-600 HU in -850 HU lung) has its own half-way threshold and is read like
a solid one. Where there is no contrast it says ``not_applicable``.

The two readings agree when they overlap (Dice >= 0.70) and their longest axial diameters differ by no more than
max(2 mm, 15%). The outcome is recorded as how well the measure is supported (``measure_support`` of the
finding); it does not send a finding to a person by itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lesion_measures import label_parts, measure_region

MARGIN_MM = 6.0            # how far beyond the first reading the threshold may extend
OPEN_MM = 2.0              # in-plane opening radius: vessels thinner than ~4 mm are cut from the lesion
GGO_MARGIN_MM = 10.0       # how far a ground-glass component may extend beyond the first reading
GGO_STEP_HU = 100.0        # ground glass: at least this much (and 3 x noise) above the aerated lung around
MIN_CNR = 2.5              # |core - surrounding| / noise of the surrounding
DICE_AGREE = 0.70
SIZE_AGREE_MM = 2.0        # longest axial diameter: within max(2 mm, 15%), the order of the variability
SIZE_AGREE_SHARE = 0.15    # between radiologists measuring the same lesion
VOLUME_MODALITIES = frozenset({"CT", "MR", "PT", "PET"})

APPLIED, NOT_APPLICABLE = "applied", "not_applicable"


@dataclass
class SecondReading:
    status: str                                # "applied" or "not_applicable"
    method: str = "half_way_threshold"
    mask: np.ndarray | None = None
    agree: bool | None = None
    dice: float | None = None
    threshold: float | None = None
    core: float | None = None
    surrounding: float | None = None
    cnr: float | None = None
    first_axial_mm: float | None = None
    second_axial_mm: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "mask"}


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


def _open_inplane(mask: np.ndarray, radius_vox: tuple[int, int]) -> np.ndarray:
    """Opening within each slice (rows and columns only): slices are often thicker than the pixels, and a vessel
    runs through the slices as easily as along them."""
    out = mask.copy()
    for axis, r in ((1, radius_vox[0]), (2, radius_vox[1])):
        for _ in range(r):
            out = out & _shift(out, axis, 1) & _shift(out, axis, -1)
    for axis, r in ((1, radius_vox[0]), (2, radius_vox[1])):
        for _ in range(r):
            out = out | _shift(out, axis, 1) | _shift(out, axis, -1)
    return out & mask


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = int(a.sum()) + int(b.sum())
    return float(2 * int((a & b).sum()) / s) if s else 1.0


def _spacing(affine) -> np.ndarray:
    a = np.asarray(affine, dtype=float)[:3, :3]
    return np.array([np.linalg.norm(a[:, 2]), np.linalg.norm(a[:, 1]), np.linalg.norm(a[:, 0])])  # slice,row,col


def _box(mask: np.ndarray, margin_vox) -> tuple[slice, ...]:
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - margin_vox, 0)
    hi = np.minimum(idx.max(0) + margin_vox + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def threshold_region(image: np.ndarray, seed: np.ndarray, affine, organ_mask: np.ndarray | None = None,
                     margin_mm: float = MARGIN_MM, mode: str = "half_way") -> tuple[np.ndarray | None, dict]:
    """The threshold region connected to ``seed``. ``half_way``: between the lesion's core and its surroundings;
    ``ground_glass``: just above aerated lung (the whole lesion, its ground-glass part included). Returns (mask
    on the full grid or None, details)."""
    from .density import distance_mm

    seed = np.asarray(seed, dtype=bool)
    sp = _spacing(affine)
    box = _box(seed, np.ceil((margin_mm + 12.0) / sp).astype(int))
    img = np.asarray(image[box], dtype=np.float32)
    s = seed[box]
    organ = None if organ_mask is None or not np.any(organ_mask) else np.asarray(organ_mask, dtype=bool)[box]
    notes: list[str] = []
    inside, _ = distance_mm(s, sp)
    deep = s & (inside >= 0.5 * inside.max())
    core = float(np.median(img[deep] if deep.sum() >= 5 else img[s]))
    outside, _ = distance_mm(~s, sp)
    shell = ~s & (outside > 2.0) & (outside <= 10.0)
    if organ is not None:
        shell &= organ
    if shell.sum() < 20:
        return None, {"notes": ("too_little_surrounding_tissue",), "core": core}
    around = img[shell]
    surrounding = float(np.median(around))
    noise = float(1.4826 * np.median(np.abs(around - surrounding))) or float(around.std()) or 1.0
    cnr = abs(core - surrounding) / noise
    details = {"core": core, "surrounding": surrounding, "cnr": cnr}
    if cnr < MIN_CNR:
        return None, {**details, "notes": ("no_density_contrast",)}
    if mode == "ground_glass":
        if surrounding >= -500.0:
            return None, {**details, "notes": ("not_in_aerated_lung",)}
        threshold = surrounding + max(GGO_STEP_HU, 3.0 * noise)
        core = max(core, threshold + 1.0)   # the lesion side is above
    else:
        threshold = (core + surrounding) / 2.0
    details["threshold"] = threshold
    zone = grow(s, np.ceil(margin_mm / sp).astype(int))
    if organ is not None:
        zone &= grow(organ, (2, 2, 2)) | s
    side = (img >= threshold) if core > surrounding else (img <= threshold)
    dense = side & zone
    r = tuple(int(x) for x in np.maximum(1, np.round(OPEN_MM / sp[1:])))
    opened = _open_inplane(dense, r)
    if not (opened & s).any():
        opened = _dilate(_erode(dense)) & dense
        notes.append("small_lesion_one_voxel_opening")
    if (opened & s).any():
        dense = opened
    else:
        notes.append("opening_skipped_small_lesion")
    labels, sizes = label_parts(dense)
    overlap = np.bincount(labels[s].ravel(), minlength=len(sizes) + 1)[1:] if sizes else np.zeros(0)
    if not sizes or not overlap.any():
        return None, {**details, "notes": tuple(notes + ["nothing_on_the_lesion_side"])}
    region = labels == int(np.argmax(overlap)) + 1
    if (region & (zone & ~_erode(zone))).any():
        notes.append("reaches_the_search_zone_edge")
    full = np.zeros(seed.shape, dtype=bool)
    full[box] = region
    return full, {**details, "notes": tuple(notes)}


def second_reading(first: np.ndarray, image: np.ndarray | None, affine, modality: str,
                   organ_mask: np.ndarray | None = None) -> SecondReading:
    """The half-way threshold reading of the region ``first`` and whether the two readings agree."""
    if image is None or first is None or not np.any(first) or (modality or "").upper() not in VOLUME_MODALITIES:
        return SecondReading(NOT_APPLICABLE)
    region, d = threshold_region(image, first, affine, organ_mask)
    first_axial = measure_region(first, affine).longest_axial_diameter_mm
    common = {"threshold": d.get("threshold"), "core": d.get("core"), "surrounding": d.get("surrounding"),
              "cnr": d.get("cnr"), "first_axial_mm": first_axial, "notes": d.get("notes", ())}
    if region is None:
        status = NOT_APPLICABLE if set(common["notes"]) & {"no_density_contrast", "too_little_surrounding_tissue"} \
            else APPLIED
        return SecondReading(status, agree=None if status == NOT_APPLICABLE else False, **common)
    second_axial = measure_region(region, affine).longest_axial_diameter_mm
    dc = dice(first, region)
    tol = max(SIZE_AGREE_MM, SIZE_AGREE_SHARE * max(first_axial, second_axial))
    agree = dc >= DICE_AGREE and abs(first_axial - second_axial) <= tol
    return SecondReading(APPLIED, mask=region, agree=agree, dice=dc, second_axial_mm=second_axial, **common)


def ground_glass_extent(first: np.ndarray, image: np.ndarray | None, affine, modality: str,
                        organ_mask: np.ndarray | None = None) -> tuple[np.ndarray | None, dict]:
    """The lesion with its ground-glass part, in aerated lung on CT.

    Third probe: on part-solid tumours VoxTell's region was the solid part, 3.7-5.1 mm short of the reference.
    Fourth probe: a threshold region with an opening found nothing more on any of them -- the opening erased the
    halo (a few millimetres thick) and a step of 150 HU above the lung left out a halo at -725 HU.

    Now the halo is grown from the region, voxel by voxel, through the ground-glass band only (values smoothed
    over 3x3x3, between ``surrounding + max(100 HU, 3 x noise of the parenchyma)`` and the solid threshold), up to
    ``GGO_MARGIN_MM``. Vessels are denser than the band, so the growth does not run along them; it stops where
    the lung is normal again."""
    from .density import SOLID_HU, _smooth3, distance_mm

    if image is None or first is None or not np.any(first) or (modality or "").upper() != "CT":
        return None, {"notes": ("not_applicable",)}
    seed = np.asarray(first, dtype=bool)
    sp = _spacing(affine)
    box = _box(seed, np.ceil((GGO_MARGIN_MM + 16.0) / sp).astype(int))
    img = np.asarray(image[box], dtype=np.float32)
    s = seed[box]
    organ = None if organ_mask is None or not np.any(organ_mask) else np.asarray(organ_mask, dtype=bool)[box]
    outside, _ = distance_mm(~s, sp)
    shell = ~s & (outside > GGO_MARGIN_MM) & (outside <= GGO_MARGIN_MM + 15.0) & (img < SOLID_HU)
    if organ is not None:
        shell &= organ
    if shell.sum() < 50:
        return None, {"notes": ("too_little_surrounding_lung",)}
    lung = img[shell]
    surrounding = float(np.median(lung))
    if surrounding >= -500.0:
        return None, {"surrounding": surrounding, "notes": ("not_in_aerated_lung",)}
    noise = float(1.4826 * np.median(np.abs(lung - surrounding))) or float(lung.std()) or 1.0
    threshold = surrounding + max(GGO_STEP_HU, 3.0 * noise)
    smooth = _smooth3(img)
    # smoothed, so one noisy voxel does not open a path; and the voxel itself not plainly lung, so the smoothing
    # does not add a ring of normal lung around the halo
    band = (smooth >= threshold) & (img >= threshold - noise) & (img < SOLID_HU)
    zone = (outside <= GGO_MARGIN_MM) | s
    if organ is not None:
        zone &= grow(organ, (1, 2, 2)) | s
    cur = s.copy()
    for _ in range(int(np.ceil(GGO_MARGIN_MM / float(min(sp))))):
        nxt = cur | (_dilate(cur) & band & zone)
        if nxt.sum() == cur.sum():
            break
        cur = nxt
    notes = []
    if (cur & ~s & (outside > GGO_MARGIN_MM - float(max(sp)))).any():
        notes.append("reaches_the_search_zone_edge")
    full = np.zeros(seed.shape, dtype=bool)
    full[box] = cur
    return full, {"threshold": threshold, "surrounding": surrounding, "noise": noise,
                  "added_voxels": int((cur & ~s).sum()), "notes": tuple(notes)}


__all__ = ["APPLIED", "NOT_APPLICABLE", "SecondReading", "dice", "ground_glass_extent", "grow", "second_reading",
           "threshold_region"]
