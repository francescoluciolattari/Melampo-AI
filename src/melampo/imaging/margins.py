"""The margins of a lesion as data: how sharp the border is and what shape it has (18 October 2026).

Reports describe margins with a closed set of words: netti / ben definiti / circoscritti (sharp), sfumati / mal
definiti (blurred), regolari (smooth), lobulati, spiculati, irregolari. The image gives the same two things as
numbers, measured on the region the finding kept, in any organ and modality:

* **sharpness** -- the width of the transition between the lesion and the tissue around it,
  ``transition_mm = |core - surroundings| / |border gradient|`` (``density.DensityProfile``: the core is the
  deepest quarter of the lesion, the gradient the change from the inner rim to the 0-2 mm outside shell). A
  border that goes from core to surroundings within the 2 mm the gradient is measured over reads about 2 mm;
  a halo or a blurred edge reads wider. Not judged when the lesion barely differs from its surroundings
  (contrast below ``MIN_CONTRAST_HU`` on CT, ``MIN_CONTRAST_RATIO`` elsewhere).
* **contour** -- two morphological measures in mm, so they mean the same at any voxel size:

  - ``spicule_share``: the share of the lesion an opening of radius ``SPICULE_MM`` removes, i.e. the parts
    thinner than ~2 x ``SPICULE_MM``: spicules, thin strands; ``spicules`` counts the removed parts long enough to
    be one (longest diameter >= ``SPICULE_MIN_LENGTH_MM``);
  - ``concavity_share``: the volume a closing of radius ``LOBULE_MM`` (relative to the lesion's size) adds,
    as a share of the lesion: the notches between lobes.

  Categories: ``spiculated`` (``spicules`` >= ``SPICULATED_COUNT`` and spicule_share >= ``SPICULATED_SHARE``),
  ``lobulated`` (concavity_share >= ``LOBULATED_SHARE``), ``smooth`` (no spicule and both shares small), else
  ``irregular``.

A lesion too small for its contour to be read at the scan's resolution (longest diameter below
``MIN_JUDGE_MM`` or ``MIN_JUDGE_VOXELS`` voxels) gets the numbers but no contour category.

The comparison with the report sends a finding to a person only on clear opposites, outside a grey zone between
the categories: a lesion written "spiculata" with no spicule at all, "a margini regolari" that is clearly
spiculated, "netti" with a transition wider than ``BLURRED_MM`` x 1.5, "sfumati" sharper than ``SHARP_MM`` / 1.25.

Thresholds are starting values written here, to be calibrated on annotated data -- LIDC-IDRI carries four
radiologists' ratings of margin, lobulation and spiculation per nodule -- not tuned to pass tests. Thick slices
blur the border along z and hide spicules: the slice thickness is recorded with the numbers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .density import DensityProfile, distance_mm, spacing_zyx
from .lesion_measures import label_parts, measure_region

SPICULE_MM = 1.5
SPICULE_MIN_LENGTH_MM = 3.0
LOBULE_SHARE_OF_RADIUS, LOBULE_MIN_MM = 0.35, 2.0
SPICULATED_COUNT, SPICULATED_SHARE = 3, 0.02
LOBULATED_SHARE = 0.06
SMOOTH_SPICULE_SHARE, SMOOTH_CONCAVITY_SHARE = 0.02, 0.03   # a voxelised sphere loses ~1% to the opening
SHARP_MM, BLURRED_MM = 3.0, 5.0
MIN_CONTRAST_HU, MIN_CONTRAST_RATIO = 60.0, 0.20
MIN_JUDGE_MM, MIN_JUDGE_VOXELS = 8.0, 6


@dataclass
class MarginProfile:
    transition_mm: float | None = None
    sharpness: str = ""             # sharp / blurred / "" (grey zone or not judged)
    spicule_share: float | None = None
    spicules: int | None = None
    concavity_share: float | None = None
    contour: str = ""               # spiculated / lobulated / smooth / irregular / "" (too small to judge)
    slice_mm: float = 0.0
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else round(v, 4) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def _box(mask: np.ndarray, margin_vox) -> tuple[slice, ...]:
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - margin_vox, 0)
    hi = np.minimum(idx.max(0) + margin_vox + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _open(m: np.ndarray, sp, r: float) -> np.ndarray:
    """Euclidean opening of radius r mm: the voxels within r of the voxels deeper than r."""
    inside, _ = distance_mm(m, sp)
    core = inside > r
    if not core.any():
        return core
    near, _ = distance_mm(~core, sp)
    return core | (near <= r)


def _close(m: np.ndarray, sp, r: float) -> np.ndarray:
    """Euclidean closing of radius r mm: the voxels farther than r from what lies more than r outside."""
    out, _ = distance_mm(~m, sp)
    grown = m | (out <= r)
    if grown.all():
        return grown
    depth, _ = distance_mm(grown, sp)                     # depth of each grown voxel below its surface
    return grown & (depth > r)


def margin_profile(region: np.ndarray, affine, density: DensityProfile | None = None) -> MarginProfile:
    region = np.asarray(region, dtype=bool)
    sp = spacing_zyx(affine)
    out = MarginProfile(slice_mm=float(sp[0]))
    notes: list[str] = []
    if not region.any():
        return out

    # sharpness, from the density profile
    if density is not None and density.border_gradient and density.surrounding is not None:
        core = next((row for row in reversed(density.depth_profile) if row.get("n")), None)
        if core is not None:
            contrast = core["mean"] - density.surrounding
            hu = density.units == "HU"
            enough = abs(contrast) >= (MIN_CONTRAST_HU if hu else MIN_CONTRAST_RATIO * abs(density.surrounding))
            if enough and density.border_gradient * contrast > 0:
                out.transition_mm = float(abs(contrast / density.border_gradient))
                out.sharpness = ("sharp" if out.transition_mm <= SHARP_MM else
                                 "blurred" if out.transition_mm >= BLURRED_MM else "")
            else:
                notes.append("sharpness_not_judged:low_contrast")

    # contour, on a crop with room for the closing
    m_all = measure_region(region, affine)
    size = m_all.longest_diameter_mm or 0.0
    r_lobe = max(LOBULE_MIN_MM, LOBULE_SHARE_OF_RADIUS * size / 2.0)
    box = _box(region, np.ceil((r_lobe + 2 * max(sp)) / sp).astype(int) + 1)
    m = region[box]
    n = float(m.sum())
    opened = _open(m, sp, SPICULE_MM)
    removed = m & ~opened
    out.spicule_share = float(removed.sum()) / n
    count = 0
    if removed.any():
        labels, sizes = label_parts(removed)
        shifted = np.asarray(affine, dtype=float).copy()
        shifted[:3, 3] = shifted[:3, :3] @ np.array([box[2].start, box[1].start, box[0].start]) + shifted[:3, 3]
        for k, s in enumerate(sizes, start=1):
            if s < 2:
                break
            if (measure_region(labels == k, shifted).longest_diameter_mm or 0.0) >= SPICULE_MIN_LENGTH_MM:
                count += 1
    out.spicules = count
    closed = _close(m, sp, r_lobe)
    out.concavity_share = float((closed & ~m).sum()) / n

    if size < max(MIN_JUDGE_MM, MIN_JUDGE_VOXELS * float(max(sp))):
        notes.append(f"contour_not_judged:size_{size:.1f}mm")
    elif count >= SPICULATED_COUNT and out.spicule_share >= SPICULATED_SHARE:
        out.contour = "spiculated"
    elif out.concavity_share >= LOBULATED_SHARE:
        out.contour = "lobulated"
    elif count == 0 and out.spicule_share < SMOOTH_SPICULE_SHARE and out.concavity_share < SMOOTH_CONCAVITY_SHARE:
        out.contour = "smooth"
    else:
        out.contour = "irregular"
    out.notes = tuple(notes)
    return out


def compare_margins_with_report(character: list[str] | tuple[str, ...], margins: MarginProfile | None) -> list[str]:
    """The report's words on the margins (``margin:<kind>``) the image clearly contradicts."""
    if margins is None:
        return []
    out = []
    judged = bool(margins.contour)
    for value in character or ():
        quality, _, kind = value.partition(":")
        if quality != "margin":
            continue
        neg = kind.startswith("not_")
        kind = kind.removeprefix("not_")
        clash = False
        if kind == "spiculated" and judged:
            none_at_all = margins.spicules == 0 and (margins.spicule_share or 0.0) < SMOOTH_SPICULE_SHARE
            clearly = (margins.spicules or 0) >= 2 * SPICULATED_COUNT and \
                (margins.spicule_share or 0.0) >= 2 * SPICULATED_SHARE
            clash = clearly if neg else none_at_all
        elif kind == "smooth" and judged and not neg:
            clash = (margins.spicules or 0) >= 2 * SPICULATED_COUNT and \
                (margins.spicule_share or 0.0) >= 2 * SPICULATED_SHARE
        elif kind == "sharp" and margins.transition_mm is not None and not neg:
            clash = margins.transition_mm >= 1.5 * BLURRED_MM
        elif kind == "blurred" and margins.transition_mm is not None and not neg:
            clash = margins.transition_mm <= SHARP_MM / 1.25
        if clash:
            out.append(f"{value}~image:{margins.contour or 'contour_not_judged'},"
                       f"transition_{margins.transition_mm if margins.transition_mm is not None else float('nan'):.1f}mm,"
                       f"spicules_{margins.spicules}")
    return out


__all__ = ["MarginProfile", "compare_margins_with_report", "margin_profile"]
