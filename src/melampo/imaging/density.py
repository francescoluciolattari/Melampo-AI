"""The density of a lesion as a distribution in space, not one number (16 October 2026).

Every region gets a ``DensityProfile``: how its values are distributed and how they change from the core to the
border and beyond, in the units the modality allows:

* **units**: CT values are Hounsfield units, the same scale on every scanner (water 0, air -1000), so absolute
  bands mean something; MR and ultrasound values depend on the scanner, the sequence and the gain, so only
  values *relative* to the surrounding tissue are kept (ratio); PET would be SUV when the caller gives it;
* **composition** (CT): the share of the region in each density band (air, ground glass, fat, fluid, soft
  tissue, high, calcium) and, for a lesion in aerated lung, the solid component (its share and its longest axial
  diameter, the measure Lung-RADS uses for part-solid nodules) and the label solid / part_solid / ground_glass;
  flags for calcium, fat, cavitation;
* **space**: the mean and spread of the values by depth inside the region (rim to core, as a share of the
  deepest point, so a 6 mm and a 40 mm lesion are comparable) and by distance outside it (0-2, 2-5, 5-10 mm): a
  solid core with a ground-glass halo, a necrotic or cavitated centre, a calcified core, a sharp or a blurred
  margin all show in this profile; the gradient across the border (HU/mm) is the sharpness of the margin; the
  spread between the eight octants around the centroid is how asymmetric the density is;
* **relative to the surroundings**: lesion minus surrounding tissue (in HU on CT, as a ratio elsewhere) and the
  word a report would use: hypo / iso / hyper.

The profile is also a vector (``vector``) for search and comparison across lesions, beside the shape vector.

The comparison with the report (``compare_with_report``) is what can send a finding to a person: the image
clearly says the opposite of what the report writes (a "solida" lesion that is ground glass, "iperdensa" that is
clearly hypodense, "calcifica" with no calcium). A ground-glass lesion is not a problem in itself: it is a
lesion with a density, recorded as such.

Thresholds are starting values written here, to be calibrated on annotated data, not tuned to pass tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .lesion_measures import label_parts, measure_region

BANDS_HU = (("air", -1e9, -900.0), ("ground_glass", -900.0, -300.0), ("low_intermediate", -300.0, -150.0),
            ("fat", -150.0, -30.0), ("fluid", -30.0, 20.0), ("soft_tissue", 20.0, 100.0),
            ("high", 100.0, 200.0), ("calcium", 200.0, 1e9))
SOLID_HU = -300.0              # solid component of a lung lesion
AERATED_SURROUNDING_HU = -500  # the lesion sits in aerated lung when its surroundings are below this
OUTSIDE_SHELLS_MM = ((0.0, 2.0), (2.0, 5.0), (5.0, 10.0))
DEPTH_BINS = (0.0, 0.25, 0.5, 0.75, 1.0001)   # rim -> core, as a share of the deepest point
ISO_HU = 15.0                  # lesion vs surroundings within +-15 HU: isodense
ISO_RATIO = 0.15               # MR/US: within +-15% of the surroundings: iso
SOLID_RATIO, GGO_SOLID_MM = 0.8, 2.0   # solid: solid component >= 80% of the diameter; ground glass: < 2 mm
CALCIUM_SHARE, FAT_SHARE, CAVITY_SHARE = 0.02, 0.05, 0.05   # fat and cavity: share of the interior (>= 2 mm deep)


def spacing_zyx(affine) -> np.ndarray:
    a = np.asarray(affine, dtype=float)[:3, :3]
    return np.array([np.linalg.norm(a[:, 2]), np.linalg.norm(a[:, 1]), np.linalg.norm(a[:, 0])])


def _crop(mask: np.ndarray, margin_vox) -> tuple[slice, ...]:
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - margin_vox, 0)
    hi = np.minimum(idx.max(0) + margin_vox + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def distance_mm(mask: np.ndarray, spacing) -> tuple[np.ndarray, str]:
    """Distance of each voxel of ``mask`` to the nearest voxel outside it, in mm (scipy's exact Euclidean
    transform, else the distance to the nearest outside voxel bordering the region, by brute force)."""
    try:
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(mask, sampling=spacing), "euclidean"
    except ImportError:
        pass
    from .threshold_check import _dilate

    mask = np.asarray(mask, dtype=bool)
    out = np.zeros(mask.shape, dtype=float)
    targets = np.argwhere(_dilate(mask) & ~mask)          # the False voxels next to the region: the nearest
    if not len(targets) or not mask.any():                 # outside voxel of any inside voxel is one of them
        return out, "none"
    if len(targets) > 6000:
        targets = targets[np.random.default_rng(0).choice(len(targets), 6000, replace=False)]
    t = targets * np.asarray(spacing, dtype=float)
    pts = np.argwhere(mask)
    best = np.empty(len(pts))
    for at in range(0, len(pts), 2000):
        c = pts[at:at + 2000] * np.asarray(spacing, dtype=float)
        best[at:at + 2000] = np.sqrt(((c[:, None, :] - t[None, :, :]) ** 2).sum(-1)).min(1)
    out[tuple(pts.T)] = best
    return out, "nearest_border_voxel"


@dataclass
class DensityProfile:
    units: str                                   # "HU", "relative", "SUV"
    n: int
    mean: float
    sd: float
    percentiles: dict[str, float]
    entropy_bits: float
    bands: dict[str, float] = field(default_factory=dict)          # share of the region per band (HU only)
    depth_profile: list[dict[str, float]] = field(default_factory=list)    # rim -> core
    outside_profile: list[dict[str, float]] = field(default_factory=list)  # 0-2, 2-5, 5-10 mm
    border_gradient: float | None = None         # (inside rim - outside 0-2 mm) / 2 mm
    octant_spread: float | None = None           # sd of the octant means / sd of the region
    surrounding: float | None = None             # median of the tissue around (2-10 mm, in the organ if given)
    relative: str = ""                           # hypo / iso / hyper vs the surroundings
    relative_value: float | None = None          # HU difference (CT) or ratio (else)
    composition: str = ""                        # lung CT: solid / part_solid / ground_glass
    solid_share: float | None = None
    solid_axial_mm: float | None = None
    flags: tuple[str, ...] = ()                  # contains_calcium, contains_fat, cavitated
    distance_method: str = ""
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}

    def vector(self) -> list[float]:
        """Band shares, depth and outside profiles (means, relative to the surroundings), border gradient, octant
        spread: comparable across lesions of the same modality."""
        ref = self.surrounding or 0.0
        scale = 1000.0 if self.units == "HU" else max(abs(ref), 1e-6)
        v = [self.bands.get(name, 0.0) for name, _, _ in BANDS_HU] if self.units == "HU" else []
        for row in self.depth_profile + self.outside_profile:
            v.append((row.get("mean", ref) - ref) / scale if row.get("n") else 0.0)
        v.append((self.border_gradient or 0.0) / scale)
        v.append(self.octant_spread or 0.0)
        return [float(x) for x in v]


def _stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"n": 0}
    return {"n": int(values.size), "mean": float(values.mean()), "sd": float(values.std())}


def density_profile(image: np.ndarray, mask: np.ndarray, affine, modality: str = "CT",
                    organ_mask: np.ndarray | None = None, units: str | None = None) -> DensityProfile:
    """The density profile of the region ``mask`` of ``image`` (both (slice, row, column), same grid)."""
    mask = np.asarray(mask, dtype=bool)
    sp = spacing_zyx(affine)
    hu = units == "HU" or (units is None and (modality or "").upper() == "CT" and float(np.min(image)) < -500)
    unit = "HU" if hu else (units or "relative")
    box = _crop(mask, np.ceil(12.0 / sp).astype(int))
    img = np.asarray(image[box], dtype=np.float32)
    m = mask[box]
    organ = None if organ_mask is None else np.asarray(organ_mask, dtype=bool)[box]
    notes: list[str] = []

    vals = img[m]
    pct = {f"p{q}": float(np.percentile(vals, q)) for q in (5, 25, 50, 75, 95)}
    hist, _ = np.histogram(vals, bins=32)
    p = hist[hist > 0] / vals.size
    profile = DensityProfile(unit, int(vals.size), float(vals.mean()), float(vals.std()), pct,
                             float(-(p * np.log2(p)).sum()))

    inside, method = distance_mm(m, sp)
    outside, _ = distance_mm(~m, sp)
    profile.distance_method = method
    deepest = float(inside.max()) or 1.0
    depth = inside / deepest
    for a, b in zip(DEPTH_BINS[:-1], DEPTH_BINS[1:]):
        sel = m & (depth > a) & (depth <= b) if a > 0 else m & (depth <= b)
        profile.depth_profile.append({"from": a, "to": min(b, 1.0), **_stats(img[sel])})
    around = ~m if organ is None or not organ.any() else (~m & organ)
    for a, b in OUTSIDE_SHELLS_MM:
        sel = around & (outside > a) & (outside <= b)
        profile.outside_profile.append({"from_mm": a, "to_mm": b, **_stats(img[sel])})
    rim = profile.depth_profile[0]
    near = profile.outside_profile[0]
    if rim.get("n") and near.get("n"):
        profile.border_gradient = (rim["mean"] - near["mean"]) / 2.0
    sur = around & (outside > 2.0) & (outside <= 10.0)
    if sur.sum() >= 20:
        profile.surrounding = float(np.median(img[sur]))
    else:
        notes.append("too_little_surrounding_tissue")

    # octants around the centroid: how evenly the density is spread
    idx = np.argwhere(m)
    c = idx.mean(0)
    octant = ((idx[:, 0] > c[0]).astype(int) * 4 + (idx[:, 1] > c[1]).astype(int) * 2 + (idx[:, 2] > c[2]))
    means = [vals_o.mean() for o in range(8) if (vals_o := img[tuple(idx[octant == o].T)]).size >= 5]
    if len(means) >= 4 and profile.sd > 0:
        profile.octant_spread = float(np.std(means) / profile.sd)

    lesion_median = pct["p50"]
    if profile.surrounding is not None:
        if hu:
            diff = lesion_median - profile.surrounding
            profile.relative_value = float(diff)
            profile.relative = "iso" if abs(diff) <= ISO_HU else ("hyper" if diff > 0 else "hypo")
        elif profile.surrounding:
            ratio = lesion_median / profile.surrounding
            profile.relative_value = float(ratio)
            profile.relative = "iso" if abs(ratio - 1) <= ISO_RATIO else ("hyper" if ratio > 1 else "hypo")

    if hu:
        profile.bands = {name: float(((vals >= lo) & (vals < hi)).mean()) for name, lo, hi in BANDS_HU}
        flags = []
        if profile.bands["calcium"] >= CALCIUM_SHARE and (vals >= 200).sum() >= 3:
            flags.append("contains_calcium")
        # fat and cavity are judged away from the rim: at the border of a lung nodule soft tissue and air average
        # into the fat range (partial volume), so every nodule would "contain fat"
        interior = m & (inside >= 2.0)
        if interior.sum() >= 10 and float(((img[interior] >= -150) & (img[interior] < -30)).mean()) >= FAT_SHARE:
            flags.append("contains_fat")
        if interior.sum() >= 10 and float((img[interior] < -900).mean()) >= CAVITY_SHARE:
            flags.append("cavitated")
        profile.flags = tuple(flags)
        if profile.surrounding is not None and profile.surrounding < AERATED_SURROUNDING_HU:
            solid = m & (img >= SOLID_HU)
            profile.solid_share = float(solid.sum() / m.sum())
            total = measure_region(m, _shifted(affine, box)).longest_axial_diameter_mm or 1e-6
            if solid.any():
                labels, sizes = label_parts(solid)
                profile.solid_axial_mm = measure_region(labels == 1, _shifted(affine, box)).longest_axial_diameter_mm
            else:
                profile.solid_axial_mm = 0.0
            if profile.solid_axial_mm < GGO_SOLID_MM:
                profile.composition = "ground_glass"
            elif profile.solid_axial_mm >= SOLID_RATIO * total:
                profile.composition = "solid"
            else:
                profile.composition = "part_solid"
    profile.notes = tuple(notes)
    return profile


def _shifted(affine, box: tuple[slice, ...]) -> np.ndarray:
    """The affine of a cropped grid: the origin moves to the first voxel of the box."""
    a = np.asarray(affine, dtype=float).copy()
    a[:3, 3] = a[:3, :3] @ np.array([box[2].start, box[1].start, box[0].start]) + a[:3, 3]
    return a


# report words (site_qualifiers "composition:<kind>", "density:<direction>") -> is the image clearly against?
def compare_with_report(character: list[str] | tuple[str, ...], profile: DensityProfile | None) -> list[str]:
    """The report's words the image clearly contradicts (empty: nothing contradicted). Only clear opposites
    count: a word the image cannot judge (MR signal without the sequence, a composition outside aerated lung)
    is not compared."""
    if profile is None:
        return []
    out = []
    hu = profile.units == "HU"
    for value in character or ():
        quality, _, kind = value.partition(":")
        if quality == "density" and hu and profile.relative_value is not None:
            d = profile.relative_value
            if (kind == "low" and d > ISO_HU) or (kind == "high" and d < -ISO_HU) or \
                    (kind == "equal" and abs(d) > 2 * ISO_HU):
                out.append(f"{value}~image:{profile.relative}({d:+.0f}HU)")
        elif quality == "composition" and hu:
            neg = kind.startswith("not_")
            kind = kind.removeprefix("not_")
            got = _composition_holds(kind, profile)
            if got is None:
                continue
            if got == neg:
                out.append(f"{value}~image:{profile.composition or ','.join(profile.flags) or 'none'}")
    return out


def _composition_holds(kind: str, p: DensityProfile) -> bool | None:
    """Whether the image shows the composition ``kind`` (None: the image cannot say)."""
    if kind in ("solid", "part_solid", "ground_glass"):
        if not p.composition:   # outside aerated lung only "solid" against a clearly fluid content is judged
            return False if kind == "solid" and -20.0 <= p.percentiles["p50"] <= 20.0 else None
        if kind == "solid":
            return p.composition == "solid"
        if kind == "ground_glass":
            return p.composition == "ground_glass"
        return p.composition == "part_solid"
    if kind == "calcified":
        return "contains_calcium" in p.flags
    if kind == "fat":
        return "contains_fat" in p.flags
    if kind == "cavitated":
        return "cavitated" in p.flags
    if kind == "fluid":
        return -20.0 <= p.percentiles["p50"] <= 30.0
    return None


__all__ = ["BANDS_HU", "DensityProfile", "compare_with_report", "density_profile", "distance_mm", "spacing_zyx"]
