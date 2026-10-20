"""The lesion read the way a radiologist reads it: the border he sees at his window, what he leaves out, and the
parts the lesion is made of (20 October 2026).

Frank: no tool made for one kind of lesion; find a solution as if we were the radiologists. How a radiologist
draws the edge of a lesion on CT:

1. **The window.** He reads aerated lung on the lung window and soft tissue on the mediastinal (soft-tissue)
   window; the edge he sees is where the density crosses the window's level, the middle grey (Lung-RADS measures
   lung nodules on lung windows). The window comes from the structure that holds the lesion (``involvement``),
   not from a kind of lesion: a lesion in a lung lobe is read on the lung window, anywhere else on the
   soft-tissue window. When the level does not lie between the lesion and its surroundings (a lesion barely
   different from the organ), the edge is half-way between the two, as the eye would put it.
2. **What is not the lesion**, even at the same density: every other structure the whole-body segmentation
   names (vessels, heart, chest wall, bones, other organs) unless the lesion itself already runs into it, and the
   thin vessels inside the lung that the segmentation does not name (cut by an in-plane opening).
3. **The parts.** In aerated lung a lesion may be solid, ground glass, or both (Lung-RADS: a part-solid nodule
   has two measures, the total and the solid component). ``solid``: the connected part denser than
   ``density.SOLID_HU``. ``total``: the solid part with the ground glass continuous with it (denser than the lung
   around by max(100 HU, 3 x its noise)), the same rule as ``threshold_check`` uses.
4. **The measures** of each reading as he takes them: the longest axial diameter, the short axis perpendicular
   to it on the same slice, their mean (``lesion_measures``).

These readings are recorded beside the segmenter's region; they decide nothing yet. They exist to learn, against
radiologists' own contours (LIDC-IDRI: four readers per nodule), which edge a radiologist draws -- never tuned
to the reference contours of the probe.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .density import SOLID_HU, distance_mm, spacing_zyx
from .lesion_measures import label_parts, measure_region
from .threshold_check import GGO_STEP_HU, OPEN_MM, _box, _dilate, _erode, _open_inplane, grow

WINDOWS = {"lung": (-600.0, 1500.0), "soft_tissue": (40.0, 400.0)}   # (level, width) in HU
ZONE_MM = 10.0                  # how far beyond the segmenter's region a reading may reach
AERATED_HU = -500.0             # the lesion lies in aerated lung when its surroundings are below this
KEEP_SHARE = 0.05               # a structure the lesion already runs into (this share of it) is not excluded


def window_for(host: str | None) -> str:
    """The window a radiologist reads a lesion on, from the structure that holds it."""
    return "lung" if host and host.lower().startswith("lung") else "soft_tissue"


def _crops(structures, box) -> dict[str, np.ndarray]:
    if structures is None:
        return {}
    if hasattr(structures, "in_box"):
        return structures.in_box(box)
    return {k: np.asarray(v[box], dtype=bool) for k, v in structures.items()}


def _region_at(img: np.ndarray, s: np.ndarray, sp, level: float, above: bool, zone: np.ndarray,
               zone_edge: np.ndarray, notes: list[str]) -> np.ndarray | None:
    side = (img >= level) if above else (img <= level)
    dense = side & zone
    r = tuple(int(x) for x in np.maximum(1, np.round(OPEN_MM / sp[1:])))
    opened = _open_inplane(dense, r)
    if not (opened & s).any():
        opened = _dilate(_erode(dense)) & dense
    if (opened & s).any():
        dense = opened
    labels, sizes = label_parts(dense)
    if not sizes:
        return None
    overlap = np.bincount(labels[s].ravel(), minlength=len(sizes) + 1)[1:]
    if not overlap.any():
        return None
    region = labels == int(np.argmax(overlap)) + 1
    if (region & zone_edge).any():   # the outer edge of the zone, not the border of an excluded structure
        notes.append("reaches_the_reading_zone_edge")
    return region


def _measures(region: np.ndarray | None, affine) -> dict:
    if region is None or not region.any():
        return {"axial_long_mm": None, "axial_short_mm": None, "axial_mean_mm": None, "volume_mm3": None}
    m = measure_region(region, affine)
    return {"axial_long_mm": m.longest_axial_diameter_mm, "axial_short_mm": m.axial_short_mm,
            "axial_mean_mm": m.axial_mean_mm, "volume_mm3": m.volume_mm3}


def radiologist_readings(image: np.ndarray, seed: np.ndarray, affine, modality: str = "CT",
                         host: str | None = None, structures: Mapping | None = None) -> dict:
    """The window, solid and total readings of the lesion ``seed`` (the segmenter's region). Returns a dict of
    readings, each with its level, measures, notes and ``mask`` (full grid, not for serialising)."""
    out: dict = {"window": None, "solid": None, "total": None, "notes": []}
    seed = np.asarray(seed, dtype=bool)
    if image is None or not seed.any() or (modality or "").upper() != "CT":
        out["notes"].append("not_applicable:needs_ct_hounsfield_units")
        return out
    sp = spacing_zyx(affine)
    box = _box(seed, np.ceil((ZONE_MM + 12.0) / sp).astype(int))
    img = np.asarray(image[box], dtype=np.float32)
    s = seed[box]
    zone = grow(s, np.ceil(ZONE_MM / sp).astype(int))
    zone_edge = zone & ~_erode(zone)
    # what is not the lesion: every named structure but the ones the lesion lies in or already runs into
    excluded = np.zeros_like(s)
    kept = []
    for name, m in _crops(structures, box).items():
        share = float((m & s).sum()) / float(s.sum())
        if name == host or share >= KEEP_SHARE:
            kept.append(name)
            continue
        excluded |= m & ~s
    zone &= ~excluded
    out["kept_structures"], out["excluded_voxels"] = kept, int(excluded.sum())

    inside, _ = distance_mm(s, sp)
    deep = s & (inside >= 0.5 * inside.max())
    core = float(np.median(img[deep] if deep.sum() >= 5 else img[s]))
    outside, _ = distance_mm(~s, sp)
    shell = ~s & ~excluded & (outside > 2.0) & (outside <= 10.0)
    if shell.sum() < 20:
        out["notes"].append("too_little_surrounding_tissue")
        return out
    around = img[shell]
    surrounding = float(np.median(around))
    noise = float(1.4826 * np.median(np.abs(around - surrounding))) or float(around.std()) or 1.0
    out.update(core=core, surrounding=surrounding, noise=noise)
    above = core > surrounding

    def reading(name: str, level: float) -> dict:
        notes: list[str] = []
        region = _region_at(img, s, sp, level, above, zone, zone_edge, notes)
        full = None
        if region is not None:
            full = np.zeros(seed.shape, dtype=bool)
            full[box] = region
        return {"level": level, "notes": notes, "mask": full, **_measures(full, affine)}

    win = window_for(host)
    level = WINDOWS[win][0]
    if not (min(core, surrounding) < level < max(core, surrounding)):
        level = (core + surrounding) / 2.0
        out["notes"].append(f"window_level_not_between_lesion_and_surroundings:{win}:half_way")
    out["window"] = {"window": win, **reading("window", level)}
    if surrounding < AERATED_HU and above:
        out["solid"] = reading("solid", SOLID_HU)
        out["total"] = reading("total", surrounding + max(GGO_STEP_HU, 3.0 * noise))
    else:
        out["notes"].append("solid_and_total_only_in_aerated_lung")
    return out


def readings_summary(r: dict) -> dict:
    """The readings without their masks (for a row of the probe or the graph)."""
    out = {k: v for k, v in r.items() if k not in ("window", "solid", "total")}
    for k in ("window", "solid", "total"):
        out[k] = None if r.get(k) is None else {kk: vv for kk, vv in r[k].items() if kk != "mask"}
    return out


__all__ = ["WINDOWS", "radiologist_readings", "readings_summary", "window_for"]
