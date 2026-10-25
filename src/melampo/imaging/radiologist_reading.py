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

Eleventh probe (25 Oct 2026): on real CT these readings ran along vessels thicker than 4 mm, the pleura and the
chest wall, and the ring around large masses mixed lung, wall and mediastinum. They now read on the patient's
atlas (``segmenters.TotalSegmentatorSegmenter.all_structures``: the main task with the lung's vessels and airways,
the chest-wall muscles, effusions, and the cavities as containers), with fat planes, and with a band reading
(``window_band``) that only refines the segmenter's edge.

These readings are recorded beside the segmenter's region; they decide nothing yet. They exist to learn, against
radiologists' own contours (LIDC-IDRI: four readers per nodule), which edge a radiologist draws -- never tuned
to the reference contours of the probe.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .density import SOLID_HU, distance_mm, spacing_zyx
from .lesion_measures import label_parts, measure_region
from .density import _smooth3
from .threshold_check import GGO_STEP_HU, OPEN_MM, _box, _dilate, _erode, _open_inplane, grow

WINDOWS = {"lung": (-600.0, 1500.0), "soft_tissue": (40.0, 400.0)}   # (level, width) in HU
ZONE_MM = 10.0                  # how far beyond the segmenter's region a reading may reach
AERATED_HU = -500.0             # the lesion lies in aerated lung when its surroundings are below this
KEEP_SHARE = 0.05               # a cavity the lesion already runs into (this share of it) is one it may be in
HOLD_SHARE = 0.20               # a structure holding at least this share of the lesion is not left out
BAND_MM = 3.0                   # the band reading: the edge refined only this close to the segmenter's edge
FAT_HU = (-190.0, -30.0)        # a fat plane (smoothed values), away from aerated lung (partial volume reads as fat)
FAT_SPREAD_HU = 40.0            # ... and homogeneous (3x3x3 standard deviation)


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
    """The readings of the lesion ``seed`` (the segmenter's region) on the patient's atlas: ``window`` (the edge
    at the window's level, up to ``ZONE_MM`` beyond the seed), ``window_band`` (the same edge only within
    ``BAND_MM`` of the seed's), ``solid`` and ``total`` (aerated lung). Each with its level, measures, notes and
    ``mask`` (full grid, not for serialising).

    What a radiologist knows and the eleventh probe's readings did not (25 Oct 2026): every voxel the atlas names
    as normal anatomy is not the lesion (vessels and airways inside the lung, muscles, effusions, other organs);
    the lesion stays in its cavity (thoracic, abdominal, mediastinum: ``structures.regions``) unless it already
    runs into another; a fat plane separates it from the wall and the mediastinum; the tissue around it, for its
    density and noise, is the parenchyma of the structure that holds it, not a ring mixing lung, wall and
    mediastinum (which gave a noise of 240-520 HU and a total reading above the tumour's own density)."""
    names = ("window", "window_band", "edge", "solid", "total")
    out: dict = {k: None for k in names} | {"notes": []}
    seed = np.asarray(seed, dtype=bool)
    if image is None or not seed.any() or (modality or "").upper() != "CT":
        out["notes"].append("not_applicable:needs_ct_hounsfield_units")
        return out
    sp = spacing_zyx(affine)
    box = _box(seed, np.ceil((ZONE_MM + 12.0) / sp).astype(int))
    img = np.asarray(image[box], dtype=np.float32)
    s = seed[box]
    n_seed = float(s.sum())
    zone = grow(s, np.ceil(ZONE_MM / sp).astype(int))
    zone_edge = zone & ~_erode(zone)

    # 1. named anatomy: every structure but the ones that hold the lesion is left out beyond the seed. A vessel
    # crossing the lesion overlaps it too (a 6 mm vessel is ~13% of a 16 mm nodule), so overlapping is not
    # holding: a holding structure has the host's name or at least HOLD_SHARE of the lesion
    excluded = np.zeros_like(s)
    kept, holding = [], np.zeros_like(s)
    for name, m in _crops(structures, box).items():
        share = float((m & s).sum()) / n_seed
        if name == host or share >= HOLD_SHARE:
            kept.append(name)
            holding |= m
            continue
        excluded |= m & ~s
    # 2. the cavity: the lesion stays in the containers that hold it
    regions = getattr(structures, "regions", None) or {}
    cavity = None
    if regions:
        crops = {k: np.asarray(v[box], dtype=bool) for k, v in regions.items()}
        inside_shares = {k: float((m & s).sum()) / n_seed for k, m in crops.items()}
        held = [k for k, v in inside_shares.items() if v >= KEEP_SHARE]
        if held:
            cavity = np.zeros_like(s)
            for k in held:
                cavity |= crops[k]
            cavity = _dilate(cavity) | s          # one voxel of tolerance at the cavity's border
            out["cavities"] = held
    # 3. fat planes: fat away from aerated lung (a tumour's edge against air reads as fat by partial volume)
    # and homogeneous: the 3x3x3 spread of a real fat plane is noise (~10-30 HU); where a lesion meets ground
    # glass the mean also passes through the fat range, but across a steep edge
    sm = _smooth3(img)
    spread = np.sqrt(np.clip(_smooth3(img * img) - sm * sm, 0, None))
    near_air = grow(img < AERATED_HU, np.maximum(1, np.round(2.0 / sp)).astype(int))
    fat = (sm >= FAT_HU[0]) & (sm <= FAT_HU[1]) & (spread <= FAT_SPREAD_HU) & ~near_air & ~s
    excluded |= fat
    allowed = ~excluded if cavity is None else (~excluded & cavity)
    out["kept_structures"], out["excluded_voxels"] = kept, int(excluded.sum())
    out["fat_plane_voxels"] = int(fat.sum())

    # 4. the tissue around: the parenchyma of the structures that hold the lesion, else what is allowed
    inside, _ = distance_mm(s, sp)
    deep = s & (inside >= 0.5 * inside.max())
    core = float(np.median(img[deep] if deep.sum() >= 5 else img[s]))
    outside, _ = distance_mm(~s, sp)
    ring = ~s & (outside > 2.0) & (outside <= 10.0)
    shell = ring & holding & allowed
    if shell.sum() >= 20:
        out["surrounding_from"] = "holding_structure"
    else:
        shell = ring & allowed
        out["surrounding_from"] = "allowed_ring"
    if shell.sum() < 20:
        out["notes"].append("too_little_surrounding_tissue")
        return out
    around = img[shell]
    surrounding = float(np.median(around))
    noise = float(1.4826 * np.median(np.abs(around - surrounding))) or float(around.std()) or 1.0
    out.update(core=core, surrounding=surrounding, noise=noise)
    above = core > surrounding
    band = grow(s, np.ceil(BAND_MM / sp).astype(int))

    def reading(level: float, within: np.ndarray, edge: np.ndarray) -> dict:
        notes: list[str] = []
        region = _region_at(img, s, sp, level, above, within & allowed, edge, notes)
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
    out["window"] = {"window": win, **reading(level, zone, zone_edge)}
    out["window_band"] = {"window": win, **reading(level, band, band & ~_erode(band))}
    # ``edge``: where a radiologist sees the edge depends on what the lesion is made of. A solid lesion in the
    # lung on the lung window: the middle grey (-600 HU) between soft tissue and air. A ground-glass lesion is
    # itself darker than that grey, and its edge is where it meets the lung: half-way between the two (first
    # CPU replay, 25 Oct: the -600 HU edge cut the part-solid lesions 7-20 mm short, and fitted the large solid
    # masses within 1-3 mm where VoxTell was 8-15 mm short). Only within the band.
    # solid or not: the median of the whole region (its deepest part is solid in a part-solid lesion too)
    lesion_median = float(np.median(img[s]))
    out["lesion_median"] = lesion_median
    if win == "lung" and lesion_median >= SOLID_HU and level == WINDOWS["lung"][0]:
        edge_level, rule = level, "solid_lesion:window_level"
    else:
        edge_level, rule = (core + surrounding) / 2.0, "half_way_between_lesion_and_surroundings"
    out["edge"] = {"window": win, "rule": rule, **reading(edge_level, band, band & ~_erode(band))}
    if surrounding < AERATED_HU and above:
        out["solid"] = reading(SOLID_HU, zone, zone_edge)
        total_level = surrounding + max(GGO_STEP_HU, 3.0 * noise)
        if total_level >= min(core, SOLID_HU):
            out["notes"].append(f"total_level_{total_level:.0f}_not_below_the_solid_part:skipped")
        else:
            out["total"] = reading(total_level, zone, zone_edge)
    else:
        out["notes"].append("solid_and_total_only_in_aerated_lung")
    return out


def readings_summary(r: dict) -> dict:
    """The readings without their masks (for a row of the probe or the graph)."""
    keys = ("window", "window_band", "edge", "solid", "total")
    out = {k: v for k, v in r.items() if k not in keys}
    for k in keys:
        out[k] = None if r.get(k) is None else {kk: vv for kk, vv in r[k].items() if kk != "mask"}
    return out


__all__ = ["BAND_MM", "WINDOWS", "radiologist_readings", "readings_summary", "window_for"]
