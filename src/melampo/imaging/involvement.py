"""Which anatomical structures a lesion involves, and where it sits in the one that holds it (18 October 2026).

Frank: no tool made for one kind of lesion. The lesion is found from the words of the report, wherever it is, and
what the image shows of it -- the detailed anatomical zones it involves, its density, its margins, its shape --
becomes data for its branch of the graph.

The named structure alone (``findings.image_evidence``: the share of the lesion inside the structure the linker
read) says whether the image agrees with the report. This module records *every* structure the lesion touches,
from the masks of the whole-body segmenter (TotalSegmentator: lobes of the lungs, segments of the spine, organs,
vessels, muscles), each with:

* ``lesion_share`` -- the share of the lesion's voxels inside the structure;
* ``surround_share`` -- the share of the lesion's outer shell (0-``CONTACT_MM`` mm outside it) that is in the
  structure: how much of the lesion's surface faces it. A tumour that replaced the tissue of an organ is not in
  the organ's mask (the mask is the tissue), but the organ surrounds it;
* ``distance_mm`` -- 0 when they overlap, else the gap between them (only structures within ``CONTACT_MM``);
* ``relation`` -- ``inside`` (lesion_share >= ``INSIDE_SHARE``), ``crosses`` (part of the lesion is in the
  structure, part is not: the lesion crosses the structure's border), ``contact`` (it touches the structure);
* ``structure_share`` -- the share of the structure the lesion takes (how much of a lobe, of a vertebra).

The structure that holds the lesion (the **host**) is the one with the largest lesion_share when it is at least
``CROSSES_SHARE``, else the one that surrounds most of it (surround_share >= ``HOST_SURROUND_SHARE``). In the host:

* ``gap_to_surface_mm`` -- the shortest distance from the lesion to the host's outer surface (the host filled
  with the lesion, so a tumour that replaced tissue is inside it): 0-``SURFACE_MM`` mm is a lesion on the surface
  (subpleural, subcapsular);
* ``relative_depth`` -- the depth of the lesion's centroid as a share of the host's deepest point, and ``zone``:
  peripheral (outer third), intermediate, central (inner third). For a lung lobe the inner third is towards the
  hilum, roughly: the zone is geometric, not the radiologist's "central" (perihilar), and is recorded as such.

These are attributes for the graph, never reasons for a person on their own: a report need not list every
structure a lesion touches.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .density import distance_mm, spacing_zyx

INSIDE_SHARE = 0.90
CROSSES_SHARE = 0.05
CONTACT_MM = 2.0
HOST_SURROUND_SHARE = 0.50
SURFACE_MM = 2.0
ZONE_THIRDS = (1.0 / 3.0, 2.0 / 3.0)
INSIDE, CROSSES, CONTACT = "inside", "crosses", "contact"


@dataclass
class Involvement:
    structure: str
    relation: str
    lesion_share: float
    surround_share: float
    distance_mm: float
    structure_share: float

    def as_dict(self) -> dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass
class Position:
    host: str
    gap_to_surface_mm: float
    touches_surface: bool
    relative_depth: float
    zone: str                       # peripheral / intermediate / central
    host_chosen_by: str             # "lesion_share" or "surround_share"

    def as_dict(self) -> dict[str, Any]:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


class StructureLabels(Mapping):
    """The structures of one multi-label map, as name -> mask without holding a full mask per structure (a
    whole-body CT has ~100 structures: 100 boolean volumes of 512 x 512 x 300 would take 8 GB)."""

    def __init__(self, labels: np.ndarray, names: dict[int, str]):
        self.labels = np.asarray(labels)
        present = np.unique(self.labels)
        self.code = {names[int(v)]: int(v) for v in present if int(v) and int(v) in names}
        self._counts = np.bincount(self.labels.ravel().astype(np.int64))

    def __getitem__(self, name: str) -> np.ndarray:
        return self.labels == self.code[name]

    def __iter__(self):
        return iter(self.code)

    def __len__(self) -> int:
        return len(self.code)

    def count(self, name: str) -> int:
        return int(self._counts[self.code[name]])

    def in_box(self, box: tuple[slice, ...]) -> dict[str, np.ndarray]:
        sub = self.labels[box]
        inv = {v: k for k, v in self.code.items()}
        return {inv[int(v)]: sub == v for v in np.unique(sub) if int(v) in inv}


def _box(mask: np.ndarray, margin_vox) -> tuple[slice, ...]:
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - margin_vox, 0)
    hi = np.minimum(idx.max(0) + margin_vox + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def involvement(region: np.ndarray, structures: dict[str, np.ndarray], affine,
                contact_mm: float = CONTACT_MM) -> tuple[list[Involvement], Position | None]:
    """The structures the lesion ``region`` involves (largest lesion_share, then surround_share, first) and its
    position in the host. ``structures``: name -> mask on the same grid as ``region``."""
    region = np.asarray(region, dtype=bool)
    if not region.any() or not structures:
        return [], None
    sp = spacing_zyx(affine)
    box = _box(region, np.ceil((contact_mm + max(sp)) / sp).astype(int) + 1)
    r = region[box]
    n = float(r.sum())
    outside, _ = distance_mm(~r, sp)                     # distance of each voxel outside the lesion to it
    shell = ~r & (outside > 0) & (outside <= contact_mm)
    n_shell = float(shell.sum()) or 1.0
    got: list[Involvement] = []
    if isinstance(structures, StructureLabels):
        crops = structures.in_box(box)
        total_of = structures.count
    else:
        crops = {name: np.asarray(full[box], dtype=bool) for name, full in structures.items()}
        def total_of(name):
            return int(np.count_nonzero(structures[name]))
    for name, s in crops.items():
        if not s.any():
            continue
        overlap = float((s & r).sum())
        near = s & ~r
        gap = 0.0 if overlap else (float(outside[near].min()) if near.any() else np.inf)
        if gap > contact_mm:
            continue
        lesion_share = overlap / n
        relation = INSIDE if lesion_share >= INSIDE_SHARE else CROSSES if lesion_share >= CROSSES_SHARE else CONTACT
        total = float(total_of(name)) or 1.0
        got.append(Involvement(name, relation, lesion_share, float((s & shell).sum()) / n_shell, gap,
                               overlap / total))
    got.sort(key=lambda i: (-i.lesion_share, -i.surround_share, i.structure))
    return got, _position(region, structures, got, sp)


def _position(region: np.ndarray, structures: dict[str, np.ndarray], got: list[Involvement],
              sp: np.ndarray) -> Position | None:
    if not got:
        return None
    by_share = max(got, key=lambda i: i.lesion_share)
    if by_share.lesion_share >= CROSSES_SHARE:
        host, chosen = by_share.structure, "lesion_share"
    else:
        by_surround = max(got, key=lambda i: i.surround_share)
        if by_surround.surround_share < HOST_SURROUND_SHARE:
            return None
        host, chosen = by_surround.structure, "surround_share"
    filled = np.asarray(structures[host], dtype=bool) | region
    box = _box(filled, np.array([1, 1, 1]))
    f, r = filled[box], region[box]
    depth, _ = distance_mm(f, sp)                         # depth of each voxel of the host below its surface
    deepest = float(depth.max()) or 1.0
    # the lesion's shortest distance to the host's outer surface: its shallowest voxel, less half a voxel (the
    # distance is measured from voxel centres to the first voxel outside)
    gap = max(0.0, float(depth[r].min()) - float(min(sp)))
    c = np.round(np.argwhere(r).mean(0)).astype(int)
    c_depth = float(depth[tuple(c)]) if r[tuple(c)] else float(np.median(depth[r]))
    rel = c_depth / deepest
    zone = "peripheral" if rel < ZONE_THIRDS[0] else "central" if rel > ZONE_THIRDS[1] else "intermediate"
    return Position(host, gap, gap <= SURFACE_MM, rel, zone, chosen)


def compare_position_with_report(character: list[str] | tuple[str, ...], position: Position | None) -> list[str]:
    """The report's words on the position (``position:surface`` -- "subpleurico", "sottocapsulare") the image
    clearly contradicts: a lesion written on the surface that is at least ``3 x SURFACE_MM`` below it."""
    if position is None:
        return []
    out = []
    for value in character or ():
        if value == "position:surface" and position.gap_to_surface_mm >= 3 * SURFACE_MM:
            out.append(f"{value}~image:gap_{position.gap_to_surface_mm:.0f}mm_in_{position.host}")
    return out


__all__ = ["CONTACT", "CROSSES", "INSIDE", "Involvement", "Position", "StructureLabels", "compare_position_with_report",
           "involvement"]
