"""What a region of a volume measures, in millimetres and in the units of the image (16 October 2026).

Given a mask (the region a segmenter isolated) and the geometry of the series (``dicom_volume.VolumeAssessment``:
the affine from voxel indices to patient millimetres), everything here is deterministic and checkable:

* **position**: centroid in patient coordinates (LPS mm), bounding box;
* **size**: volume (voxel count x voxel volume from the affine, so a sheared or anisotropic grid is right),
  longest diameter in 3D, longest axial diameter (the RECIST measure), the three principal axes;
* **shape** (the "vectorisation" of the region, decision D18 of 3 October: descriptors until the shared SDF
  decoder exists): surface area, sphericity, elongation, flatness, surface-to-volume ratio, and the boundary as a
  point set in millimetres (a mesh when scikit-image is installed);
* **intensity**: mean, standard deviation, percentiles of the image values inside the region (Hounsfield units
  for CT; for MR the raw value means nothing across scanners, so only the ratio to a reference region is kept).

Precision, stated: a boundary is limited by the voxel and the partial-volume effect. Surface area from counted
voxel faces over-estimates a smooth surface by 3/2 on average over orientations (the mean of |n_x|+|n_y|+|n_z|
over the unit sphere); that factor is corrected and the method is written beside the number. A region under
``MIN_VOXELS`` voxels is measured but its shape is not reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

MIN_VOXELS = 27            # below a 3x3x3 block, shape descriptors are noise
MAX_BOUNDARY_POINTS = 4000  # longest-diameter search on a sample of the boundary beyond this
FACE_COUNT_CORRECTION = 2.0 / 3.0


def _affine(affine) -> np.ndarray:
    a = np.asarray(affine, dtype=float)
    if a.shape != (4, 4):
        raise ValueError("the affine must be 4x4 (column, row, slice, 1) -> patient mm")
    return a


def to_mm(indices_zyx: np.ndarray, affine) -> np.ndarray:
    """Voxel indices (slice, row, column) -> patient mm (LPS), through the series' affine (column, row, slice)."""
    a = _affine(affine)
    idx = np.asarray(indices_zyx, dtype=float).reshape(-1, 3)
    crs1 = np.column_stack([idx[:, 2], idx[:, 1], idx[:, 0], np.ones(len(idx))])
    return (crs1 @ a.T)[:, :3]


def voxel_volume_mm3(affine) -> float:
    a = _affine(affine)
    return float(abs(np.linalg.det(a[:3, :3])))


def boundary(mask: np.ndarray) -> np.ndarray:
    """The voxels of the region with at least one 6-neighbour outside it, as (slice, row, column) indices."""
    m = np.pad(mask.astype(bool), 1)
    inner = m[1:-1, 1:-1, 1:-1]
    interior = inner.copy()
    for axis in range(3):
        for shift in (-1, 1):
            interior &= np.roll(m, shift, axis=axis)[1:-1, 1:-1, 1:-1]
    return np.argwhere(inner & ~interior)


def _face_area(mask: np.ndarray, affine) -> float:
    """Area of the exposed voxel faces in mm^2 (each face's area from the affine's columns)."""
    a = _affine(affine)[:3, :3]
    col, row, sl = a[:, 0], a[:, 1], a[:, 2]
    # a face perpendicular to the slice axis spans the column and row vectors, and so on
    area_of = {0: np.linalg.norm(np.cross(col, row)), 1: np.linalg.norm(np.cross(col, sl)),
               2: np.linalg.norm(np.cross(row, sl))}
    m = np.pad(mask.astype(bool), 1)
    total = 0.0
    for axis in range(3):
        exposed = np.count_nonzero(m & ~np.roll(m, 1, axis=axis)) + np.count_nonzero(m & ~np.roll(m, -1, axis=axis))
        total += exposed * area_of[axis]
    return float(total)


def _mesh_area(mask: np.ndarray, affine) -> tuple[float, np.ndarray, np.ndarray] | None:
    try:
        from skimage.measure import marching_cubes, mesh_surface_area
    except ImportError:
        return None
    verts, faces, _, _ = marching_cubes(np.pad(mask.astype(np.uint8), 1), level=0.5)
    verts_mm = to_mm(verts - 1.0, affine)
    return float(mesh_surface_area(verts_mm, faces)), verts_mm, faces


def _longest(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    if len(points) > MAX_BOUNDARY_POINTS:
        rng = np.random.default_rng(0)
        points = points[rng.choice(len(points), MAX_BOUNDARY_POINTS, replace=False)]
    best = 0.0
    for chunk in np.array_split(points, max(1, len(points) // 500)):
        d = np.sqrt(((chunk[:, None, :] - points[None, :, :]) ** 2).sum(-1)).max()
        best = max(best, float(d))
    return best


@dataclass
class RegionMeasures:
    voxels: int
    volume_mm3: float
    centroid_mm: tuple[float, float, float]
    bbox_mm: tuple[tuple[float, float, float], tuple[float, float, float]]
    longest_diameter_mm: float
    longest_axial_diameter_mm: float
    principal_axes_mm: tuple[float, float, float]      # full axis lengths of the equivalent solid ellipsoid
    surface_mm2: float | None = None
    surface_method: str = ""
    sphericity: float | None = None
    elongation: float | None = None                    # second / first principal axis
    flatness: float | None = None                      # third / second principal axis
    surface_to_volume: float | None = None
    intensity: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}

    def shape_vector(self) -> list[float]:
        """The shape descriptors as one vector (comparable across lesions: no encoder behind it)."""
        return [float(x) if x is not None else float("nan") for x in (
            np.log10(max(self.volume_mm3, 1e-6)), self.sphericity, self.elongation, self.flatness,
            self.surface_to_volume)]


def measure_region(mask: np.ndarray, affine, image: np.ndarray | None = None,
                   reference_mask: np.ndarray | None = None) -> RegionMeasures:
    """The measures of the region ``mask`` (slice, row, column) of a series with this ``affine``."""
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n == 0:
        raise ValueError("empty region")
    notes: list[str] = []
    idx = np.argwhere(mask)
    pts = to_mm(idx, affine)
    centroid = tuple(float(x) for x in pts.mean(axis=0))
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    volume = n * voxel_volume_mm3(affine)

    edge = boundary(mask)
    edge_mm = to_mm(edge, affine)
    longest = _longest(edge_mm)
    axial = 0.0
    for s in np.unique(edge[:, 0]):
        axial = max(axial, _longest(edge_mm[edge[:, 0] == s]))

    cov = np.cov(pts.T) if n > 1 else np.zeros((3, 3))
    eig = np.sort(np.clip(np.linalg.eigvalsh(cov), 0, None))[::-1]
    # a solid ellipsoid with semi-axis a has variance a^2/5 along it: full length 2*sqrt(5*variance)
    axes = tuple(float(2.0 * np.sqrt(5.0 * e)) for e in eig)

    measures = RegionMeasures(
        voxels=n, volume_mm3=float(volume), centroid_mm=centroid,
        bbox_mm=(tuple(float(x) for x in lo), tuple(float(x) for x in hi)),
        longest_diameter_mm=longest, longest_axial_diameter_mm=axial, principal_axes_mm=axes,
    )
    if n >= MIN_VOXELS:
        mesh = _mesh_area(mask, affine)
        if mesh is not None:
            area, method = mesh[0], "marching_cubes"
        else:
            area, method = _face_area(mask, affine) * FACE_COUNT_CORRECTION, "voxel_faces_x2/3"
        measures.surface_mm2, measures.surface_method = area, method
        measures.sphericity = float(np.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / area) if area > 0 else None
        measures.surface_to_volume = float(area / volume)
        if axes[0] > 0:
            measures.elongation = axes[1] / axes[0]
        if axes[1] > 0:
            measures.flatness = axes[2] / axes[1]
    else:
        notes.append(f"shape_not_reported_under_{MIN_VOXELS}_voxels")

    if image is not None:
        values = np.asarray(image)[mask].astype(float)
        measures.intensity = {"mean": float(values.mean()), "sd": float(values.std()),
                              "p10": float(np.percentile(values, 10)), "p50": float(np.percentile(values, 50)),
                              "p90": float(np.percentile(values, 90))}
        if reference_mask is not None and np.asarray(reference_mask).any():
            ref = float(np.asarray(image)[np.asarray(reference_mask, dtype=bool)].astype(float).mean())
            if ref:
                measures.intensity["ratio_to_reference"] = float(values.mean() / ref)
    measures.notes = tuple(notes)
    return measures


def boundary_points_mm(mask: np.ndarray, affine) -> np.ndarray:
    """The region's boundary as points in patient mm: the shape kept beside its descriptors (a mesh when
    scikit-image is installed: ``mesh_mm``)."""
    return to_mm(boundary(np.asarray(mask, dtype=bool)), affine)


def mesh_mm(mask: np.ndarray, affine) -> tuple[np.ndarray, np.ndarray] | None:
    got = _mesh_area(np.asarray(mask, dtype=bool), affine)
    return None if got is None else (got[1], got[2])


def label_parts(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """The 6-connected parts of a region: a label array (0 outside, 1 the largest part, 2 the next, ...) and the
    sizes in voxels, largest first. scipy when installed, else plain numpy (scipy is not a dependency)."""
    mask = np.asarray(mask, dtype=bool)
    try:
        from scipy.ndimage import label
        labels, n = label(mask)
    except ImportError:
        labels, n = _label_numpy(mask)
    if n == 0:
        return np.zeros(mask.shape, dtype=np.int32), []
    sizes = np.bincount(labels.ravel())[1:]
    order = np.argsort(-sizes, kind="stable")
    relabel = np.zeros(n + 1, dtype=np.int32)
    relabel[order + 1] = np.arange(1, n + 1)
    return relabel[labels], [int(sizes[i]) for i in order]


def _label_numpy(mask: np.ndarray) -> tuple[np.ndarray, int]:
    from collections import deque

    labels = np.zeros(mask.shape, dtype=np.int32)
    n = 0
    for start in map(tuple, np.argwhere(mask)):
        if labels[start]:
            continue
        n += 1
        labels[start] = n
        queue = deque([start])
        while queue:
            z, y, x = queue.popleft()
            for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                q = (z + dz, y + dy, x + dx)
                if all(0 <= q[i] < mask.shape[i] for i in range(3)) and mask[q] and not labels[q]:
                    labels[q] = n
                    queue.append(q)
    return labels, n


@dataclass
class Fragment:
    """A part of the region detached from its largest part."""
    voxels: int
    volume_mm3: float
    share_of_main: float          # its volume / the largest part's volume
    distance_mm: float            # closest distance between its voxels and the largest part's boundary


def split_main_part(mask: np.ndarray, affine) -> tuple[np.ndarray, list[Fragment]]:
    """The largest part of the region and the detached rest (16 October 2026, case lung_006 of the Kaggle probe:
    a region with a good overlap measured 137 mm against 27 mm of reference, because a detached speck far from
    the lesion entered the longest diameter). Sizes are measured on the largest part; the rest is reported."""
    labels, sizes = label_parts(mask)
    if not sizes:
        return np.zeros(np.shape(mask), dtype=bool), []
    main = labels == 1
    main_edge = to_mm(boundary(main), affine)
    if len(main_edge) > MAX_BOUNDARY_POINTS:
        main_edge = main_edge[np.random.default_rng(0).choice(len(main_edge), MAX_BOUNDARY_POINTS, replace=False)]
    vox = voxel_volume_mm3(affine)
    fragments = []
    for k, size in enumerate(sizes[1:], start=2):
        pts = to_mm(np.argwhere(labels == k), affine)
        if len(pts) > 500:
            pts = pts[np.random.default_rng(0).choice(len(pts), 500, replace=False)]
        d = min(float(np.sqrt(((c[:, None, :] - main_edge[None, :, :]) ** 2).sum(-1)).min())
                for c in np.array_split(pts, max(1, len(pts) // 100)))
        fragments.append(Fragment(size, size * vox, size / sizes[0], d))
    return main, fragments


def _hull2d(points: np.ndarray) -> np.ndarray:
    """Convex hull of 2D integer points (monotone chain), counter-clockwise."""
    pts = sorted(set(map(tuple, points.tolist())))
    if len(pts) <= 2:
        return np.array(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def inside_hull_share(region: np.ndarray, organ_parts) -> float:
    """Share of the region's voxels inside the convex hull, slice by slice, of each organ part (a lobe, a
    kidney...). A tumour that replaces lung tissue is missing from the lung mask (TotalSegmentator segments the
    parenchyma), so the plain overlap can be low for a tumour that is in the lung (fourth probe, lung_026: 40%);
    inside the outline of the lobe on its slices it is not. Only the slices of the region are computed."""
    region = np.asarray(region, dtype=bool)
    parts = [np.asarray(p, dtype=bool) for p in (organ_parts if isinstance(organ_parts, (list, tuple)) else [organ_parts])]
    inside = np.zeros(region.shape, dtype=bool)
    for z in np.unique(np.argwhere(region)[:, 0]):
        rz = np.argwhere(region[z])
        for part in parts:
            pts = np.argwhere(part[z])
            if len(pts) < 3:
                continue
            hull = _hull2d(pts)
            if len(hull) < 3:
                continue
            ok = np.ones(len(rz), dtype=bool)
            for i in range(len(hull)):
                a, b = hull[i], hull[(i + 1) % len(hull)]
                ok &= (b[0] - a[0]) * (rz[:, 1] - a[1]) - (b[1] - a[1]) * (rz[:, 0] - a[0]) >= -0.5
            inside[z][tuple(rz[ok].T)] = True
    return float(inside.sum() / max(region.sum(), 1))


def lps_to_ras_affine(affine) -> np.ndarray:
    """The same grid in RAS (what VoxTell and NIfTI expect): x and y change sign."""
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ _affine(affine)


def overlap_inside(region: np.ndarray, organ: np.ndarray, margin_voxels: int = 2) -> float:
    """Share of the region's voxels inside the organ grown by a margin (a lesion on the surface of an organ)."""
    region = np.asarray(region, dtype=bool)
    grown = np.asarray(organ, dtype=bool).copy()
    for _ in range(margin_voxels):
        nxt = grown.copy()
        for axis in range(3):
            for shift in (-1, 1):
                nxt |= np.roll(grown, shift, axis=axis)
        grown = nxt
    return float((region & grown).sum() / max(region.sum(), 1))


__all__ = ["MIN_VOXELS", "Fragment", "inside_hull_share", "RegionMeasures", "label_parts", "split_main_part", "boundary", "boundary_points_mm", "lps_to_ras_affine", "measure_region",
           "mesh_mm", "overlap_inside", "to_mm", "voxel_volume_mm3"]
