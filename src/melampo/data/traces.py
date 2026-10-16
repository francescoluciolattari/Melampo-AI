"""Traces in reports, read as numbers: the curve of a protein electrophoresis and the other curves a report draws
(16 October 2026).

Frank: in a digital PDF the curve is drawn as vectors -> numbers and peaks; in a scan the picture of the trace is
found, *vectorised*, and the same method is applied. That is the whole design:

1. **Curves out of the document**
   * ``vector_curves(pdf)``: poppler's ``pdftocairo -svg`` (poppler is already required for the text layer)
     turns each page into SVG; every drawn path is flattened (lines, Bezier curves sampled) into a polyline in
     page coordinates. A trace is a long, left-to-right polyline; axes, frames and table rules are short or
     straight and are left out.
   * ``raster_curve(image)``: the same polyline from a picture: grid lines and axes (rows or columns that are dark
     across most of the plot) are removed, then for every column the highest dark pixel is the curve; short gaps
     are filled, long ones stay gaps. This is the vectorisation of the scan.
2. **One analysis for both** (``analyse_electrophoresis``): baseline, smoothing, valleys that split the zones, the
   share of the area in each zone, the peaks with their width. The albumin peak (the tallest) fixes the direction
   of the axis. A *narrow* peak inside the gamma or beta zone is described as such ("narrow peak in gamma zone,
   width x% of the zone") and sent to a person: it is how a monoclonal component looks on the trace, but the
   reading here is a measurement, never a diagnosis.
3. **The check against the table**: the zone shares read from the curve are compared with the percentages printed
   in the report's table (``lab_results``); a disagreement beyond ``TABLE_TOLERANCE`` points at a misread curve,
   not at the patient.

Other traces (ECG, spirometry, cardiotocography...) are extracted as curves with the same code and kept as series
with ``interpreted = false``: each needs its own analysis, to be added one at a time.
"""

from __future__ import annotations

import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MIN_TRACE_POINTS = 40
MIN_TRACE_WIDTH_SHARE = 0.25     # of the page width
LINE_DARK_SHARE = 0.6            # a row/column dark over this share of the plot is a grid line or an axis
MAX_GAP_COLUMNS = 6
TABLE_TOLERANCE = 3.0            # percentage points between curve and table
NARROW_PEAK_SHARE = 0.35         # a peak narrower than this share of its zone is "narrow"
SPE_ZONES = ("albumin", "alpha1", "alpha2", "beta", "gamma")


# --- 1. curves out of the document ---------------------------------------------------------------------------
_CMD = re.compile(r"([MmLlHhVvCcSsQqTtZz])|(-?\d*\.?\d+(?:e-?\d+)?)")


def _matrix(attr: str | None) -> np.ndarray:
    m = np.eye(3)
    if not attr:
        return m
    for name, args in re.findall(r"(\w+)\(([^)]*)\)", attr):
        v = [float(x) for x in re.split(r"[ ,]+", args.strip()) if x]
        t = np.eye(3)
        if name == "matrix" and len(v) == 6:
            t = np.array([[v[0], v[2], v[4]], [v[1], v[3], v[5]], [0, 0, 1]])
        elif name == "translate":
            t = np.array([[1, 0, v[0]], [0, 1, v[1] if len(v) > 1 else 0], [0, 0, 1]])
        elif name == "scale":
            t = np.array([[v[0], 0, 0], [0, v[1] if len(v) > 1 else v[0], 0], [0, 0, 1]])
        m = m @ t
    return m


def _path_points(d: str) -> list[np.ndarray]:
    """The sub-paths of an SVG path as point arrays (Bezier curves sampled at 8 points)."""
    tokens = _CMD.findall(d)
    out, cur, pts, cmd, nums = [], np.zeros(2), [], None, []

    def flush():
        nonlocal pts
        if len(pts) > 1:
            out.append(np.array(pts))
        pts = []

    def take(cmd, nums):
        nonlocal cur
        rel = cmd.islower()
        c = cmd.upper()
        while nums:
            if c == "M" and len(nums) >= 2:
                flush()
                p = np.array(nums[:2]) + (cur if rel else 0)
                del nums[:2]
                cur = p
                pts.append(p)
                c = "L"  # implicit lineto after the first pair
            elif c == "L" and len(nums) >= 2:
                p = np.array(nums[:2]) + (cur if rel else 0)
                del nums[:2]
                cur = p
                pts.append(p)
            elif c == "H" and nums:
                cur = np.array([nums.pop(0) + (cur[0] if rel else 0), cur[1]])
                pts.append(cur)
            elif c == "V" and nums:
                cur = np.array([cur[0], nums.pop(0) + (cur[1] if rel else 0)])
                pts.append(cur)
            elif c == "C" and len(nums) >= 6:
                p1, p2, p3 = (np.array(nums[i:i + 2]) + (cur if rel else 0) for i in (0, 2, 4))
                del nums[:6]
                for t in np.linspace(0.125, 1.0, 8):
                    pts.append((1 - t) ** 3 * cur + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)
                cur = p3
            else:
                break

    for c, n in tokens:
        if c:
            if cmd:
                take(cmd, nums)
            cmd, nums = c, []
            if c in "Zz":
                flush()
                cmd = None
        else:
            nums.append(float(n))
    if cmd:
        take(cmd, nums)
    flush()
    return out


def svg_curves(svg: str) -> list[np.ndarray]:
    """Every polyline drawn in an SVG page, in page units with y upwards."""
    root = ET.fromstring(svg)
    height = float(re.sub(r"[a-z]+", "", root.get("height", "0") or "0") or 0)
    curves = []

    def walk(node, m):
        m = m @ _matrix(node.get("transform"))
        if node.tag.endswith("path") and node.get("d"):
            for pts in _path_points(node.get("d")):
                xy = (np.column_stack([pts, np.ones(len(pts))]) @ m.T)[:, :2]
                xy[:, 1] = height - xy[:, 1]
                curves.append(xy)
        for child in node:
            walk(child, m)

    walk(root, np.eye(3))
    return curves


def is_trace(xy: np.ndarray, page_width: float) -> bool:
    """A long polyline that runs left to right (an x that mostly increases) and is not a straight line."""
    if len(xy) < MIN_TRACE_POINTS:
        return False
    width = xy[:, 0].max() - xy[:, 0].min()
    if page_width and width < MIN_TRACE_WIDTH_SHARE * page_width:
        return False
    dx = np.diff(xy[:, 0])
    if (dx >= -1e-6).mean() < 0.9 and (dx <= 1e-6).mean() < 0.9:
        return False
    return float(np.ptp(xy[:, 1])) > 0.02 * width


def vector_curves(pdf: bytes, page: int = 1) -> list[np.ndarray]:
    """The traces drawn as vectors on a page of a digital PDF (empty when there are none, or no poppler)."""
    try:
        done = subprocess.run(["pdftocairo", "-svg", "-f", str(page), "-l", str(page), "-", "-"],  # noqa: S607
                              input=pdf, capture_output=True, timeout=60, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if done.returncode != 0 or not done.stdout:
        return []
    svg = done.stdout.decode("utf-8", "replace")
    root_width = re.search(r'width="([\d.]+)', svg)
    width = float(root_width.group(1)) if root_width else 0.0
    return [c for c in svg_curves(svg) if is_trace(c, width)]


def raster_curve(image: np.ndarray, dark: float | None = None) -> np.ndarray:
    """The curve of a picture of a trace, as (x, y) points with y upwards: the vectorisation of a scan."""
    g = np.asarray(image, dtype=float)
    if g.ndim == 3:
        g = g[..., :3].mean(axis=-1)
    threshold = dark if dark is not None else (g.min() + g.max()) / 2
    ink = g < threshold
    ink[:, ink.mean(axis=0) > LINE_DARK_SHARE] = False   # vertical grid lines and axes
    ink[ink.mean(axis=1) > LINE_DARK_SHARE, :] = False   # horizontal grid lines and the baseline axis
    # the trace is one connected stroke that runs across the plot: keep the connected part that spans the most
    # columns (dotted grids, labels and specks are short)
    ink = _widest_component(ink)
    rows, cols = ink.shape
    ys = np.full(cols, np.nan)
    for c in range(cols):
        hit = np.flatnonzero(ink[:, c])
        if hit.size:
            ys[c] = rows - 1 - hit[0]   # the highest dark pixel, y upwards
    # fill short gaps only
    good = np.flatnonzero(~np.isnan(ys))
    if good.size < 2:
        return np.empty((0, 2))
    for a, b in zip(good[:-1], good[1:], strict=True):
        if 1 < b - a <= MAX_GAP_COLUMNS + 1:
            ys[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [ys[a], ys[b]])
    keep = ~np.isnan(ys)
    return np.column_stack([np.arange(cols)[keep].astype(float), ys[keep]])


def _widest_component(ink: np.ndarray) -> np.ndarray:
    """The stroke of the trace: the 8-connected dark component that spans the most columns, plus the pieces of the
    same stroke broken by a steep slope (components over at least 2% of the width that do not share a column with
    what is already kept). Dotted grids, labels and specks overlap the trace's columns or are too short."""
    try:
        from scipy.ndimage import label

        labels, n = label(ink, structure=np.ones((3, 3)))
    except ImportError:
        labels, n = _label8(ink)
    if n == 0:
        return ink
    spans = []
    for k in range(1, n + 1):
        cols = np.flatnonzero((labels == k).any(axis=0))
        spans.append((cols[-1] - cols[0] if cols.size else -1, k, set(cols.tolist())))
    spans.sort(key=lambda t: -t[0])
    keep, used = [spans[0][1]], set(spans[0][2])
    for span, k, cols in spans[1:]:
        if span >= 0.02 * ink.shape[1] and not (cols & used):
            keep.append(k)
            used |= cols
    return np.isin(labels, keep)


def _label8(ink: np.ndarray) -> tuple[np.ndarray, int]:
    labels = np.zeros(ink.shape, dtype=np.int32)
    n = 0
    rows, cols = ink.shape
    for r0, c0 in zip(*np.nonzero(ink), strict=True):
        if labels[r0, c0]:
            continue
        n += 1
        stack = [(r0, c0)]
        labels[r0, c0] = n
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols and ink[rr, cc] and not labels[rr, cc]:
                        labels[rr, cc] = n
                        stack.append((rr, cc))
    return labels, n


# --- 2. the analysis -------------------------------------------------------------------------------------------
def resample(xy: np.ndarray, n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(xy[:, 0], kind="stable")
    x, y = xy[order, 0], xy[order, 1]
    x, idx = np.unique(x, return_index=True)
    y = y[idx]
    grid = np.linspace(x.min(), x.max(), n)
    return grid, np.interp(grid, x, y)


def _smooth(y: np.ndarray, k: int = 7) -> np.ndarray:
    k = max(1, k | 1)
    pad = np.pad(y, k // 2, mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def _extrema(y: np.ndarray, kind: str, min_prominence: float) -> list[int]:
    s = y if kind == "max" else -y
    idx = [i for i in range(1, len(s) - 1) if s[i] >= s[i - 1] and s[i] > s[i + 1]]
    out = []
    for i in idx:
        left, right = s[: i + 1].min() if i else s[i], s[i:].min()
        # prominence against the higher of the two lowest points on each side, up to the next higher point
        lo = i
        while lo > 0 and s[lo - 1] <= s[i]:
            lo -= 1
        hi = i
        while hi < len(s) - 1 and s[hi + 1] <= s[i]:
            hi += 1
        prom = s[i] - max(s[lo:i + 1].min(), s[i:hi + 1].min())
        if prom >= min_prominence:
            out.append(i)
    return out


@dataclass
class Peak:
    position: float          # 0-1 along the axis, albumin side first
    height: float            # share of the tallest peak
    width_share_of_zone: float
    zone: str

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ElectrophoresisReading:
    zones: dict[str, float]                    # share of the area (%) per zone, from the curve
    peaks: list[Peak]
    narrow_peaks: list[Peak]                   # inside beta or gamma: to a person
    table_check: dict[str, Any] = field(default_factory=dict)
    source: str = ""                           # "vector" or "raster"
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"zones": self.zones, "peaks": [p.as_dict() for p in self.peaks],
                "narrow_peaks": [p.as_dict() for p in self.narrow_peaks], "table_check": self.table_check,
                "source": self.source, "notes": list(self.notes)}


def analyse_electrophoresis(xy: np.ndarray, table: dict[str, float] | None = None,
                            source: str = "") -> ElectrophoresisReading:
    """Zones, peaks and narrow peaks of a serum protein electrophoresis curve."""
    notes: list[str] = []
    x, y = resample(xy)
    y = _smooth(y) - _smooth(y).min()
    if y.max() <= 0:
        return ElectrophoresisReading({}, [], [], source=source, notes=("flat_curve",))
    y = y / y.max()
    # direction: albumin (the tallest peak) on the left
    if np.argmax(y) > len(y) / 2:
        y = y[::-1]
        notes.append("axis_reversed_albumin_right")
    valleys = _extrema(y, "min", 0.01)
    # the five zones are split by the four deepest valleys, kept in order
    if len(valleys) < 4:
        notes.append(f"only_{len(valleys)}_valleys_zones_not_all_separated")
    cuts = sorted(sorted(valleys, key=lambda i: y[i])[:4])
    edges = [0, *cuts, len(y) - 1]
    names = SPE_ZONES if len(edges) == 6 else tuple(f"zone{i + 1}" for i in range(len(edges) - 1))
    total = float(np.trapezoid(y)) if hasattr(np, "trapezoid") else float(np.trapz(y))
    area = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    zones = {names[i]: round(100.0 * float(area(y[edges[i]:edges[i + 1] + 1])) / total, 1)
             for i in range(len(edges) - 1)}
    peaks = []
    for i in _extrema(y, "max", 0.02):
        z = next(k for k in range(len(edges) - 1) if edges[k] <= i <= edges[k + 1])
        zone_width = edges[z + 1] - edges[z]
        half = y[i] - (y[i] - max(y[edges[z]], y[edges[z + 1]])) / 2
        lo = i
        while lo > edges[z] and y[lo] > half:
            lo -= 1
        hi = i
        while hi < edges[z + 1] and y[hi] > half:
            hi += 1
        peaks.append(Peak(round(i / (len(y) - 1), 3), round(float(y[i]), 3),
                          round((hi - lo) / max(zone_width, 1), 3), names[z]))
    narrow = [p for p in peaks if p.zone in ("beta", "gamma") and p.width_share_of_zone < NARROW_PEAK_SHARE]
    check: dict[str, Any] = {}
    if table:
        diffs = {k: round(zones[k] - v, 1) for k, v in table.items() if k in zones}
        check = {"differences": diffs,
                 "consistent": bool(diffs) and all(abs(d) <= TABLE_TOLERANCE for d in diffs.values())}
    return ElectrophoresisReading(zones, peaks, narrow, check, source, tuple(notes))


_FRACTION_NAMES = {"albumin": r"albumin", "alpha1": r"alfa\s*-?\s*1|alpha\s*-?\s*1|α\s*1",
                   "alpha2": r"alfa\s*-?\s*2|alpha\s*-?\s*2|α\s*2", "beta": r"^beta|β", "gamma": r"gamma|γ"}


def table_fractions(lab_rows) -> dict[str, float]:
    """The percentages of the fractions printed in the report's table (``lab_results`` rows with unit %)."""
    out: dict[str, float] = {}
    for r in lab_rows:
        if (r.unit or "").strip() != "%" or r.value is None:
            continue
        name = r.analyte.lower()
        for key, rx in _FRACTION_NAMES.items():
            if re.search(rx, name) and key not in out:
                out[key] = float(r.value)
    if "beta" not in out:
        betas = [float(r.value) for r in lab_rows if (r.unit or "") == "%" and r.value is not None
                 and re.search(r"beta\s*-?\s*[12]|β\s*[12]", r.analyte.lower())]
        if betas:
            out["beta"] = sum(betas)
    return out


__all__ = ["ElectrophoresisReading", "Peak", "analyse_electrophoresis", "is_trace", "raster_curve", "resample",
           "svg_curves", "table_fractions", "vector_curves"]


def read_document_traces(pdf: bytes | None, embedded_png: tuple[bytes, ...] = (), lab_rows=(),
                         electrophoresis: bool = False, max_pages: int = 10) -> dict[str, Any]:
    """Every trace of a document: vector curves of a digital PDF first, otherwise the pictures embedded in it,
    vectorised. With ``electrophoresis`` the curve is analysed and checked against the table; any other trace is
    kept as a series, not interpreted."""
    found: list[tuple[str, int, np.ndarray]] = []
    if pdf:
        for page in range(1, max_pages + 1):
            curves = vector_curves(pdf, page)
            found += [("vector", page, c) for c in curves]
            if not curves and page > 1:
                break
    if not found:
        for i, png in enumerate(embedded_png, start=1):
            try:
                import io

                from PIL import Image

                image = np.asarray(Image.open(io.BytesIO(png)).convert("L"), dtype=float)
            except Exception:  # noqa: BLE001 -- an unreadable picture is reported, not fatal
                continue
            curve = raster_curve(image)
            if len(curve) >= MIN_TRACE_POINTS and np.ptp(curve[:, 0]) >= MIN_TRACE_WIDTH_SHARE * image.shape[1]:
                found.append(("raster", i, curve))
    out: dict[str, Any] = {"traces": len(found), "items": []}
    table = table_fractions(lab_rows) if electrophoresis else {}
    for source, where, curve in found:
        item: dict[str, Any] = {"source": source, "where": where, "points": len(curve)}
        if electrophoresis:
            reading = analyse_electrophoresis(curve, table or None, source)
            item.update(kind="protein_electrophoresis", interpreted=True, reading=reading.as_dict())
        else:
            x, y = resample(curve, 200)
            item.update(kind="unknown", interpreted=False, series=[round(float(v), 4) for v in y])
        out["items"].append(item)
    if electrophoresis and not found:
        out["note"] = "electrophoresis_without_a_readable_trace"
    return out
