"""The findings of one person, linked from one exam to the next (16 October 2026).

Two exams of the same (pseudonymous) person: which finding of the later one is the finding of the earlier one?
Each pair is scored on what both exams say, and only what they say:

* **the structure** must be the same (the linker's TotalSegmentator class): a nodule of the right upper lobe never
  follows one of the left lower lobe;
* **the side and the place**: when both images share the scanner's frame of reference, the distance between the
  centroids in mm; otherwise the position *inside the organ* (centroid relative to the organ's bounding box, which
  survives a different patient position), or nothing when there is no image;
* **the size**: a plausible change (a lesion that went from 8 mm to 80 mm in a month is not the same lesion
  without saying so);
* **the words**: the same character (density, signal) agrees, a different one does not rule out (a lesion can
  calcify).

Pairs are assigned one to one, best first; when two candidates are within ``TIE_MARGIN`` of each other the link is
written with ``ambiguous = true`` and goes to a person. Nothing is merged: two findings stay two nodes, linked by
``FOLLOWS`` with the interval in days, the change in the longest diameter, the volume ratio and the volume
doubling time (Schwartz: VDT = dt * ln 2 / ln(V2/V1)), and whether the two acquisitions are comparable for
measuring change (``dicom_volume.compare_acquisitions``: same thickness, kernel and scanner model, as QIBA asks).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

from .findings import AGREE, UNCHECKED, Finding

MIN_SCORE = 0.5
TIE_MARGIN = 0.1
SAME_FRAME_DISTANCE_MM = 15.0      # centroids within this distance in one frame of reference: same place
IN_ORGAN_DISTANCE = 0.25           # relative position inside the organ (0-1 per axis) within this: same place
MAX_GROWTH_PER_MONTH = 3.0         # longest diameter ratio per 30 days beyond which the pair is implausible


@dataclass
class Follow:
    earlier: str
    later: str
    score: float
    days: int | None
    diameter_change_pct: float | None
    volume_ratio: float | None
    doubling_time_days: float | None
    comparable_acquisitions: bool | None
    ambiguous: bool
    reasons: tuple[str, ...]

    def as_row(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else (-1 if v is None else v)) for k, v in self.__dict__.items()}


def _days(a: str, b: str) -> int | None:
    try:
        return (date.fromisoformat(b[:10]) - date.fromisoformat(a[:10])).days
    except (TypeError, ValueError):
        return None


def _size(f: Finding) -> float | None:
    m = f.image.measures
    if m is not None and f.image.status in (AGREE, UNCHECKED):
        return m.longest_axial_diameter_mm or m.longest_diameter_mm
    return f.text.largest_size_mm


def _volume(f: Finding) -> float | None:
    m = f.image.measures
    return m.volume_mm3 if m is not None and f.image.status in (AGREE, UNCHECKED) else None


def pair_score(a: Finding, b: Finding, relative_positions: dict[str, tuple[float, float, float]] | None = None,
               days: int | None = None) -> tuple[float, list[str]]:
    """How much the later finding ``b`` looks like the earlier ``a`` (0 when it cannot be the same)."""
    reasons: list[str] = []
    if a.text.structure != b.text.structure:
        return 0.0, ["different_structure"]
    if a.text.qualifiers.get("polarity") == "negated" or b.text.qualifiers.get("polarity") == "negated":
        return 0.0, ["negated_finding"]
    score, weight = 0.4, 0.4   # same structure
    reasons.append("same_structure")
    ma, mb = a.image.measures, b.image.measures
    rel = relative_positions or {}
    if ma is not None and mb is not None and a.image.frame_of_reference and \
            a.image.frame_of_reference == b.image.frame_of_reference:
        d = math.dist(ma.centroid_mm, mb.centroid_mm)
        s = max(0.0, 1.0 - d / (2 * SAME_FRAME_DISTANCE_MM))
        score, weight = score + 0.35 * s, weight + 0.35
        reasons.append(f"centroid_distance_mm:{d:.1f}")
    elif a.id in rel and b.id in rel:
        d = math.dist(rel[a.id], rel[b.id])
        s = max(0.0, 1.0 - d / (2 * IN_ORGAN_DISTANCE))
        score, weight = score + 0.35 * s, weight + 0.35
        reasons.append(f"position_in_organ:{d:.2f}")
    sa, sb = _size(a), _size(b)
    if sa and sb:
        ratio = max(sa, sb) / min(sa, sb)
        months = max((days or 30) / 30.0, 1.0)
        plausible = ratio <= MAX_GROWTH_PER_MONTH ** min(months, 12)
        s = 1.0 if plausible else 0.0
        score, weight = score + 0.15 * s, weight + 0.15
        reasons.append(f"size_ratio:{ratio:.2f}" + ("" if plausible else ":implausible"))
    ca, cb = set(a.text.qualifiers.get("character", [])), set(b.text.qualifiers.get("character", []))
    if ca and cb:
        s = 1.0 if ca & cb else 0.5
        score, weight = score + 0.10 * s, weight + 0.10
        reasons.append("same_character" if ca & cb else "different_character")
    return score / weight, reasons


def link_exams(earlier: list[Finding], later: list[Finding], comparable_acquisitions: bool | None = None,
               relative_positions: dict[str, tuple[float, float, float]] | None = None) -> list[Follow]:
    """One-to-one links from the findings of the earlier exam to those of the later one."""
    candidates = []
    for b in later:
        for a in earlier:
            days = _days(a.study_date, b.study_date)
            s, why = pair_score(a, b, relative_positions, days)
            if s >= MIN_SCORE:
                candidates.append((s, a, b, days, why))
    candidates.sort(key=lambda c: -c[0])
    used_a, used_b, out = set(), set(), []
    for s, a, b, days, why in candidates:
        if a.id in used_a or b.id in used_b:
            continue
        rivals = [c for c in candidates if c[0] >= s - TIE_MARGIN and (c[1] is not a or c[2] is not b)
                  and (c[1] is a or c[2] is b) and c[1].id not in used_a and c[2].id not in used_b]
        used_a.add(a.id)
        used_b.add(b.id)
        sa, sb, va, vb = _size(a), _size(b), _volume(a), _volume(b)
        change = 100.0 * (sb - sa) / sa if sa and sb else None
        ratio = vb / va if va and vb else None
        vdt = (days * math.log(2) / math.log(ratio)) if ratio and days and ratio != 1.0 else None
        out.append(Follow(a.id, b.id, round(s, 3), days, change, ratio, vdt, comparable_acquisitions,
                          bool(rivals), tuple(why)))
    return out


def load_follows(graph, follows: list[Follow]) -> int:
    rows = [f.as_row() for f in follows]
    if rows:
        graph.query(
            "UNWIND $rows AS row "
            "MATCH (a:Finding {id: row.earlier}), (b:Finding {id: row.later}) "
            "MERGE (b)-[r:FOLLOWS]->(a) SET r += row",
            params={"rows": rows},
        )
    return len(rows)


def relative_position(centroid_mm, organ_bbox_mm) -> tuple[float, float, float]:
    """Where a point lies inside an organ's bounding box, 0-1 on each axis (survives a change of patient position)."""
    lo, hi = organ_bbox_mm
    return tuple((c - lo_) / (hi_ - lo_) if hi_ > lo_ else 0.5 for c, lo_, hi_ in zip(centroid_mm, lo, hi, strict=True))


__all__ = ["Follow", "link_exams", "load_follows", "pair_score", "relative_position"]
