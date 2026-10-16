"""The finding: what the report says about a place and what the image shows there, as one node of the graph
(16 October 2026).

"The report says what, the image says where and how" (decision of 2 October). A ``Finding`` joins:

* **the text**: the structure the linker accepted (a TotalSegmentator class), its roles, the qualifiers of the site
  (``site_qualifiers``: approximate or exact, degree and character, certainty), the sizes written in the clause
  ("8 mm", "2,1 x 1,5 cm"), the clause itself as the source of every vector;
* **the image**, when there is one: the region a segmenter isolated (``segmenters``), its measures and shape
  (``lesion_measures``), and the outcome of the checks (decision D19): ``agree`` (the region lies in the organ the
  report names), ``disagree`` (it does not: the image part is set aside, the finding lives on the text), ``not_found``,
  ``ambiguous`` (several separate regions of comparable size), ``unchecked`` (no organ mask to check against),
  ``no_image``. Absence of visual evidence is an explicit state, never zeros. Sizes are measured on the largest
  part of the region; detached parts are reported beside it. A finding goes to a person (``needs_review``, with
  the reasons) when the image does not confirm the report: outside the named structure, not found, several
  regions, a size that differs from the one written, a second reading (``threshold_check``) that disagrees, or a
  measure nothing independent confirms;
* **the vectors**: the shape descriptors (no encoder behind them, comparable across lesions) and, once decision D21
  is taken, the text vector of the chosen encoder with its identity; the source text is always kept to recompute.

The graph (FalkorDB): ``(:Patient)-[:HAS_STUDY]->(:Study)-[:HAS_FINDING]->(:Finding)-[:AT]->(:Structure)``, and
``(:Finding)-[:FOLLOWS]->(:Finding)`` between the exams of one person (``longitudinal``). Identifiers are
pseudonymous keys; no name, no date of birth, no free identifier reaches the graph.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .lesion_measures import RegionMeasures, label_parts, measure_region, overlap_inside, split_main_part
from .segmenters import Segmentation
from .threshold_check import APPLIED, second_reading

AGREE, DISAGREE, NOT_FOUND, AMBIGUOUS, UNCHECKED, NO_IMAGE = (
    "agree", "disagree", "not_found", "ambiguous", "unchecked", "no_image")
IN_ORGAN_SHARE = 0.5          # share of the region inside the (grown) organ for "agree"
SIZE_TOLERANCE_MM = 3.0       # text size vs measured size: within max(3 mm, 30%)
SIZE_TOLERANCE_SHARE = 0.30
SECOND_COMPONENT_SHARE = 0.5  # a second separate region at least half the first: ambiguous
DETACHED_SHARE = 0.10         # a detached part of at least a tenth of the main one: perhaps a second lesion

# why a finding goes to a person
OUTSIDE_STRUCTURE, NOT_IN_IMAGE, SEVERAL_REGIONS = "outside_the_named_structure", "not_found_in_image", "several_regions"
SIZE_DIFFERS, SECOND_DISAGREES = "size_differs_from_report", "second_reading_disagrees"
DETACHED_PARTS, UNCONFIRMED, STRUCTURE_UNCHECKED = "detached_parts", "measure_unconfirmed", "structure_not_checked"

_NUM = r"\d+(?:[.,]\d+)?"
_SIZE_RE = re.compile(rf"(?<![\w.,])(?P<a>{_NUM})(?:\s*[x×]\s*(?P<b>{_NUM}))?(?:\s*[x×]\s*(?P<c>{_NUM}))?\s*"
                      r"(?P<u>mm|cm)\b", re.IGNORECASE)


def sizes_in_text(text: str) -> list[dict[str, Any]]:
    """Every size written in the text, in millimetres, with the words and their offsets ("2,1 x 1,5 cm")."""
    out = []
    for m in _SIZE_RE.finditer(text):
        factor = 10.0 if m.group("u").lower() == "cm" else 1.0
        dims = [float(v.replace(",", ".")) * factor for v in (m.group("a"), m.group("b"), m.group("c")) if v]
        out.append({"text": m.group(0), "start": m.start(), "end": m.end(), "mm": dims, "largest_mm": max(dims)})
    return out


@dataclass
class TextSide:
    report_id: str
    clause: str                      # pseudonymised clause: the source of the text vector and of the prompt
    mention: str
    start: int
    end: int
    structure: str                   # TotalSegmentator class the linker accepted
    roles: tuple[str, ...] = ()
    qualifiers: dict[str, Any] = field(default_factory=dict)
    sizes: list[dict[str, Any]] = field(default_factory=list)
    language: str = "it"

    @property
    def largest_size_mm(self) -> float | None:
        return max((s["largest_mm"] for s in self.sizes), default=None)


@dataclass
class ImageSide:
    status: str
    provider: str = ""
    version: str = ""
    prompt: str = ""
    measures: RegionMeasures | None = None
    in_organ_share: float | None = None
    size_check: str = ""             # "consistent", "inconsistent", "" (nothing to compare)
    components: int = 0
    frame_of_reference: str = ""
    notes: tuple[str, ...] = ()
    review: tuple[str, ...] = ()     # reasons for a person; empty: the image confirms the report
    fragments: tuple[dict[str, float], ...] = ()
    whole_axial_mm: float | None = None   # longest axial diameter of the whole region, detached parts included
    second: dict[str, Any] = field(default_factory=dict)   # the threshold reading (threshold_check)

    @property
    def needs_review(self) -> bool:
        return bool(self.review)


@dataclass
class Finding:
    id: str
    patient: str                     # pseudonymous key
    study: str                       # pseudonymous key
    study_date: str                  # as given by the caller (ISO date; may be shifted for pseudonymisation)
    modality: str
    text: TextSide
    image: ImageSide
    vectors: dict[str, list[float]] = field(default_factory=dict)   # name -> vector ("shape", "text:<encoder>")

    def as_row(self) -> dict[str, Any]:
        q = self.text.qualifiers or {}
        m = self.image.measures
        row = {
            "id": self.id, "patient": self.patient, "study": self.study, "study_date": self.study_date,
            "modality": self.modality, "report": self.text.report_id, "structure": self.text.structure,
            "mention": self.text.mention, "clause": self.text.clause, "roles": list(self.text.roles),
            "language": self.text.language,
            "location": q.get("location", ""), "degree": q.get("degree", ""), "character": list(q.get("character", [])),
            "certainty": q.get("certainty", ""), "polarity": q.get("polarity", ""),
            "text_sizes_mm": [s["largest_mm"] for s in self.text.sizes],
            "image_status": self.image.status, "image_provider": self.image.provider,
            "image_version": self.image.version, "size_check": self.image.size_check,
            "in_organ_share": self.image.in_organ_share if self.image.in_organ_share is not None else -1.0,
            "image_notes": list(self.image.notes),
            "needs_review": self.image.needs_review, "review_reasons": list(self.image.review),
            "detached_parts": len(self.image.fragments),
            "second_reading": json.dumps(self.image.second) if self.image.second else "",
        }
        if m is not None and self.image.status in (AGREE, UNCHECKED, AMBIGUOUS):
            row.update(volume_mm3=m.volume_mm3, longest_mm=m.longest_diameter_mm,
                       longest_axial_mm=m.longest_axial_diameter_mm, centroid_mm=list(m.centroid_mm),
                       sphericity=m.sphericity if m.sphericity is not None else -1.0,
                       frame=self.image.frame_of_reference, measures_json=json.dumps(m.as_dict()))
            if "mean" in m.intensity:
                row["intensity_mean"] = m.intensity["mean"]
        for name, vec in self.vectors.items():
            row[f"vec_{re.sub(r'[^a-z0-9]+', '_', name.lower())}"] = [float(x) for x in vec]
        return row


def finding_id(report_id: str, start: int, end: int) -> str:
    return hashlib.sha256(f"{report_id}|{start}|{end}".encode()).hexdigest()[:20]


def _components(mask: np.ndarray) -> list[int]:
    """Sizes of the 6-connected parts of the region, largest first (scipy when installed, else plain numpy:
    scipy is not a dependency of the project, and a count that silently says "one part" hid two regions)."""
    return label_parts(mask)[1]


def _components_numpy(mask: np.ndarray) -> list[int]:
    from collections import deque

    seen = np.zeros_like(mask, dtype=bool)
    sizes = []
    for start in map(tuple, np.argwhere(mask)):
        if seen[start]:
            continue
        seen[start] = True
        queue, size = deque([start]), 0
        while queue:
            z, y, x = queue.popleft()
            size += 1
            for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                n = (z + dz, y + dy, x + dx)
                if all(0 <= n[i] < mask.shape[i] for i in range(3)) and mask[n] and not seen[n]:
                    seen[n] = True
                    queue.append(n)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def image_evidence(seg: Segmentation | None, affine, image: np.ndarray | None = None,
                   organ_mask: np.ndarray | None = None, text_size_mm: float | None = None,
                   frame_of_reference: str = "", modality: str = "", structure=None) -> ImageSide:
    """The image side of a finding, with the checks of decision D19 and the reasons a person must look."""
    if seg is None:
        return ImageSide(NO_IMAGE)
    if seg.mask is None or not seg.mask.any():
        return ImageSide(NOT_FOUND, seg.provider, seg.version, seg.prompt, notes=seg.notes, review=(NOT_IN_IMAGE,))
    main, fragments = split_main_part(seg.mask, affine)
    share = overlap_inside(seg.mask, organ_mask) if organ_mask is not None and np.any(organ_mask) else None
    review: list[str] = []
    if fragments and fragments[0].share_of_main >= SECOND_COMPONENT_SHARE:
        status = AMBIGUOUS
        review.append(SEVERAL_REGIONS)
    elif share is None:
        status = UNCHECKED
        review.append(STRUCTURE_UNCHECKED)
    else:
        status = AGREE if share >= IN_ORGAN_SHARE else DISAGREE
        if status == DISAGREE:
            review.append(OUTSIDE_STRUCTURE)
    if any(f.share_of_main >= DETACHED_SHARE for f in fragments) and status != AMBIGUOUS:
        review.append(DETACHED_PARTS)
    measures = measure_region(main, affine, image)
    whole = measure_region(seg.mask, affine).longest_axial_diameter_mm if fragments else measures.longest_axial_diameter_mm
    size_check = ""
    if text_size_mm:
        measured = measures.longest_axial_diameter_mm or measures.longest_diameter_mm
        tol = max(SIZE_TOLERANCE_MM, SIZE_TOLERANCE_SHARE * text_size_mm)
        size_check = "consistent" if abs(measured - text_size_mm) <= tol else "inconsistent"
        if size_check == "inconsistent":
            review.append(SIZE_DIFFERS)
    second = second_reading(main, image, affine, modality, structure) if image is not None else None
    if second is not None and second.status == APPLIED and not second.agree:
        review.append(SECOND_DISAGREES)
    confirmed = size_check == "consistent" or (second is not None and second.agree is True)
    if status in (AGREE, UNCHECKED) and not confirmed:
        review.append(UNCONFIRMED)
    frags = tuple({"voxels": f.voxels, "volume_mm3": round(f.volume_mm3, 1), "share_of_main": round(f.share_of_main, 4),
                   "distance_mm": round(f.distance_mm, 1)} for f in fragments[:20])
    return ImageSide(status, seg.provider, seg.version, seg.prompt, measures, share, size_check, 1 + len(fragments),
                     frame_of_reference, seg.notes, tuple(dict.fromkeys(review)), frags, whole,
                     second.as_dict() if second is not None else {})


def build_finding(patient: str, study: str, study_date: str, modality: str, text: TextSide,
                  image: ImageSide) -> Finding:
    vectors: dict[str, list[float]] = {}
    if image.measures is not None and image.status in (AGREE, UNCHECKED) and not image.needs_review:
        shape = image.measures.shape_vector()
        if all(np.isfinite(shape)):   # a region too small for shape descriptors has no shape vector
            vectors["shape"] = shape
    return Finding(finding_id(text.report_id, text.start, text.end), patient, study, study_date, modality, text,
                   image, vectors)


def load_findings(graph, findings: list[Finding]) -> int:
    """Write the findings (MERGE on the id: running twice changes nothing)."""
    rows = [f.as_row() for f in findings]
    for at in range(0, len(rows), 200):
        graph.query(
            "UNWIND $rows AS row "
            "MERGE (p:Patient {id: row.patient}) "
            "MERGE (st:Study {id: row.study}) SET st.date = row.study_date, st.modality = row.modality "
            "MERGE (p)-[:HAS_STUDY]->(st) "
            "MERGE (f:Finding {id: row.id}) SET f += row "
            "MERGE (st)-[:HAS_FINDING]->(f) "
            "MERGE (s:Structure {cid: row.structure}) "
            "MERGE (f)-[:AT]->(s)",
            params={"rows": rows[at:at + 200]},
        )
    return len(rows)


def findings_of(graph, patient: str) -> list[dict[str, Any]]:
    got = graph.query(
        "MATCH (:Patient {id: $p})-[:HAS_STUDY]->(st:Study)-[:HAS_FINDING]->(f:Finding) "
        "RETURN f ORDER BY st.date, f.id", params={"p": patient})
    return [dict(r[0].properties) for r in got.result_set]


__all__ = ["AGREE", "AMBIGUOUS", "DETACHED_PARTS", "OUTSIDE_STRUCTURE", "SECOND_DISAGREES", "SEVERAL_REGIONS",
           "SIZE_DIFFERS", "UNCONFIRMED", "NOT_IN_IMAGE", "STRUCTURE_UNCHECKED", "DISAGREE", "NOT_FOUND", "NO_IMAGE", "UNCHECKED", "Finding", "ImageSide",
           "TextSide", "build_finding", "finding_id", "findings_of", "image_evidence", "load_findings",
           "sizes_in_text", "asdict"]
