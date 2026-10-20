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
  part of the region; detached parts are reported beside it (``other_regions``). Every region has a density
  profile (``density``: distribution, composition, behaviour from core to border and around). How well the
  measure is supported is recorded (``measure_support``: the report's size, the second reading of
  ``threshold_check``, or a single reading). A finding goes to a person (``needs_review``, with the reasons) only
  when the lesion the report describes is not identified in the image (not found, several comparable regions,
  outside the named structure, the structure not checked) or the image differs from what the report writes (a
  size, a density or a composition clearly against the report's words);
* **the vectors**: the shape descriptors (no encoder behind them, comparable across lesions) and, once decision D21
  is taken, the text vector of the chosen encoder with its identity; the source text is always kept to recompute.

The image side also records every anatomical structure the lesion involves (``involvement``: inside, crossing
the border, in contact, with the shares and the gap) and its position in the structure that holds it (depth,
on the surface or not), and its margins (``margins``: sharpness of the border, spicules, lobulation) -- data for
the finding's branch, compared with the report's words where the report has them (18 October 2026).

The graph (FalkorDB): ``(:Patient)-[:HAS_STUDY]->(:Study)-[:HAS_FINDING]->(:Finding)-[:AT]->(:Structure)``, the
structures the image shows the lesion involves as ``(:Finding)-[:INVOLVES {relation, lesion_share,
surround_share, distance_mm, structure_share}]->(:Structure)``, and
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

from .lesion_measures import (RegionMeasures, inside_hull_share, label_parts, measure_region, overlap_inside,
                              split_main_part)
from .segmenters import Segmentation
from .density import DensityProfile, compare_with_report, density_profile
from .threshold_check import APPLIED, ground_glass_extent, second_reading
from .involvement import compare_position_with_report, involvement
from .margins import MarginProfile, compare_margins_with_report, margin_profile

AGREE, DISAGREE, NOT_FOUND, AMBIGUOUS, UNCHECKED, NO_IMAGE = (
    "agree", "disagree", "not_found", "ambiguous", "unchecked", "no_image")
IN_ORGAN_SHARE = 0.5          # share of the region inside the (grown) organ for "agree"
IN_HULL_SHARE, TOUCH_SHARE = 0.8, 0.2   # else: inside the organ's outline slice by slice, and touching the organ
SIZE_TOLERANCE_MM = 2.0       # text size vs measured size: within max(2 mm, 10%), the probe's strict bound (sixth probe: lung_016 passed at 2.7 mm and was 3.0 off) -- 30% let a 60 mm measure
SIZE_TOLERANCE_SHARE = 0.10   # pass for a 78 mm lesion (third probe), 15% a 49 mm one for 57 mm (fifth probe)
SECOND_COMPONENT_SHARE = 0.5  # a second separate region at least half the first: ambiguous
DETACHED_SHARE = 0.10         # a detached part of at least a tenth of the main one: perhaps a second lesion

# why a finding goes to a person
OUTSIDE_STRUCTURE, NOT_IN_IMAGE, SEVERAL_REGIONS = "outside_the_named_structure", "not_found_in_image", "several_regions"
SIZE_DIFFERS, CHARACTER_DIFFERS = "size_differs_from_report", "character_differs_from_report"
STRUCTURE_UNCHECKED = "structure_not_checked"

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
    measure_support: tuple[str, ...] = ()   # "report", "second_reading"; empty: a single reading
    other_regions: int = 0                  # detached parts large enough to be another lesion (not a reason alone)
    density: DensityProfile | None = None
    character_differences: tuple[str, ...] = ()
    with_ground_glass_axial_mm: float | None = None   # the lesion with its ground-glass part (aerated lung, CT)
    measured_on: str = "segmenter_region"             # the region the measures are taken on (always the segmenter's)
    involvement: tuple[dict[str, Any], ...] = ()      # every structure the region involves (``involvement``)
    position: dict[str, Any] = field(default_factory=dict)   # in the structure that holds it
    margins: MarginProfile | None = None

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
            "detached_parts": len(self.image.fragments), "other_regions": self.image.other_regions,
            "second_reading": json.dumps(self.image.second, default=float) if self.image.second else "",
            "measure_support": list(self.image.measure_support) or ["single_reading"],
            "character_differences": list(self.image.character_differences),
            "measured_on": self.image.measured_on,
        }
        if self.image.with_ground_glass_axial_mm is not None:
            row["with_ground_glass_axial_mm"] = self.image.with_ground_glass_axial_mm
        d = self.image.density
        if d is not None:
            row.update(density_units=d.units, density_median=d.percentiles["p50"], density_relative=d.relative,
                       density_composition=d.composition, density_flags=list(d.flags),
                       density_json=json.dumps(d.as_dict(), default=float))
            if d.solid_axial_mm is not None:
                row["solid_axial_mm"] = d.solid_axial_mm
        if m is not None and self.image.status in (AGREE, UNCHECKED, AMBIGUOUS):
            row.update(volume_mm3=m.volume_mm3, longest_mm=m.longest_diameter_mm,
                       longest_axial_mm=m.longest_axial_diameter_mm, centroid_mm=list(m.centroid_mm),
                       short_axial_mm=m.axial_short_mm if m.axial_short_mm is not None else -1.0,
                       mean_axial_mm=m.axial_mean_mm if m.axial_mean_mm is not None else -1.0,
                       sphericity=m.sphericity if m.sphericity is not None else -1.0,
                       frame=self.image.frame_of_reference, measures_json=json.dumps(m.as_dict()))
            if "mean" in m.intensity:
                row["intensity_mean"] = m.intensity["mean"]
        if self.image.involvement and self.image.status in (AGREE, UNCHECKED, AMBIGUOUS):
            row.update(involved_structures=[i["structure"] for i in self.image.involvement],
                       involvement_json=json.dumps(list(self.image.involvement)))
        if self.image.position and self.image.status in (AGREE, UNCHECKED, AMBIGUOUS):
            p = self.image.position
            row.update(host_structure=p["host"], gap_to_surface_mm=p["gap_to_surface_mm"],
                       touches_surface=p["touches_surface"], position_zone=p["zone"],
                       relative_depth=p["relative_depth"])
        g = self.image.margins
        if g is not None and self.image.status in (AGREE, UNCHECKED, AMBIGUOUS):
            row.update(margin_contour=g.contour, margin_sharpness=g.sharpness, margins_json=json.dumps(g.as_dict()))
            if g.transition_mm is not None:
                row["margin_transition_mm"] = g.transition_mm
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
                   frame_of_reference: str = "", modality: str = "", structure=None,
                   character: list[str] | tuple[str, ...] = (), contrast: str = "unknown",
                   organ_parts=None, structures: dict[str, np.ndarray] | None = None) -> ImageSide:
    """The image side of a finding: the checks of decision D19, the measures and the density of the region,
    and the reasons a person must look (the lesion is not identified, or the image differs from the report)."""
    if seg is None:
        return ImageSide(NO_IMAGE)
    if seg.mask is None or not seg.mask.any():
        return ImageSide(NOT_FOUND, seg.provider, seg.version, seg.prompt, notes=seg.notes, review=(NOT_IN_IMAGE,))
    if organ_mask is None and structures and isinstance(structure, str) and structure in structures:
        organ_mask = structures[structure]
    main, fragments = split_main_part(seg.mask, affine)
    seg_notes_extra: list[str] = []
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
        if status == DISAGREE and share >= TOUCH_SHARE:
            # a tumour replaces the tissue the organ mask is made of: check the outline of the organ's parts
            hull = inside_hull_share(main, organ_parts if organ_parts is not None else organ_mask)
            if hull >= IN_HULL_SHARE:
                status = AGREE
                seg_notes_extra.append(f"inside_the_outline_of_the_structure:{hull:.2f}")
        if status == DISAGREE:
            review.append(OUTSIDE_STRUCTURE)
    measures = measure_region(main, affine, image)
    whole = measure_region(seg.mask, affine).longest_axial_diameter_mm if fragments else measures.longest_axial_diameter_mm
    # in aerated lung the lesion may have a ground-glass part the segmenter left out: measured beside it
    extent, ext_notes = ground_glass_extent(main, image, affine, modality, organ_mask)
    ext_measures = None
    if extent is not None and "reaches_the_search_zone_edge" not in ext_notes.get("notes", ()):
        ext_measures = measure_region(extent, affine, image)
    region, measured_on = main, "segmenter_region"
    support: list[str] = []
    size_check = ""
    if text_size_mm:
        tol = max(SIZE_TOLERANCE_MM, SIZE_TOLERANCE_SHARE * text_size_mm)

        def error(m):
            return abs((m.longest_axial_diameter_mm or m.longest_diameter_mm) - text_size_mm)

        # The measure is the segmenter's region, whatever the report says. Until the ninth probe the reading
        # nearest the report's size was kept between this and the lesion with its ground-glass part (``extent``);
        # that is a choice made with the answer in hand: two windows of acceptance instead of one (on the 29
        # isolated lesions of the eighth and ninth probes the share of other lesions' reports passing the size
        # check went from 9.6% to 12.6%, for 4 more right ones passed in 29). The extension is kept as an
        # attribute (``with_ground_glass_axial_mm``) and as the evidence of a part-solid lesion, never chosen
        # by the report.
        size_check = "consistent" if error(measures) <= tol else "inconsistent"
        if size_check == "inconsistent":
            review.append(SIZE_DIFFERS)
        else:
            support.append("report")
    second = second_reading(region, image, affine, modality, organ_mask) if image is not None else None
    if second is not None and second.status == APPLIED and second.agree:
        support.append("second_reading")
    density = differences = None
    if image is not None:
        # the density of the whole lesion: with its ground-glass part when there is one (a part-solid lesion
        # measured on its solid part alone reads "solid")
        # fifth probe: computed on the extended region whenever there was one, two solid tumours read "part_solid"
        # (the extension had grown into the lung around them): the density is that of the region measured
        dens_region = region
        density = density_profile(image, dens_region, affine, modality, organ_mask, contrast=contrast)
        differences = compare_with_report(character, density)
    margins = margin_profile(region, affine, density) if image is not None else None
    involved, position = involvement(region, structures, affine) if structures else ([], None)
    differences = list(differences or ()) + compare_margins_with_report(character, margins) \
        + compare_position_with_report(character, position)
    if differences:
        review.append(CHARACTER_DIFFERS)
    frags = tuple({"voxels": f.voxels, "volume_mm3": round(f.volume_mm3, 1), "share_of_main": round(f.share_of_main, 4),
                   "distance_mm": round(f.distance_mm, 1)} for f in fragments[:20])
    others = sum(1 for f in fragments if f.share_of_main >= DETACHED_SHARE) if status != AMBIGUOUS else 0
    return ImageSide(status, seg.provider, seg.version, seg.prompt, measures, share, size_check, 1 + len(fragments),
                     frame_of_reference, tuple(seg.notes) + tuple(seg_notes_extra), tuple(dict.fromkeys(review)), frags,
                     whole,
                     second.as_dict() if second is not None else {}, tuple(support), others, density,
                     tuple(differences or ()),
                     ext_measures.longest_axial_diameter_mm if ext_measures is not None else None, measured_on,
                     tuple(i.as_dict() for i in involved), position.as_dict() if position is not None else {},
                     margins)


def build_finding(patient: str, study: str, study_date: str, modality: str, text: TextSide,
                  image: ImageSide) -> Finding:
    vectors: dict[str, list[float]] = {}
    if image.measures is not None and image.status in (AGREE, UNCHECKED) and not image.needs_review:
        shape = image.measures.shape_vector()
        if all(np.isfinite(shape)):   # a region too small for shape descriptors has no shape vector
            vectors["shape"] = shape
        if image.density is not None:
            vectors[f"density_{image.density.units.lower()}"] = image.density.vector()
    return Finding(finding_id(text.report_id, text.start, text.end), patient, study, study_date, modality, text,
                   image, vectors)


def load_findings(graph, findings: list[Finding]) -> int:
    """Write the findings (MERGE on the id: running twice changes nothing)."""
    rows = [f.as_row() for f in findings]
    edges = [{"id": f.id, **i} for f in findings if f.image.status in (AGREE, UNCHECKED, AMBIGUOUS)
             for i in f.image.involvement]
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
    for at in range(0, len(edges), 500):
        graph.query(
            "UNWIND $edges AS e "
            "MATCH (f:Finding {id: e.id}) "
            "MERGE (s:Structure {cid: e.structure}) "
            "MERGE (f)-[r:INVOLVES]->(s) "
            "SET r.relation = e.relation, r.lesion_share = e.lesion_share, r.surround_share = e.surround_share, "
            "r.distance_mm = e.distance_mm, r.structure_share = e.structure_share",
            params={"edges": edges[at:at + 500]},
        )
    return len(rows)


def findings_of(graph, patient: str) -> list[dict[str, Any]]:
    got = graph.query(
        "MATCH (:Patient {id: $p})-[:HAS_STUDY]->(st:Study)-[:HAS_FINDING]->(f:Finding) "
        "RETURN f ORDER BY st.date, f.id", params={"p": patient})
    return [dict(r[0].properties) for r in got.result_set]


__all__ = ["AGREE", "AMBIGUOUS", "CHARACTER_DIFFERS", "OUTSIDE_STRUCTURE", "SEVERAL_REGIONS", "SIZE_DIFFERS",
           "NOT_IN_IMAGE", "STRUCTURE_UNCHECKED", "DISAGREE", "NOT_FOUND", "NO_IMAGE", "UNCHECKED", "Finding", "ImageSide",
           "TextSide", "build_finding", "finding_id", "findings_of", "image_evidence", "load_findings",
           "sizes_in_text", "asdict"]
