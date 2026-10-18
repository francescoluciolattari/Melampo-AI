"""The imaging interpreter: from a report and its exam to the findings of the graph (16 October 2026).

For every site the linker accepted in an imaging report (``report_graph.site_row`` rows, with the qualifiers of
``site_qualifiers``), a finding is made when the clause *describes* something at that site: a size, a radiological
character ("ipodensa", "hyperintense"), a degree, or a hedged statement. A bare site ("fegato nei limiti") stays a
SITE edge and is not a finding; a negated one ("non lesioni focali epatiche") is not a finding either (it is
already recorded, negated, on the SITE edge).

When the exam's volume is there, the lesion segmenter (VoxTell) is prompted with the clause itself -- the
pseudonymised words of the report, the free text VoxTell is built for (decision D20: the prompt comes from the
reading of the report; the language is to be measured) -- and the organ segmenter (TotalSegmentator) gives the
organ mask of the structure the linker named, for the check. A segmenter that is not installed or cannot run
leaves the finding on the text with the reason written (``segmenter_unavailable``), never a silent gap.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..memory.site_qualifiers import read_site_qualifiers
from .findings import NO_IMAGE, Finding, ImageSide, TextSide, build_finding, image_evidence, sizes_in_text
from .segmenters import SegmenterUnavailable


def _describes(qualifiers: dict[str, Any], sizes: list) -> bool:
    if qualifiers.get("polarity") == "negated":
        return False
    return bool(sizes or qualifiers.get("character") or qualifiers.get("degree")
                or qualifiers.get("certainty") in ("possible", "hypothetical"))


def interpret_exam(patient: str, study: str, study_date: str, modality: str, report_id: str, report_text: str,
                   sites: list[dict[str, Any]], language: str = "it", volume: np.ndarray | None = None,
                   affine=None, frame_of_reference: str = "", lesion_segmenter=None,
                   organ_segmenter=None, contrast: str = "unknown",
                   structures: dict[str, np.ndarray] | None = None) -> list[Finding]:
    """The findings of one exam. ``sites``: rows with mention, start, end (offsets in ``report_text``), cid, roles.
    ``structures``: every anatomical mask of the exam (name -> mask), for the structures each lesion involves;
    when not given and the organ segmenter can segment the whole body (``all_structures``), it is asked once."""
    findings: list[Finding] = []
    asked = structures is not None
    for site in sites:
        if "procedure_site" in (site.get("roles") or ()):
            continue  # "TC del torace": the examined structure is not a finding
        start, end = int(site["start"]), int(site["end"])
        q = read_site_qualifiers(report_text, start, end, language)
        lo, hi = q.clause
        clause = report_text[lo:hi].strip()
        sizes = sizes_in_text(clause)
        qd = q.as_dict()
        if not _describes(qd, sizes):
            continue
        text = TextSide(report_id, clause, site.get("mention", report_text[start:end]), start, end, site["cid"],
                        tuple(site.get("roles") or ()), qd, sizes, language)
        image = ImageSide(NO_IMAGE)
        if volume is not None and affine is not None and lesion_segmenter is not None:
            try:
                seg = lesion_segmenter.segment(volume, affine, [clause])[0]
                if not asked and organ_segmenter is not None and hasattr(organ_segmenter, "all_structures"):
                    asked = True
                    try:
                        structures = organ_segmenter.all_structures(volume, affine)
                    except SegmenterUnavailable:
                        structures = None
                organ = None
                if structures and site["cid"] in structures:
                    organ = structures[site["cid"]]
                elif organ_segmenter is not None:
                    organ = organ_segmenter.segment(volume, affine, [site["cid"]])[0].mask
                image = image_evidence(seg, affine, volume, organ, text.largest_size_mm, frame_of_reference,
                                       modality=modality, structure=site["cid"],
                                       character=qd.get("character", ()), contrast=contrast,
                                       structures=structures)
            except SegmenterUnavailable as error:
                image = ImageSide(NO_IMAGE, notes=(f"segmenter_unavailable:{error}",))
        findings.append(build_finding(patient, study, study_date, modality, text, image))
    return findings


__all__ = ["interpret_exam"]
