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

from dataclasses import replace
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
                   structures: dict[str, np.ndarray] | None = None, ownership=None,
                   set_aside: list | None = None, questions: list | None = None,
                   statements: list | None = None) -> list[Finding]:
    """The findings of one exam. ``sites``: rows with mention, start, end (offsets in ``report_text``), cid, roles.
    ``structures``: every anatomical mask of the exam (name -> mask), for the structures each lesion involves;
    when not given and the organ segmenter can segment the whole body (``all_structures``), it is asked once.
    ``ownership``: a ``finding_owner.FindingOwner`` (31 October 2026): is the clause about this structure, or does
    the structure only place something else? When the reader and the model agree that it is not, no finding is
    made at the site and the clause is appended to ``set_aside`` for a person; when they do not agree, the
    finding is made and goes to a person. Either way the question for the person, complete in itself
    (``review_question``: kind, sentence, structure, closed answers, the readings to show after the answer), is
    appended to ``questions``.

    When the owner reads what the sentence says of the structure (``question="statement"``, 1 November 2026),
    what is attached is that statement, with its piece of the sentence: a finding (or an involvement) is a
    finding made from the piece, not the whole sentence; a measure is a finding that is only a measure (no
    phenotype is looked for in it); a fact of history and a normal statement are not current findings and are
    appended to ``statements``; "nothing" leaves the finding as it was."""
    findings: list[Finding] = []
    asked = structures is not None
    for site in sites:
        if "procedure_site" in (site.get("roles") or ()):
            continue  # "TC del torace": the examined structure is not a finding
        start, end = int(site["start"]), int(site["end"])
        q = read_site_qualifiers(report_text, start, end, language)
        lo, hi = q.clause
        raw = report_text[lo:hi]
        clause = raw.strip()
        sizes = sizes_in_text(clause)
        qd = q.as_dict()
        if not _describes(qd, sizes):
            continue
        owner = None
        if ownership is not None:
            from ..memory.finding_owner import report_context
            from ..memory.review_question import build_question

            at = start - lo - (len(raw) - len(raw.lstrip()))
            mention = site.get("mention", report_text[start:end])
            context = report_context(report_text, lo)
            owner = ownership(mention, clause, at, language, cid=site["cid"], context=context)
            question = None
            if owner.verdict != "here" or getattr(owner, "review", ""):
                name = getattr(ownership, "linked_name", lambda c: c.replace("_", " "))(site["cid"])
                question = build_question(owner, report=report_id, language=language, sentence=clause,
                                          mention=mention, start=at, structure=site["cid"], structure_name=name,
                                          context=context).as_dict()
                if questions is not None:
                    questions.append(question)
            if owner.verdict == "elsewhere":
                if set_aside is not None:
                    set_aside.append({"mention": mention, "structure": site["cid"], "clause": clause,
                                      "reason": "the_structure_only_places_something_else", **owner.as_dict(),
                                      "question": question["id"]})
                continue
        statement = owner.statement if owner is not None and owner.verdict == "here" else ""
        if owner is not None and owner.claims and owner.verdict == "here":
            # claim by claim (2 November 2026): the claims that are not a current finding go to ``statements``;
            # a finding is made from its own words, with the certainty the sentence gives it
            mention = site.get("mention", report_text[start:end])
            other = {k: owner.claims[k] for k in ("normal", "past", "device", "not_assessable", "indication",
                                                   "procedure") if owner.claims.get(k)}
            if statements is not None:
                for kind, words in other.items():
                    statements.append({"mention": mention, "structure": site["cid"], "statement":
                                       {"past": "history"}.get(kind, kind), "piece": " … ".join(words),
                                       "clause": clause, "report": report_id})
            if statements is not None:
                # a measure compared with a reference value (4 November 2026): the row cited and what it says,
                # checked by the reader before anything was attached; never a finding the sentence does not state
                for ref in owner.claims.get("measure_refs", []):
                    statements.append({"mention": mention, "structure": site["cid"], "statement": "measure",
                                       "piece": ref["words"], "reference": ref["reference"],
                                       "against": ref["against"], "clause": clause, "report": report_id})
            claimed = owner.claims.get("findings", [])
            measures = owner.claims.get("measure", [])
            if claimed:
                words = [o["words"] for o in claimed]
                clause, sizes = " … ".join(words), sizes_in_text(" ".join(words + measures))
                statement = "finding" if any(o.get("of") == "it" for o in claimed) else "involved"
                if owner.claims.get("alternatives"):
                    qd = {**qd, "certainty": "hypothetical"}
                elif owner.claims.get("certainty"):
                    qd = {**qd, "certainty": "possible"}
            elif measures:
                clause, sizes, statement = " … ".join(measures), sizes_in_text(" ".join(measures)), "measure"
                qd = {**qd, "character": [], "degree": ""}
            elif other:
                continue            # only what is not a current finding: it is in ``statements``
            else:
                statement = "nothing"   # the sentence states nothing of it: the finding stays as it was
        if statement in ("history", "normal"):
            if statements is not None:
                statements.append({"mention": site.get("mention", report_text[start:end]), "structure": site["cid"],
                                   "statement": statement, "piece": owner.piece, "clause": clause,
                                   "report": report_id})
            continue
        if statement in ("finding", "measure") and owner.piece:
            clause, sizes = owner.piece, sizes_in_text(owner.piece)
            if statement == "measure":
                qd = {**qd, "character": [], "degree": ""}
        text = TextSide(report_id, clause, site.get("mention", report_text[start:end]), start, end, site["cid"],
                        tuple(site.get("roles") or ()), qd, sizes, language, statement)
        image = ImageSide(NO_IMAGE)
        if volume is not None and affine is not None and lesion_segmenter is not None:
            try:
                seg = lesion_segmenter.segment(volume, affine, [clause])[0]
                if not asked and organ_segmenter is not None and hasattr(organ_segmenter, "all_structures"):
                    asked = True
                    try:
                        try:
                            structures = organ_segmenter.all_structures(volume, affine, modality=modality)
                        except TypeError:   # a segmenter whose atlas does not depend on the modality
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
        if owner is not None and owner.verdict == "unsure":
            image = replace(image, review=(*image.review, f"is_the_finding_of_this_structure:{owner.reason}"))
        if owner is not None and getattr(owner, "review", ""):
            image = replace(image, review=(*image.review, owner.review))
        findings.append(build_finding(patient, study, study_date, modality, text, image))
    return findings


__all__ = ["interpret_exam"]
