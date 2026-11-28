"""The discordances of a case (28 November 2026): what the reading of the case sends to a person, counted by kind.

They replace the coherence between "areas" (``area_coherence``, removed): a measure built from the words two
summaries shared and from weights written by hand, never measured. These are concrete and each has its sentence:

* ``card``: a field the two interpreters read differently, a finding only one read, a part the heading contradicts;
* ``reference``: a measure the interpreters compared with different rows of the library;
* ``image``: what the image shows against what the report says (the image side's review);
* ``document``: a document whose identity is in doubt, of no recognised kind, or set aside;
* ``unread``: a document no interpreter could read (none configured, or its reading failed);
* ``term``: a mention the previous reader (``narrative_reader``) could not link to one concept.

The principle of doubt (7 November 2026): a doubt stays in the diagnostic path; a discordance goes to a person. The
count decides how many reasoning paths a case needs (``model_router.pick_mode``) and what is escalated
(``safety.rails``); it never blocks a result.
"""

from __future__ import annotations

from typing import Any

from ..imaging.findings import (
    CHARACTER_DIFFERS,
    NOT_IN_IMAGE,
    OUTSIDE_STRUCTURE,
    SEVERAL_REGIONS,
    SIZE_DIFFERS,
    STRUCTURE_UNCHECKED,
)

KINDS = ("card", "reference", "image", "document", "unread", "term")
_IMAGE = {CHARACTER_DIFFERS, NOT_IN_IMAGE, OUTSIDE_STRUCTURE, SEVERAL_REGIONS, SIZE_DIFFERS, STRUCTURE_UNCHECKED}


def kind_of(reason: str) -> str:
    """The kind of a reason written by ``data.report_reading`` (and ``reading.case_sides``)."""
    reason = str(reason or "")
    if reason.startswith("document_"):
        return "document"
    if reason == "doubt:reference_chosen_differently":
        return "reference"
    if reason.startswith("doubt:"):
        return "card"
    if reason.startswith("not read"):
        return "unread"
    parts = [part.split(":", 1)[0] for part in reason.split(",")]
    if any(part in _IMAGE or part == "is_the_finding_of_this_structure" for part in parts):
        return "image"
    return "term"


def case_discordances(provenance: dict[str, Any] | None) -> dict[str, Any]:
    """The discordances of the case's reading (``provenance["report_reading"]["to_a_person"]``)."""
    reading = (provenance or {}).get("report_reading") or {}
    items = [d for d in reading.get("to_a_person", []) or [] if isinstance(d, dict)]
    kinds = dict.fromkeys(KINDS, 0)
    for d in items:
        kinds[kind_of(d.get("reason", ""))] += 1
    return {"count": len(items), "kinds": kinds,
            "reports": sorted({str(d.get("report", "")) for d in items if d.get("report")})}


__all__ = ["KINDS", "case_discordances", "kind_of"]
