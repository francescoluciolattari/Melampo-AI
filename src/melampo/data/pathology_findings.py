"""The diagnosis of a pathology report given to its findings (8 November 2026).

A pathologist's report is read twice: by the pathology reader (``pathology_report``: sections, TNM, grades,
margins, nodes, invasion, markers, each with its words and offsets) and by the narrative reader (the sites of the
text, as for any report). Until now the two readings travelled side by side: the finding of "adenocarcinoma of the
sigmoid colon, pT3 pN1a, margins free" did not know its TNM, its grade or its margins, and a site named in the
clinical information ("suspected neoplasm of the rectum") was a finding as much as the diagnosis.

Here they meet, as a clinician reads the report, with what is certain only:

* **the section of each finding** (the pathology reader's sections): a finding in the clinical information is the
  question asked, never a current finding (it becomes a statement, ``indication``), as a referral is in a
  radiology report; a site of the material is where the sample comes from (``specimen``); a finding of the
  diagnosis is the pathologist's diagnosis (``pathology.diagnosis``), the others (macroscopy, microscopy, comment)
  stay findings with their section;
* **the items** (TNM, grade, margins, nodes, invasion, markers, molecular) go to the finding of the diagnosis they
  belong to: an item written in the diagnosis to the finding whose sentence holds it, or the one before it in the
  same line; an item of another section (immunohistochemistry, molecular) only when the diagnosis names one
  structure. Otherwise the item stays with the report, ``unplaced``, and says why: nothing is attached by guess;
* **a diagnosis that names no structure** ("Adenocarcinoma G2" after "Material: biopsy of the sigmoid colon") is of
  the structure of the material, when the material names exactly one.

Values are never converted and never inferred from one another (a pT3 does not make a stage).
"""

from __future__ import annotations

from typing import Any

INDICATION = "indication"
SPECIMEN = "specimen"
# the pathology reader's sections that are not current findings
_NOT_A_FINDING = {"clinical": INDICATION, "material": SPECIMEN}


def _section(sections: dict[str, list[int]], at: int) -> str:
    """The pathology reader's section of a position ("header" before the first heading)."""
    for name, (lo, hi) in sections.items():
        if lo <= at < hi:
            return name
    return "header"


def _line(text: str, at: int) -> tuple[int, int]:
    lo = text.rfind("\n", 0, at) + 1
    hi = text.find("\n", at)
    return lo, len(text) if hi < 0 else hi


def _item(i: dict[str, Any]) -> dict[str, Any]:
    return {"kind": i["kind"], "text": i["text"], "value": dict(i.get("value") or {}), "section": i["section"]}


def give_the_diagnosis(rows: list[dict[str, Any]], statements: list[dict[str, Any]], text: str,
                       pathology: dict[str, Any], specimen_sites: list[dict[str, Any]] | None = None,
                       ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """The findings of a pathology report with its diagnosis and its items; the statements it adds (the clinical
    information, the material); and what stays with the report (the diagnosis, the items placed nowhere).

    ``rows``: the findings of the report (``Finding.as_row``, with ``start``); ``specimen_sites``: the sites the
    linker read as the examined structure ("biopsy of the sigmoid colon"), for a diagnosis that names none."""
    sections = {k: list(v) for k, v in (pathology.get("sections") or {}).items()}
    span = pathology.get("diagnosis_span")
    report: dict[str, Any] = {"diagnosis": pathology.get("diagnosis", ""), "unplaced": []}
    kept, added = [], []
    for row in rows:
        where = _section(sections, int(row.get("start", -1)))
        if where in _NOT_A_FINDING:
            added.append({"mention": row["mention"], "structure": row["structure"], "clause": row["clause"],
                          "statement": _NOT_A_FINDING[where], "section": where})
            continue
        kept.append({**row, "pathology_section": where})
    in_diagnosis = [r for r in kept if span and span[0] <= int(r.get("start", -1)) < span[1]]
    specimen = None
    if span and not in_diagnosis and specimen_sites:
        # the diagnosis names no structure: it is of the material's, when the material names exactly one
        named = {s["cid"] for s in specimen_sites}
        if len(named) == 1:
            specimen = {"structure": next(iter(named)), "diagnosis": report["diagnosis"], "items": []}
            report["diagnosis_of_the_specimen"] = specimen
    by_id = {id(r): r for r in in_diagnosis}
    pathology_of: dict[int, list[dict[str, Any]]] = {id(r): [] for r in in_diagnosis}
    structures = {r["structure"] for r in in_diagnosis}
    for i in pathology.get("items") or []:
        if span and span[0] <= i["start"] < span[1] and in_diagnosis:
            lo, hi = _line(text, i["start"])
            before = [r for r in in_diagnosis if lo <= int(r["start"]) <= i["start"]]
            holder = before[-1] if before else None
            if holder is None and len({r["structure"] for r in in_diagnosis if lo <= int(r["start"]) < hi}) == 1:
                holder = next(r for r in in_diagnosis if lo <= int(r["start"]) < hi)
            if holder is not None:
                pathology_of[id(holder)].append(_item(i))
                continue
            report["unplaced"].append({**_item(i), "why": "no_structure_of_the_diagnosis_before_it_in_its_line"})
        elif len(structures) == 1:
            for r in in_diagnosis:          # every finding of the one structure the diagnosis names
                pathology_of[id(r)].append(_item(i))
        elif specimen is not None:
            specimen["items"].append(_item(i))
        else:
            why = ("the_diagnosis_names_no_structure" if not structures
                   else "the_diagnosis_names_more_than_one_structure")
            report["unplaced"].append({**_item(i), "why": why})
    out = []
    for r in kept:
        if id(r) in by_id:
            lo = max(span[0], text.rfind("\n", 0, int(r["start"])) + 1)
            r = {**r, "pathology": {"diagnosis": text[lo:_line(text, int(r["start"]))[1]].strip(),
                                    "items": pathology_of[id(r)]}}
        out.append(r)
    return out, statements + added, report


__all__ = ["INDICATION", "SPECIMEN", "give_the_diagnosis"]
