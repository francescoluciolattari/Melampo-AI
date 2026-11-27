"""The graph of a case from what Melampo already reads (14 November 2026): step 2 of "Il grafo del caso".

Nothing here reads a document again: each function takes what an existing module already produced and puts it in
its branch, with the words and their place.

- ``add_cards``: the cards of the one procedure of reading (``melampo.reading``, 25 November 2026: the same for
  the documents of the case and for the publications) -> one finding per manifestation of the patient.
- ``add_report``: the sites of a report (the reader before the one procedure, ``MELAMPO_REPORT_READER=anatomy``) (``data.report_reading.ReportReaders.read_sites``: rows of
  ``memory.report_graph.site_row``) -> one finding per accepted site, with its qualifiers (polarity, certainty,
  temporality) and the concept the readers chose, when there is one; the exam's areas from its title
  (``memory.exam_area``).
- ``add_laboratory``: the rows of ``data.lab_results.extract_lab_results`` -> one measurement per row, its words
  the printed line.
- ``add_measures``: the measures of the reports that the readers compared with a verified row of the reference
  library (``memory.reference_values``; the reading's ``statements`` of kind ``measure``) -> one measurement each,
  its words the report's, the row's range beside it and the comparison computed again from the number, never
  taken from a model (20 November 2026). The row's deviation (``above_hpo``: "Splenomegaly") lets the diagnostic
  path count the manifestation present, or absent when the measure is within the range.
- ``add_imprint``: an image's imprint (``memory.visual_imprint``) -> one imprint node under its study.
- ``add_medications``: the drugs and supplements as the payload lists them; a drug without the words it was
  written with is refused, never invented.
- ``add_declared_diseases``: diseases named in the documents or by the physician, present (established, prior)
  or hypothesized. The candidate branch is never written here: it starts empty and the diagnostic path fills it.

``build_case_graph`` does all of it for one payload and returns the graph with the problems ``validate`` found.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

from . import case_graph as cg

_POLARITY_NEGATED = {"negated", "negative", "absent", "neg"}
_HYPOTHESIZED = {"hypothesized", "suspected", "possible", "probable", "ipotizzata", "sospetta", "rule_out"}
_PRESENT = {"established", "present", "prior", "confirmed", "known", "pregressa", "accertata", "chronic"}


def _digest(*parts: Any) -> str:
    return hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:12]


def _concept(value: Any, system: str = "") -> cg.Concept | None:
    if not value:
        return None
    if isinstance(value, cg.Concept):
        return value
    if isinstance(value, str):
        return cg.Concept(system=value.split(":", 1)[0] if ":" in value else system, code=value)
    code = value.get("code") or value.get("id") or value.get("curie") or ""
    if not code:
        return None
    return cg.Concept(system=value.get("system") or (code.split(":", 1)[0] if ":" in code else system),
                      code=code, label=value.get("label") or value.get("name") or "")


def add_document(graph: cg.CaseGraph, document_id: str, kind: str, *, title: str = "", date: str = "",
                 modality: str = "", areas: Iterable[str] | None = None, exam_areas: Any = None) -> cg.Document:
    existing = graph.document(document_id)
    if existing is not None:
        return existing
    found = list(areas or [])
    if not found and exam_areas is not None and title:
        found = sorted(exam_areas.of(title))
    doc = cg.Document(document_id=document_id, kind=kind, title=title, date=date, areas=found, modality=modality)
    graph.documents.append(doc)
    return doc


def _first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")[:200]


def add_report(graph: cg.CaseGraph, document_id: str, text: str, reading: dict[str, Any], *, title: str = "",
               date: str = "", modality: str = "", exam_areas: Any = None,
               kind: str = "imaging_report") -> list[cg.Finding]:
    """One finding per accepted site of the reading. The words are the report's own, cut at the offsets the
    reading gives: what the statement says of the site when the readers copied it (``piece``, found in the
    sentence), the mention otherwise."""
    add_document(graph, document_id, kind, title=title or _first_line(text), date=date, modality=modality,
                 exam_areas=exam_areas)
    out = []
    for row in reading.get("sites", []):
        start, end = int(row["start"]), int(row["end"])
        words, w_start, w_end = text[start:end], start, end
        piece = str(row.get("piece") or "").strip()
        if piece:
            at = text.rfind(piece, max(0, start - 400), min(len(text), end + 400))
            if at >= 0:
                words, w_start, w_end = piece, at, at + len(piece)
        polarity = cg.NEGATED if str(row.get("polarity", "")).lower() in _POLARITY_NEGATED else cg.AFFIRMED
        finding = cg.Finding(
            finding_id=f"{document_id}#f{_digest(document_id, w_start, w_end, row.get('cid'))}",
            document=document_id, words=words,
            provenance=cg.Provenance(document=document_id, words=words, start=w_start, end=w_end),
            site=str(row.get("cid", "")), site_words=str(row.get("mention", "")), polarity=polarity,
            certainty=str(row.get("certainty") or ""), temporality=str(row.get("temporality") or ""),
            concept=_concept(row.get("concept"), "HP"), date=date,
            details={k: row[k] for k in ("degree", "character", "location", "relation", "part", "applies_to")
                     if row.get(k)})
        if graph.facts().get(finding.finding_id) is None:
            graph.findings.append(finding)
            out.append(finding)
    return out


def add_cards(graph: cg.CaseGraph, document_id: str, text: str, cards: Iterable[dict[str, Any]], *,
              title: str = "", date: str = "", modality: str = "", exam_areas: Any = None,
              kind: str = "note") -> tuple[list[cg.Finding], list[str]]:
    """One finding per card of the one procedure of reading (``melampo.reading``, 25 November 2026) that is a
    manifestation of the patient: its words the sentence's, its state the interpreters' agreed one, its concept
    the one given afterwards (HPO, or a concept of the thesaurus), its site the lexicon's class when there is one.
    A card the interpreters did not agree on (a field read differently, a card read by one alone) is a finding
    stated with uncertainty: principle of doubt, kept and never decided here. A family member's finding is not
    the patient's: it is left out, and said. A card of a pathology report's clinical information or material is
    not a finding; the diagnosis's items (TNM, grade, margins), the row of the library a measure was compared with
    and the image side stay in the finding's details (``reading.case_sides``, 27 November 2026); the measure
    itself becomes a measurement (``add_measures``)."""
    add_document(graph, document_id, kind, title=title or _first_line(text), date=date, modality=modality,
                 exam_areas=exam_areas)
    out, notes = [], []
    for card in cards:
        if card.get("category") != "manifestation" or card.get("specific") is False:
            continue
        if card.get("subject") == "family_member":
            notes.append(f"{document_id}: {card.get('name')!r} is a family member's finding, not the patient's")
            continue
        if card.get("pathology_statement"):
            # the clinical information or the material of a pathology report: the question asked, or where the
            # sample comes from, not a current finding (27 November 2026)
            continue
        quote = str(card.get("quote") or "")
        start = text.find(quote) if quote else -1
        words = quote if start >= 0 else str(card.get("words") or card.get("name") or "")
        state = card.get("state")
        doubts = list(card.get("doubts") or [])
        concept = card.get("concept") or {}
        code = concept.get("code") or concept.get("concept") or ""
        found = None
        if code:
            found = cg.Concept("HPO" if code.startswith("HP:") else "MELAMPO", code, card.get("name", ""))
        details = {k: card[k] for k in ("name", "what", "interpretation", "value", "threshold", "side", "time",
                                        "observed_by", "paired_by", "reference", "pathology", "pathology_section",
                                        "image") if card.get(k)}
        if doubts:
            details["doubts"] = doubts
        finding = cg.Finding(
            finding_id=f"{document_id}#c{_digest(document_id, start, card.get('name'), card.get('state'))}",
            document=document_id, words=words,
            provenance=cg.Provenance(document=document_id, words=words, start=start if start >= 0 else None,
                                     end=start + len(quote) if start >= 0 else None),
            site=str(card.get("site_code") or ""), site_words=str(card.get("site") or ""),
            polarity=cg.NEGATED if state == "absent" else cg.AFFIRMED,
            # a field one interpreter did not give and the other did is kept in the details, not a doubt of
            # the finding (26 November 2026); every other doubt is
            certainty="uncertain" if state == "uncertain" or [d for d in doubts if not d.endswith("_given_by_one")]
            else "certain",
            temporality=str(card.get("temporality") or ""), concept=found, date=date, details=details)
        if graph.facts().get(finding.finding_id) is None:
            graph.findings.append(finding)
            out.append(finding)
    return out, notes


def add_laboratory(graph: cg.CaseGraph, document_id: str, text: str, extraction: Any, *, title: str = "",
                   date: str = "", codes: dict[str, Any] | None = None) -> list[cg.Measurement]:
    """One measurement per laboratory row; its words are the printed line (``line_number`` counts from 1)."""
    add_document(graph, document_id, "laboratory", title=title, date=date)
    lines = text.splitlines()
    out = []
    for row in getattr(extraction, "results", ()):
        number = int(row.line_number)
        line = lines[number - 1].strip() if 0 < number <= len(lines) else ""
        words = line or " ".join(x for x in (row.analyte, row.value_text, row.unit or "") if x)
        code = _concept((codes or {}).get(row.analyte), "LOINC")
        m = cg.Measurement(
            measurement_id=f"{document_id}#m{number}", document=document_id, analyte=row.analyte,
            value=row.value if row.value is not None else row.value_text,
            provenance=cg.Provenance(document=document_id, words=words, line=number),
            unit=row.unit or "", reference=row.reference.as_dict() if row.reference else None,
            interpretation=row.range_status, code=code, date=date)
        graph.measurements.append(m)
        out.append(m)
    return out


def add_measures(graph: cg.CaseGraph, statements: Iterable[dict[str, Any]], texts: dict[str, str],
                 references: Any = None) -> tuple[list[cg.Measurement], list[str]]:
    """One measurement per measure a reader compared with a row of the reference library. Only verified rows; the
    number is read again from the report's words and compared with the row here. Returns the measurements and the
    notes of what was left out (a row not verified, no number in the row's unit)."""
    from ..memory import reference_values as rv

    measures = [s for s in statements if s.get("statement") == "measure" and s.get("reference")]
    if not measures:
        return [], []
    references = references or rv.ReferenceValues.load()
    added, notes = [], []
    for st in measures:
        row = references.by_id.get(st["reference"])
        words, document = str(st.get("piece") or "").strip(), str(st.get("report") or "")
        if row is None or row.status != rv.VERIFIED:
            notes.append(f"measure {words!r}: row {st['reference']!r} is not a verified row of the library")
            continue
        found = [m for m in rv.measures_in(words) if m["unit"] == row.unit]
        if not words or not found or document not in texts:
            notes.append(f"measure {words!r}: no number in {row.unit} or no document")
            continue
        text = texts[document]
        start = text.find(words)
        clause = str(st.get("clause") or "")
        if start < 0 and clause:
            at = text.find(clause)
            start = at + clause.find(words) if at >= 0 and words in clause else -1
        add_document(graph, document, "imaging_report")
        value = found[0]["value"]
        m = cg.Measurement(
            measurement_id=f"{document}#r{_digest(document, words, row.id)}", document=document,
            analyte=f"{st.get('structure', '')} {row.aspect}".strip(), value=value,
            provenance=cg.Provenance(document=document, words=words, start=start if start >= 0 else None,
                                     end=start + len(words) if start >= 0 else None),
            unit=row.unit,
            reference={"row": row.id, "low": row.low, "high": row.high, "above_means": row.above_means,
                       "below_means": row.below_means, "above_hpo": row.above_hpo, "below_hpo": row.below_hpo,
                       "source": {k: row.source.get(k, "") for k in ("title", "publisher", "url", "section", "ref")}},
            interpretation=rv.compare(value, row))
        if graph.facts().get(m.measurement_id) is None:
            graph.measurements.append(m)
            added.append(m)
    return added, notes


def add_imprint(graph: cg.CaseGraph, document_id: str, imprint: dict[str, Any], *, study: str = "",
                date: str = "", findings: Iterable[str] = ()) -> cg.Imprint:
    add_document(graph, document_id, "imaging_study", date=date)
    digest = str(imprint.get("vector_hash") or imprint.get("digest") or imprint.get("imprint_id") or
                 _digest(sorted(imprint.items())))
    node = cg.Imprint(imprint_id=f"{document_id}#i{digest[:12]}", document=document_id, study=study,
                      digest=digest, findings=list(findings), date=date)
    graph.imprints.append(node)
    return node


def add_medications(graph: cg.CaseGraph, items: Iterable[dict[str, Any]]) -> tuple[list[cg.Medication], list[str]]:
    """The drugs and supplements as listed; an item without its words is refused with its reason."""
    added, refused = [], []
    for index, item in enumerate(items):
        words = str(item.get("words") or item.get("text") or "").strip()
        if not words:
            refused.append(f"medication {index}: no words as written")
            continue
        document = str(item.get("document") or "payload")
        med = cg.Medication(
            medication_id=f"drug:{_digest(document, words, item.get('start', ''))}", words=words,
            provenance=cg.Provenance(document=document, words=words, start=item.get("offset_start"),
                                     end=item.get("offset_end")),
            name=str(item.get("name") or ""), kind=str(item.get("kind") or "drug"), dose=str(item.get("dose") or ""),
            unit=str(item.get("unit") or ""), frequency=str(item.get("frequency") or ""),
            route=str(item.get("route") or ""), start=str(item.get("start") or ""), end=str(item.get("end") or ""),
            codes=[c for c in (_concept(x) for x in item.get("codes", [])) if c])
        graph.medications.append(med)
        added.append(med)
    return added, refused


def add_declared_diseases(graph: cg.CaseGraph, items: Iterable[dict[str, Any]], *,
                          origin: str = cg.NAMED_IN_DOCUMENTS) -> tuple[list[cg.Disease], list[str]]:
    """Diseases named in the documents or by the physician: established or prior -> present, suspected ->
    hypothesized. A status that says neither is refused (never guessed)."""
    added, refused = [], []
    for index, item in enumerate(items):
        words = str(item.get("words") or "").strip()
        status = str(item.get("status") or "").strip().lower()
        if not words:
            refused.append(f"disease {index}: no words as written")
            continue
        if status in _PRESENT:
            branch = cg.PRESENT_DISEASES
        elif status in _HYPOTHESIZED:
            branch = cg.HYPOTHESIZED_DISEASES
        else:
            refused.append(f"disease {index} ({words}): status {status!r} is neither present nor hypothesized")
            continue
        document = str(item.get("document") or "")
        provenance = cg.Provenance(document=document, words=words, start=item.get("start"), end=item.get("end")) \
            if document else None
        disease = cg.Disease(disease_id=f"dis:{_digest(words.lower(), branch)}", branch=branch, words=words,
                             origin=origin, provenance=provenance, concept=_concept(item.get("concept"), "MONDO"),
                             status_words=str(item.get("status_words") or ""), date=str(item.get("date") or ""))
        if graph.disease(disease.disease_id) is None:
            graph.add_disease(disease)
            added.append(disease)
    return added, refused


def build_case_graph(case_id: str, *, reports: Iterable[dict[str, Any]] = (),
                     laboratory: Iterable[dict[str, Any]] = (), imprints: Iterable[dict[str, Any]] = (),
                     medications: Iterable[dict[str, Any]] = (), diseases: Iterable[dict[str, Any]] = (),
                     physician_diseases: Iterable[dict[str, Any]] = (), sources: dict[str, str] | None = None,
                     taken: str = "", exam_areas: Any = None) -> tuple[cg.CaseGraph, list[str]]:
    """The graph of one case. Each report: ``{"document", "text", "reading", "title", "date", "modality"}``;
    each laboratory document: ``{"document", "text", "extraction", "date", "codes"}``; each imprint:
    ``{"document", "imprint", "study", "date"}``. Returns the graph and what was refused or breaks its rules."""
    graph = cg.CaseGraph(case_id=case_id, snapshot=cg.KnowledgeSnapshot(sources=dict(sources or {}), taken=taken))
    notes: list[str] = []
    for r in reports:
        add_report(graph, r["document"], r["text"], r["reading"], title=r.get("title", ""), date=r.get("date", ""),
                   modality=r.get("modality", ""), exam_areas=exam_areas)
    for lab in laboratory:
        add_laboratory(graph, lab["document"], lab["text"], lab["extraction"], title=lab.get("title", ""),
                       date=lab.get("date", ""), codes=lab.get("codes"))
    for i in imprints:
        add_imprint(graph, i["document"], i["imprint"], study=i.get("study", ""), date=i.get("date", ""))
    notes += add_medications(graph, medications)[1]
    notes += add_declared_diseases(graph, diseases)[1]
    notes += add_declared_diseases(graph, physician_diseases, origin=cg.PHYSICIAN)[1]
    notes += graph.validate()
    return graph, notes


def from_ingestion(case_id: str, note: str, bundle: Any, reading: dict[str, Any], payload: dict[str, Any], *,
                   exam_areas: Any = None) -> tuple[cg.CaseGraph, list[str]]:
    """The graph of a case from what ingestion already holds: the reading of the note and of the attachments
    (``ReportReaders.read``), the laboratory rows of each attachment with their LOINC code when exactly one term
    fits, and the drugs and diseases the payload declares (``medications``; ``diseases`` named in the documents,
    ``physician_diseases`` by the physician)."""
    texts = {"note": note}
    attachments = list(getattr(bundle, "attachments", ()) or ())
    for a in attachments:
        texts[f"attachment-{a.index}"] = a.text
    if exam_areas is None:
        try:
            from ..memory.exam_area import ExamAreas

            exam_areas = ExamAreas.load()
        except (OSError, ValueError):
            exam_areas = None
    reports = []
    for entry in (reading or {}).get("texts", []):
        source = entry.get("report", "")
        if not entry.get("read") or source not in texts:
            continue
        kinds = entry.get("kinds") or ()
        reports.append({"document": source, "text": texts[source], "reading": entry,
                        "kind": "imaging_report" if "imaging" in kinds else (
                            "pathology" if "pathology" in kinds else "note")})
    laboratory = []
    for a in attachments:
        lab = getattr(a, "lab", None)
        if lab is not None and getattr(lab, "results", ()):
            codes = {row.get("analyte"): {"system": "LOINC", "code": f"LOINC:{row['code']}"}
                     for row in getattr(a, "loinc", ()) if row.get("status") == "mapped" and row.get("code")}
            laboratory.append({"document": f"attachment-{a.index}", "text": a.text, "extraction": lab,
                               "codes": codes})
    graph, notes = build_case_graph(
        case_id, reports=[], laboratory=laboratory, medications=payload.get("medications") or [],
        diseases=payload.get("diseases") or [], physician_diseases=payload.get("physician_diseases") or [],
        exam_areas=exam_areas)
    for r in reports:
        if r["reading"].get("procedure") == "one":
            notes += add_cards(graph, r["document"], r["text"], r["reading"].get("cards") or [], kind=r["kind"],
                               exam_areas=exam_areas)[1]
        else:
            add_report(graph, r["document"], r["text"], r["reading"], kind=r["kind"], exam_areas=exam_areas)
    try:
        notes += add_measures(graph, (reading or {}).get("statements") or [], texts)[1]
    except (OSError, ValueError, ImportError) as error:      # the library could not be read: the case goes on
        notes.append(f"measures not added: {type(error).__name__}")
    notes = [n for n in notes if "is not in the case" not in n] + graph.validate()
    return graph, sorted(set(notes))


__all__ = [
    "add_cards",
    "add_declared_diseases",
    "add_measures",
    "add_document",
    "add_imprint",
    "add_laboratory",
    "add_medications",
    "add_report",
    "build_case_graph",
    "from_ingestion",
]
