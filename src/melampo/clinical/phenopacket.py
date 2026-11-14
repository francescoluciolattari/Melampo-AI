"""The facts of a case as a GA4GH Phenopacket v2, and back (14 November 2026).

Phenopacket v2 (Jacobsen et al., Nat Biotechnol 2022; https://phenopacket-schema.readthedocs.io/en/2.0.0/) is the
exchange format chosen in "Melampo — Il grafo del caso": it already has excluded phenotypic features, measurements,
medical actions, diseases and interpretations in progress. What it carries here:

- a finding -> ``phenotypicFeatures``: the concept (or ``melampo:finding`` with the words as label), the words in
  ``description``, ``excluded`` for a negated finding, the document and the offsets in the evidence reference;
- a value -> ``measurements``: the analyte (LOINC when coded), the value with its unit and printed range;
- a drug -> ``medicalActions.treatment``: the agent, the dose interval with its dates;
- a present disease -> ``diseases``; a hypothesized or candidate disease -> ``interpretations`` IN_PROGRESS.

What Phenopacket has no field for (the place of a value or a drug in its document, the site, the certainty, the
temporality, the branch, the origin) goes in ``metaData.externalReferences`` as ``melampo:<kind>/<id>`` with a JSON
description, so that ``from_phenopacket(to_phenopacket(g))`` gives the same facts back. Knowledge (manifestations)
and inference (their states) are not facts of the patient and are not exported.
"""

from __future__ import annotations

import json
from typing import Any

from . import case_graph as cg

PHENOPACKET_SCHEMA = "2.0"
LOCAL = "melampo"
EVIDENCE_TRACEABLE = {"id": "ECO:0000033", "label": "author statement supported by traceable reference"}


def _term(concept: cg.Concept | None, words: str, local: str) -> dict[str, str]:
    if concept is not None:
        return {"id": concept.curie, "label": concept.label or words}
    return {"id": f"{LOCAL}:{local}", "label": words}


def _concept(term: dict[str, Any] | None) -> cg.Concept | None:
    if not term or str(term.get("id", "")).startswith(f"{LOCAL}:"):
        return None
    system, _, _ = str(term["id"]).partition(":")
    return cg.Concept(system=system, code=term["id"], label=term.get("label", ""))


def _ref(kind: str, item_id: str, extra: dict[str, Any]) -> dict[str, str]:
    return {"id": f"{LOCAL}:{kind}/{item_id}", "reference": kind,
            "description": json.dumps(extra, ensure_ascii=False, sort_keys=True)}


def _time(date: str) -> dict[str, str] | None:
    return {"timestamp": date} if date else None


def to_phenopacket(graph: cg.CaseGraph) -> dict[str, Any]:
    refs: list[dict[str, str]] = []
    features = []
    for f in graph.findings:
        feature = {"type": _term(f.concept, f.words, "finding"), "description": f.words,
                   "excluded": f.polarity == cg.NEGATED,
                   "evidence": [{"evidenceCode": EVIDENCE_TRACEABLE,
                                 "reference": {"id": f.provenance.document,
                                               "description": f"{f.provenance.start}-{f.provenance.end}"}}]}
        if f.date:
            feature["onset"] = _time(f.date)
        features.append(feature)
        refs.append(_ref("finding", f.finding_id, {
            "document": f.document, "provenance": f.provenance.as_dict(), "site": f.site, "site_words": f.site_words,
            "certainty": f.certainty, "temporality": f.temporality, "details": f.details}))
    measurements = []
    for m in graph.measurements:
        if isinstance(m.value, (int, float)) and not isinstance(m.value, bool):
            quantity: dict[str, Any] = {"unit": {"id": f"UCUM:{m.unit}" if m.unit else f"{LOCAL}:no_unit",
                                                 "label": m.unit}, "value": m.value}
            ref = m.reference or {}
            if ref.get("low") is not None or ref.get("high") is not None:
                quantity["referenceRange"] = {"unit": quantity["unit"], "low": ref.get("low"),
                                              "high": ref.get("high")}
            value: dict[str, Any] = {"quantity": quantity}
        else:
            value = {"ontologyClass": {"id": f"{LOCAL}:value", "label": str(m.value)}}
        entry = {"assay": _term(m.code, m.analyte, "analyte"), "value": value, "description": m.provenance.words}
        if m.date:
            entry["timeObserved"] = _time(m.date)
        measurements.append(entry)
        refs.append(_ref("measurement", m.measurement_id, {
            "document": m.document, "provenance": m.provenance.as_dict(), "unit": m.unit, "value": m.value,
            "reference": m.reference, "interpretation": m.interpretation}))
    actions = []
    for d in graph.medications:
        interval = {k: v for k, v in (("start", d.start), ("end", d.end)) if v}
        dose = {"scheduleFrequency": {"id": f"{LOCAL}:frequency", "label": d.frequency}} if d.frequency else {}
        if d.dose:
            dose["quantity"] = {"unit": {"id": f"UCUM:{d.unit}" if d.unit else f"{LOCAL}:no_unit", "label": d.unit},
                                "value": d.dose}
        if interval:
            dose["interval"] = interval
        treatment: dict[str, Any] = {"agent": _term(d.codes[0] if d.codes else None, d.name or d.words, "drug")}
        if d.route:
            treatment["routeOfAdministration"] = {"id": f"{LOCAL}:route", "label": d.route}
        if dose:
            treatment["doseIntervals"] = [dose]
        actions.append({"treatment": treatment})
        refs.append(_ref("medication", d.medication_id, {
            "provenance": d.provenance.as_dict(), "words": d.words, "kind": d.kind, "dose": d.dose,
            "codes": [c.as_dict() for c in d.codes]}))
    diseases, interpretations = [], []
    for d in graph.diseases:
        term = _term(d.concept, d.words, "disease")
        refs.append(_ref("disease", d.disease_id, {
            "branch": d.branch, "origin": d.origin, "words": d.words, "status_words": d.status_words,
            "provenance": d.provenance.as_dict() if d.provenance else None, "citations": d.citations}))
        if d.branch == cg.PRESENT_DISEASES:
            entry = {"term": term}
            if d.date:
                entry["onset"] = _time(d.date)
            diseases.append(entry)
        else:
            interpretations.append({"id": d.disease_id, "progressStatus": "IN_PROGRESS",
                                    "diagnosis": {"disease": term}})
    for doc in graph.documents:
        refs.append(_ref("document", doc.document_id, doc.as_dict()))
    packet: dict[str, Any] = {
        "id": graph.case_id, "subject": {"id": graph.case_id},
        "phenotypicFeatures": features, "measurements": measurements, "medicalActions": actions,
        "diseases": diseases, "interpretations": interpretations,
        "metaData": {"createdBy": "Melampo", "phenopacketSchemaVersion": PHENOPACKET_SCHEMA,
                     "resources": [{"id": k, "version": v} for k, v in sorted(graph.snapshot.sources.items())],
                     "externalReferences": refs},
    }
    if graph.snapshot.taken:
        packet["metaData"]["created"] = graph.snapshot.taken
    return packet


def from_phenopacket(packet: dict[str, Any]) -> cg.CaseGraph:
    refs: dict[tuple[str, str], dict[str, Any]] = {}
    order: dict[str, list[str]] = {}
    for r in packet.get("metaData", {}).get("externalReferences", []):
        rid = str(r.get("id", ""))
        if rid.startswith(f"{LOCAL}:") and "/" in rid:
            kind, item = rid[len(LOCAL) + 1:].split("/", 1)
            refs[(kind, item)] = json.loads(r.get("description") or "{}")
            order.setdefault(kind, []).append(item)
    graph = cg.CaseGraph(case_id=packet.get("id") or packet.get("subject", {}).get("id", ""))
    graph.documents = [cg.Document.from_dict(refs[("document", i)]) for i in order.get("document", [])]
    for item, feature in zip(order.get("finding", []), packet.get("phenotypicFeatures", [])):
        extra = refs[("finding", item)]
        graph.findings.append(cg.Finding(
            finding_id=item, document=extra["document"], words=feature.get("description", ""),
            provenance=cg.Provenance.from_dict(extra["provenance"]), site=extra.get("site", ""),
            site_words=extra.get("site_words", ""),
            polarity=cg.NEGATED if feature.get("excluded") else cg.AFFIRMED, certainty=extra.get("certainty", ""),
            temporality=extra.get("temporality", ""), concept=_concept(feature.get("type")),
            date=(feature.get("onset") or {}).get("timestamp", ""), details=extra.get("details") or {}))
    for item, entry in zip(order.get("measurement", []), packet.get("measurements", [])):
        extra = refs[("measurement", item)]
        graph.measurements.append(cg.Measurement(
            measurement_id=item, document=extra["document"], analyte=entry["assay"].get("label", ""),
            value=extra.get("value"), provenance=cg.Provenance.from_dict(extra["provenance"]),
            unit=extra.get("unit", ""), reference=extra.get("reference"),
            interpretation=extra.get("interpretation", ""), code=_concept(entry.get("assay")),
            date=(entry.get("timeObserved") or {}).get("timestamp", "")))
    for item, action in zip(order.get("medication", []), packet.get("medicalActions", [])):
        extra = refs[("medication", item)]
        treatment = action.get("treatment", {})
        dose = (treatment.get("doseIntervals") or [{}])[0]
        graph.medications.append(cg.Medication(
            medication_id=item, words=extra.get("words", ""), provenance=cg.Provenance.from_dict(extra["provenance"]),
            name=treatment.get("agent", {}).get("label", ""), kind=extra.get("kind", "drug"),
            dose=extra.get("dose", ""), unit=(dose.get("quantity") or {}).get("unit", {}).get("label", ""),
            frequency=(dose.get("scheduleFrequency") or {}).get("label", ""),
            route=(treatment.get("routeOfAdministration") or {}).get("label", ""),
            start=(dose.get("interval") or {}).get("start", ""), end=(dose.get("interval") or {}).get("end", ""),
            codes=[cg.Concept.from_dict(c) for c in extra.get("codes", [])]))
    present = iter(packet.get("diseases", []))
    pending = {i.get("id"): i for i in packet.get("interpretations", [])}
    for item in order.get("disease", []):
        extra = refs[("disease", item)]
        if extra["branch"] == cg.PRESENT_DISEASES:
            entry = next(present)
            term, date = entry.get("term"), (entry.get("onset") or {}).get("timestamp", "")
        else:
            term, date = pending[item]["diagnosis"]["disease"], ""
        graph.diseases.append(cg.Disease(
            disease_id=item, branch=extra["branch"], words=extra["words"], origin=extra["origin"],
            provenance=cg.Provenance.from_dict(extra["provenance"]) if extra.get("provenance") else None,
            concept=_concept(term), status_words=extra.get("status_words", ""), date=date,
            citations=extra.get("citations") or []))
    graph.snapshot = cg.KnowledgeSnapshot(
        sources={r["id"]: r.get("version", "") for r in packet.get("metaData", {}).get("resources", [])},
        taken=packet.get("metaData", {}).get("created", ""))
    return graph


__all__ = ["from_phenopacket", "to_phenopacket"]
