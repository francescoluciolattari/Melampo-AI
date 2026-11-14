"""The graph of a case in FalkorDB (14 November 2026), next to the sites of the reports (``report_graph``).

Nodes, each with its ``case`` and its words:
``(:Case)``, ``(:Document)``, ``(:Finding)``, ``(:Measurement)``, ``(:Imprint)``, ``(:Medication)``,
``(:Disease {branch})``, ``(:Manifestation)``.

Edges: ``(Case)-[:HAS]->(Document)``, ``(Document)-[:STATES]->(Finding)``, ``(Document)-[:MEASURES]->(Measurement)``,
``(Document)-[:IMAGES]->(Imprint)``, ``(Case)-[:TAKES]->(Medication)``, ``(Case)-[:BRANCH {name}]->(Disease)``,
``(Disease)-[:MANIFESTS]->(Manifestation)``, ``(Manifestation)-[:STATE {state, basis, case}]->(fact)`` for each fact
a state rests on (a state with no fact, such as "not assessed", is kept on the MANIFESTS edge), and
``(Finding)-[:AT]->(:Structure {cid})``, the same Structure nodes the SITE edges of ``report_graph`` use.

Writing a case replaces what the graph held for it, so the graph always says what the case's graph says; the full
record stays in ``CaseGraph.to_dict`` (a JSON document), and ``read_case`` gives it back from the graph.
"""

from __future__ import annotations

import json
from typing import Any

from ..clinical import case_graph as cg


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def load_case(graph, case: cg.CaseGraph) -> dict[str, int]:
    """Write the case (replacing any earlier version of it); returns how many nodes of each kind."""
    cid = case.case_id
    graph.query("MATCH (n {case: $case}) DETACH DELETE n", params={"case": cid})
    graph.query("MERGE (c:Case {id: $case}) SET c.case = $case, c.schema = $schema, c.snapshot = $snapshot",
                params={"case": cid, "schema": case.schema, "snapshot": _json(case.snapshot.as_dict())})

    def rows(items, label, link, parent_key, parent_label, record):
        if not items:
            return
        graph.query(
            f"UNWIND $rows AS row MATCH (p:{parent_label} {{case: $case, id: row.parent}}) "
            f"CREATE (p)-[:{link}]->(n:{label} {{case: $case, id: row.id, words: row.words, record: row.record}})",
            params={"case": cid, "rows": [{"id": i, "parent": p, "words": w, "record": _json(r)}
                                          for i, p, w, r in (record(x) for x in items)]})

    graph.query("UNWIND $rows AS row MATCH (c:Case {id: $case}) "
                "CREATE (c)-[:HAS]->(:Document {case: $case, id: row.id, words: row.title, record: row.record})",
                params={"case": cid, "rows": [{"id": d.document_id, "title": d.title, "record": _json(d.as_dict())}
                                              for d in case.documents]})
    rows(case.findings, "Finding", "STATES", "document", "Document",
         lambda f: (f.finding_id, f.document, f.words, f.as_dict()))
    rows(case.measurements, "Measurement", "MEASURES", "document", "Document",
         lambda m: (m.measurement_id, m.document, m.provenance.words, m.as_dict()))
    rows(case.imprints, "Imprint", "IMAGES", "document", "Document",
         lambda i: (i.imprint_id, i.document, i.digest, i.as_dict()))
    if case.medications:
        graph.query("UNWIND $rows AS row MATCH (c:Case {id: $case}) "
                    "CREATE (c)-[:TAKES]->(:Medication {case: $case, id: row.id, words: row.words, record: row.record})",
                    params={"case": cid, "rows": [{"id": m.medication_id, "words": m.words, "record": _json(m.as_dict())}
                                                  for m in case.medications]})
    sites = [{"id": f.finding_id, "cid": f.site} for f in case.findings if f.site]
    if sites:
        graph.query("UNWIND $rows AS row MATCH (f:Finding {case: $case, id: row.id}) "
                    "MERGE (s:Structure {cid: row.cid}) CREATE (f)-[:AT]->(s)", params={"case": cid, "rows": sites})
    for d in case.diseases:
        record = d.as_dict()
        record.pop("manifestations"), record.pop("states")
        graph.query("MATCH (c:Case {id: $case}) CREATE (c)-[:BRANCH {name: $branch}]->"
                    "(:Disease {case: $case, id: $id, branch: $branch, words: $words, record: $record})",
                    params={"case": cid, "id": d.disease_id, "branch": d.branch, "words": d.words,
                            "record": _json(record)})
        states = {s.manifestation_id: s for s in d.states}
        if d.manifestations:
            graph.query(
                "UNWIND $rows AS row MATCH (d:Disease {case: $case, id: $disease}) "
                "CREATE (d)-[:MANIFESTS {state: row.state, basis: row.basis, state_record: row.state_record}]->"
                "(:Manifestation {case: $case, id: row.id, disease: $disease, words: row.words, record: row.record})",
                params={"case": cid, "disease": d.disease_id, "rows": [
                    {"id": f"{d.disease_id}/{m.manifestation_id}", "words": m.words, "record": _json(m.as_dict()),
                     "state": states[m.manifestation_id].state if m.manifestation_id in states else "",
                     "basis": states[m.manifestation_id].basis if m.manifestation_id in states else "",
                     "state_record": _json(states[m.manifestation_id].as_dict())
                     if m.manifestation_id in states else ""} for m in d.manifestations]})
        evidence = [{"m": f"{d.disease_id}/{s.manifestation_id}", "fact": e, "state": s.state, "basis": s.basis}
                    for s in d.states for e in s.evidence]
        if evidence:
            graph.query("UNWIND $rows AS row MATCH (m:Manifestation {case: $case, id: row.m}) "
                        "MATCH (f {case: $case, id: row.fact}) "
                        "CREATE (m)-[:STATE {case: $case, state: row.state, basis: row.basis}]->(f)",
                        params={"case": cid, "rows": evidence})
    return {"documents": len(case.documents), "findings": len(case.findings),
            "measurements": len(case.measurements), "imprints": len(case.imprints),
            "medications": len(case.medications), "diseases": len(case.diseases),
            "manifestations": sum(len(d.manifestations) for d in case.diseases)}


def read_case(graph, case_id: str) -> cg.CaseGraph:
    """The case as the graph holds it."""
    def records(label: str) -> list[dict[str, Any]]:
        got = graph.query(f"MATCH (n:{label} {{case: $case}}) RETURN n.record ORDER BY n.id",
                          params={"case": case_id})
        return [json.loads(r[0]) for r in got.result_set]

    head = graph.query("MATCH (c:Case {id: $case}) RETURN c.schema, c.snapshot", params={"case": case_id})
    if not head.result_set:
        raise cg.CaseGraphError(f"case {case_id!r} is not in the graph")
    schema, snapshot = head.result_set[0]
    diseases = []
    for record in records("Disease"):
        got = graph.query("MATCH (d:Disease {case: $case, id: $id})-[e:MANIFESTS]->(m:Manifestation) "
                          "RETURN m.record, e.state_record ORDER BY m.id",
                          params={"case": case_id, "id": record["disease_id"]})
        record["manifestations"] = [json.loads(m) for m, _ in got.result_set]
        record["states"] = [json.loads(s) for _, s in got.result_set if s]
        diseases.append(record)
    return cg.CaseGraph.from_dict({
        "case_id": case_id, "schema": schema, "snapshot": json.loads(snapshot),
        "documents": records("Document"), "findings": records("Finding"), "measurements": records("Measurement"),
        "imprints": records("Imprint"), "medications": records("Medication"), "diseases": diseases})


__all__ = ["load_case", "read_case"]
