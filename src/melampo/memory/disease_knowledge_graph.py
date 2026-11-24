"""The disease knowledge as a graph in FalkorDB, version by version (22 November 2026).

The versions are kept on the ``knowledge`` branch (``scripts/knowledge_publish.py``): that is the record that
survives. FalkorDB is where the diagnosis reads them: each version is loaded once, as

    (:KVersion {id, date, current})<-[:IN]-(:KDisease {key, words, record})
    (:KDisease)-[:SHOWS {frequency, frequency_basis, sensitivity, specificity, lr_positive, lr_negative,
                         typicality, evidence, record}]->(:KManifestation {key, words})
    (:KDisease)-[:DETECTED_BY {sensitivity, specificity, lr_positive, lr_negative, setting}]->(:KTest {name})

so that "which diseases show these manifestations" is one query, and every case records the version it was read
with. Loading is idempotent: a version already in the graph is not loaded again; a graph that lost its data (the
free FalkorDB Cloud instance keeps none across a restart) is loaded again from the branch. Only public knowledge
goes there, never a patient's case: the free instance has no TLS.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from ..clinical import case_graph as cg

GRAPH = "melampo_knowledge"


def _key(m: cg.Manifestation) -> str:
    from .profile_library import norm

    return m.concept.curie if m.concept is not None and m.concept.code else norm(m.words)


def _rows(result) -> list[list[Any]]:
    return list(getattr(result, "result_set", None) or [])


def has_version(graph, version: str) -> bool:
    return bool(_rows(graph.query("MATCH (v:KVersion {id: $v}) RETURN v.id", params={"v": version})))


def load_version(graph, version: str, profiles: Iterable[dict], *, date: str = "", names: dict | None = None,
                 tests: dict | None = None, thesaurus: Iterable[dict] = ()) -> dict[str, int]:
    """Load one version and make it the current one; nothing is loaded twice. ``thesaurus``: the rows of
    ``thesaurus.jsonl`` (``melampo.reading.thesaurus``), as ``KConcept`` and ``KLabel`` nodes of the version."""
    profiles = list(profiles)
    if not has_version(graph, version):
        graph.query("CREATE (:KVersion {id: $v, date: $d, names: $n, current: false})",
                    params={"v": version, "d": date, "n": json.dumps(names or {}, ensure_ascii=False)})
        diseases, shows = [], []
        for p in profiles:
            d = cg.Disease.from_dict(p)
            diseases.append({"key": d.disease_id, "words": d.words, "record": json.dumps(p, ensure_ascii=False)})
            for m in d.manifestations:
                acc = m.accuracy or {}
                shows.append({"d": d.disease_id, "key": _key(m), "words": m.words,
                              "frequency": m.frequency, "basis": m.frequency_basis or "",
                              "sens": acc.get("sensitivity"), "spec": acc.get("specificity"),
                              "lrp": acc.get("lr_positive"), "lrn": acc.get("lr_negative"),
                              "typicality": m.typicality or "", "evidence": m.evidence or "",
                              "record": json.dumps(m.as_dict(), ensure_ascii=False)})
        if diseases:
            graph.query("UNWIND $rows AS r MATCH (v:KVersion {id: $v}) "
                        "CREATE (:KDisease {key: r.key, version: $v, words: r.words, record: r.record})-[:IN]->(v)",
                        params={"v": version, "rows": diseases})
        for start in range(0, len(shows), 500):
            graph.query("UNWIND $rows AS r MATCH (d:KDisease {key: r.d, version: $v}) "
                        "MERGE (m:KManifestation {key: r.key}) ON CREATE SET m.words = r.words "
                        "CREATE (d)-[:SHOWS {version: $v, words: r.words, frequency: r.frequency, "
                        "frequency_basis: r.basis, sensitivity: r.sens, specificity: r.spec, lr_positive: r.lrp, "
                        "lr_negative: r.lrn, typicality: r.typicality, evidence: r.evidence, record: r.record}]->(m)",
                        params={"v": version, "rows": shows[start:start + 500]})
        detected = []
        for disease_words, rows in (tests or {}).items():
            key = next((p["disease_id"] for p in profiles if p.get("words") == disease_words), disease_words)
            for t in rows:
                acc = t.get("accuracy") or {}
                detected.append({"d": key, "name": t.get("name", ""), "sens": acc.get("sensitivity"),
                                 "spec": acc.get("specificity"), "lrp": acc.get("lr_positive"),
                                 "lrn": acc.get("lr_negative"), "setting": acc.get("setting", "")})
        if detected:
            graph.query("UNWIND $rows AS r MATCH (d:KDisease {key: r.d, version: $v}) "
                        "MERGE (t:KTest {name: r.name}) "
                        "CREATE (d)-[:DETECTED_BY {version: $v, sensitivity: r.sens, specificity: r.spec, "
                        "lr_positive: r.lrp, lr_negative: r.lrn, setting: r.setting}]->(t)",
                        params={"v": version, "rows": detected})
        concepts = [{"id": c["id"], "name": c.get("name", ""), "code": c.get("code", ""),
                     "labels": [{"label": x.get("label", ""), "signature": json.dumps(x.get("signature", {}),
                                                                                       sort_keys=True),
                                 "evidence": json.dumps(x, ensure_ascii=False)} for x in c.get("labels") or []]}
                    for c in thesaurus]
        for start in range(0, len(concepts), 200):
            graph.query("UNWIND $rows AS r "
                        "CREATE (c:KConcept {id: r.id, version: $v, name: r.name, code: r.code}) "
                        "WITH c, r UNWIND r.labels AS l "
                        "CREATE (:KLabel {label: l.label, signature: l.signature, evidence: l.evidence, "
                        "version: $v})-[:LABEL_OF]->(c)",
                        params={"v": version, "rows": concepts[start:start + 200]})
    graph.query("MATCH (v:KVersion) SET v.current = (v.id = $v)", params={"v": version})
    return counts(graph, version)


def read_thesaurus(graph, version: str = "") -> list[dict]:
    """The thesaurus of a version (the current one when not given), as the rows of ``thesaurus.jsonl``."""
    version = version or current_version(graph)
    rows = _rows(graph.query("MATCH (c:KConcept {version: $v}) OPTIONAL MATCH (l:KLabel)-[:LABEL_OF]->(c) "
                             "RETURN c.id, c.name, c.code, collect(l.evidence)", params={"v": version}))
    return [{"id": r[0], "name": r[1], "code": r[2], "labels": [json.loads(e) for e in r[3] if e]}
            for r in sorted(rows, key=lambda r: r[0])]


def counts(graph, version: str) -> dict[str, int]:
    q = lambda c: _rows(graph.query(c, params={"v": version}))[0][0]  # noqa: E731
    return {"diseases": q("MATCH (d:KDisease {version: $v}) RETURN count(d)"),
            "manifestations": q("MATCH (:KDisease {version: $v})-[s:SHOWS]->() RETURN count(s)"),
            "tests": q("MATCH (:KDisease {version: $v})-[t:DETECTED_BY]->() RETURN count(t)"),
            "concepts": q("MATCH (c:KConcept {version: $v}) RETURN count(c)")}


def current_version(graph) -> str:
    rows = _rows(graph.query("MATCH (v:KVersion {current: true}) RETURN v.id"))
    return str(rows[0][0]) if rows else ""


def versions(graph) -> list[tuple[str, str, bool]]:
    return [(str(r[0]), str(r[1]), bool(r[2])) for r in
            _rows(graph.query("MATCH (v:KVersion) RETURN v.id, v.date, v.current ORDER BY v.date"))]


def read_profiles(graph, version: str = "") -> tuple[str, list[cg.Disease], dict]:
    """The profiles of a version (the current one when none is named), with the names stored beside them."""
    version = version or current_version(graph)
    if not version:
        return "", [], {}
    rows = _rows(graph.query("MATCH (d:KDisease {version: $v}) RETURN d.record ORDER BY d.key",
                             params={"v": version}))
    names = _rows(graph.query("MATCH (v:KVersion {id: $v}) RETURN v.names", params={"v": version}))
    return version, [cg.Disease.from_dict(json.loads(r[0])) for r in rows], json.loads(names[0][0] or "{}") if names else {}


def diseases_with(graph, keys: Iterable[str], version: str = "") -> list[tuple[str, int]]:
    """The diseases of a version that show any of these manifestations, with how many: one query."""
    version = version or current_version(graph)
    rows = _rows(graph.query("MATCH (d:KDisease {version: $v})-[:SHOWS]->(m:KManifestation) WHERE m.key IN $keys "
                             "RETURN d.key, count(m) AS n ORDER BY n DESC, d.key",
                             params={"v": version, "keys": list(keys)}))
    return [(str(r[0]), int(r[1])) for r in rows]


__all__ = ["GRAPH", "counts", "current_version", "diseases_with", "has_version", "load_version", "read_profiles",
           "read_thesaurus",
           "versions"]
