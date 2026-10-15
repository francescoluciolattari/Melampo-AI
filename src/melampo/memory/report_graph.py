"""The sites of a report in the graph, with what the report says about each (15 October 2026).

Each accepted link of a report becomes an edge ``(:Report)-[:SITE]->(:Structure)`` that carries the mention as
written, its offsets, the roles of the structure and the qualifiers of the site (``site_qualifiers``): how
precisely it is placed (as named, approximate, range, region, with the words), how marked the finding is and its
radiological character, how certain (ConText). The graph is FalkorDB (embedded or remote,
``falkordb_connection.connect``); the rows are plain dicts, so the same records can be written as JSON.

Nothing here decides a link: an edge is written only for a link the linker accepted, and the qualifiers describe
it. Text is pseudonymised before it reaches this module (``report_id`` is a pseudonymous key).
"""

from __future__ import annotations

from typing import Any

EDGE_FIELDS = ("location", "location_cues", "degree", "degree_cues", "degree_scope", "character", "character_cues",
               "character_scope", "certainty",
               "polarity", "temporality", "certainty_cues", "language")


def site_row(report_id: str, mention: str, start: int, end: int, result) -> dict[str, Any] | None:
    """The row for one link, or None when the link was not accepted."""
    if getattr(result, "status", "") != "accepted" or not getattr(result, "cid", None):
        return None
    q = dict(getattr(result, "qualifiers", {}) or {})
    row: dict[str, Any] = {"report": report_id, "cid": result.cid, "mention": mention, "start": int(start),
                           "end": int(end), "relation": result.relation, "part": result.part,
                           "roles": list(result.roles or ((result.role,) if result.role else ()))}
    # the structure examined ("TC del torace", "CT brain"): what the clause says is about the findings of the
    # examination, not about the structure itself
    row["applies_to"] = "exam_findings" if "procedure_site" in row["roles"] else "site"
    for name in EDGE_FIELDS:
        value = q.get(name)
        row[name] = list(value) if isinstance(value, (list, tuple)) else (value if value is not None else "")
    return row


def load_sites(graph, rows: list[dict[str, Any]]) -> int:
    """Write the rows (``site_row``) as SITE edges; returns how many were written."""
    rows = [r for r in rows if r]
    for at in range(0, len(rows), 500):
        graph.query(
            "UNWIND $rows AS row "
            "MERGE (r:Report {id: row.report}) "
            "MERGE (s:Structure {cid: row.cid}) "
            "CREATE (r)-[:SITE {mention: row.mention, start: row.start, end: row.end, relation: row.relation, "
            "part: row.part, roles: row.roles, applies_to: row.applies_to, location: row.location, location_cues: row.location_cues, "
            "degree: row.degree, degree_cues: row.degree_cues, degree_scope: row.degree_scope, "
            "character: row.character, character_cues: row.character_cues, character_scope: row.character_scope, certainty: row.certainty, polarity: row.polarity, "
            "temporality: row.temporality, certainty_cues: row.certainty_cues, language: row.language}]->(s)",
            params={"rows": rows[at:at + 500]},
        )
    return len(rows)


def sites_of(graph, report_id: str) -> list[dict[str, Any]]:
    """The sites of one report, in the order of the text."""
    got = graph.query(
        "MATCH (r:Report {id: $id})-[e:SITE]->(s:Structure) RETURN s.cid, e ORDER BY e.start",
        params={"id": report_id},
    )
    out = []
    for cid, edge in got.result_set:
        props = dict(edge.properties)
        props["cid"] = cid
        out.append(props)
    return out


__all__ = ["EDGE_FIELDS", "load_sites", "site_row", "sites_of"]
