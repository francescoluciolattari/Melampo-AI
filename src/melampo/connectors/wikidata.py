"""Query Wikidata's SPARQL endpoint for disease symptoms (P780), keyed by Mondo or DOID.

Wikidata's structured data is CC0. It is community-edited, so this
connector returns, for every statement, how many references support it --
`memory/symptom_sources.links_from_wikidata` turns that count into an
evidence tier rather than letting an unreferenced statement weigh the same
as a referenced one. Deprecated-rank statements are excluded in the query.

Diseases are looked up by the identifiers Wikidata itself records: Mondo
(P5270) first, the Disease Ontology (P699) as a second key, because a
Wikidata item often carries one and not the other. The value format of
P5270 has varied between ``0005148`` and ``MONDO:0005148``; both are
queried, so a formatting convention cannot silently empty the result.

Etiquette the endpoint asks for: a descriptive User-Agent with a contact,
POST for long queries, and modest batches.
"""

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
DEFAULT_USER_AGENT = "melampo-symptom-coverage/0.1 (francesco.lucio.lattari@gmail.com)"
BATCH_SIZE = 50
SECONDS_BETWEEN_QUERIES = 1.0

KEY_MONDO = "mondo"
KEY_DOID = "doid"
_KEY_PROPERTY = {KEY_MONDO: "P5270", KEY_DOID: "P699"}


def _literal(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def key_values(key: str, identifiers: Iterable[str]) -> list[str]:
    """The literal values to match for each identifier, in every format Wikidata may use."""
    values: list[str] = []
    for identifier in identifiers:
        identifier = identifier.strip()
        if not identifier:
            continue
        if key == KEY_MONDO:
            local = identifier.split(":", 1)[1] if identifier.upper().startswith("MONDO:") else identifier
            candidates = [local, f"MONDO:{local}"]
        else:
            candidates = [identifier]
        for candidate in candidates:
            if candidate not in values:
                values.append(candidate)
    return values


def symptom_query(key: str, identifiers: Sequence[str]) -> str:
    """SPARQL for P780 statements on items carrying one of ``identifiers`` under ``key``."""
    if key not in _KEY_PROPERTY:
        raise ValueError(f"key must be one of {sorted(_KEY_PROPERTY)}, got {key!r}")
    values = " ".join(_literal(value) for value in key_values(key, identifiers))
    prop = _KEY_PROPERTY[key]
    return f"""
SELECT ?disease ?diseaseLabel ?{key} ?symptom ?symptomLabel ?symptomLabelIt ?hpo (COUNT(DISTINCT ?ref) AS ?refs)
WHERE {{
  VALUES ?{key} {{ {values} }}
  ?disease wdt:{prop} ?{key} .
  ?disease p:P780 ?statement .
  ?statement ps:P780 ?symptom ; wikibase:rank ?rank .
  FILTER(?rank != wikibase:DeprecatedRank)
  OPTIONAL {{ ?statement prov:wasDerivedFrom ?ref . }}
  OPTIONAL {{ ?symptom wdt:P3841 ?hpo . }}
  OPTIONAL {{ ?disease rdfs:label ?diseaseLabel . FILTER(LANG(?diseaseLabel) = "en") }}
  OPTIONAL {{ ?symptom rdfs:label ?symptomLabel . FILTER(LANG(?symptomLabel) = "en") }}
  OPTIONAL {{ ?symptom rdfs:label ?symptomLabelIt . FILTER(LANG(?symptomLabelIt) = "it") }}
}}
GROUP BY ?disease ?diseaseLabel ?{key} ?symptom ?symptomLabel ?symptomLabelIt ?hpo
""".strip()


@dataclass
class WikidataConnector:
    """Batch P780 lookups against the public SPARQL endpoint."""

    user_agent: str = DEFAULT_USER_AGENT
    endpoint: str = WIKIDATA_SPARQL
    batch_size: int = BATCH_SIZE
    pause_seconds: float = SECONDS_BETWEEN_QUERIES
    transport: Any = None
    """Injectable (query) -> parsed JSON, for testing without the live endpoint."""

    def symptom_bindings(self, key: str, identifiers: Sequence[str]) -> list[dict[str, Any]]:
        """Every result row for the identifiers, batched. Errors propagate: a
        coverage measurement must not mistake a failed query for "no symptoms"."""
        rows: list[dict[str, Any]] = []
        unique = list(dict.fromkeys(identifier for identifier in identifiers if identifier))
        for start in range(0, len(unique), self.batch_size):
            batch = unique[start : start + self.batch_size]
            payload = self._query(symptom_query(key, batch))
            rows.extend(payload.get("results", {}).get("bindings", []))
            if self.transport is None and start + self.batch_size < len(unique):
                time.sleep(self.pause_seconds)
        return rows

    def _query(self, query: str) -> dict[str, Any]:
        if self.transport is not None:
            return self.transport(query)
        return self._post(query)  # pragma: no cover - network call

    def _post(self, query: str) -> dict[str, Any]:  # pragma: no cover - network call
        request = Request(
            self.endpoint,
            data=urlencode({"query": query}).encode("utf-8"),
            headers={
                "Accept": "application/sparql-results+json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": self.user_agent,
            },
            method="POST",
        )
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
