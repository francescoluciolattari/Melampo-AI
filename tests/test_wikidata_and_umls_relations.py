"""Tests for connectors/wikidata.py and the UMLS source-relations additions."""

import io
from urllib.error import HTTPError

import pytest

from melampo.connectors.umls import UmlsConfig, UmlsConnector
from melampo.connectors.wikidata import (
    KEY_DOID,
    KEY_MONDO,
    WikidataConnector,
    key_values,
    symptom_query,
)

# --------------------------------------------------------------------------
# Wikidata
# --------------------------------------------------------------------------


def test_mondo_ids_are_queried_in_both_formats():
    assert key_values(KEY_MONDO, ["MONDO:0005812"]) == ["0005812", "MONDO:0005812"]
    assert key_values(KEY_MONDO, ["0005812", "MONDO:0005812"]) == ["0005812", "MONDO:0005812"]


def test_doid_values_are_used_as_given():
    assert key_values(KEY_DOID, ["DOID:8469"]) == ["DOID:8469"]


def test_query_filters_deprecated_statements_and_counts_references():
    query = symptom_query(KEY_MONDO, ["MONDO:0005812"])
    assert "wdt:P5270" in query
    assert "wikibase:DeprecatedRank" in query
    assert "prov:wasDerivedFrom" in query
    assert 'COUNT(DISTINCT ?ref) AS ?refs' in query
    assert '"0005812"' in query and '"MONDO:0005812"' in query


def test_query_escapes_quotes():
    assert '\\"' in symptom_query(KEY_DOID, ['DOID:"x'])


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError):
        symptom_query("icd", ["E11"])


def test_connector_batches_and_concatenates():
    seen = []

    def transport(query):
        seen.append(query)
        return {"results": {"bindings": [{"n": {"value": str(len(seen))}}]}}

    connector = WikidataConnector(batch_size=2, transport=transport)
    rows = connector.symptom_bindings(KEY_MONDO, ["MONDO:1", "MONDO:2", "MONDO:3", "MONDO:1"])
    assert len(seen) == 2  # three unique ids, batches of two
    assert len(rows) == 2


def test_connector_errors_propagate():
    def transport(query):
        raise RuntimeError("endpoint down")

    with pytest.raises(RuntimeError):
        WikidataConnector(transport=transport).symptom_bindings(KEY_MONDO, ["MONDO:1"])


# --------------------------------------------------------------------------
# UMLS source relations
# --------------------------------------------------------------------------


def _http_error(code):
    return HTTPError("https://uts-ws.nlm.nih.gov/x", code, "err", {}, io.BytesIO(b""))


def _connector(transport):
    return UmlsConnector(config=UmlsConfig(api_key="k"), transport=transport)


def test_source_relations_pages_until_a_short_page():
    calls = []

    def transport(url, params):
        calls.append((url, dict(params)))
        size = 2 if params["pageNumber"] == "1" else 1
        return {"result": [{"additionalRelationLabel": "disease_has_finding"}] * size}

    rows = _connector(transport).source_relations("NCI", "C53482", additional_labels=["disease_has_finding"], page_size=2)
    assert len(rows) == 3
    assert calls[0][0].endswith("/content/current/source/NCI/C53482/relations")
    assert calls[0][1]["includeAdditionalRelationLabels"] == "disease_has_finding"
    assert [params["pageNumber"] for _, params in calls] == ["1", "2"]


def test_source_relations_404_means_none():
    def transport(url, params):
        raise _http_error(404)

    assert _connector(transport).source_relations("NCI", "C1") == []


def test_source_relations_other_errors_are_not_swallowed():
    def transport(url, params):
        raise _http_error(500)

    with pytest.raises(HTTPError):
        _connector(transport).source_relations("NCI", "C1")


def test_source_relations_requires_a_key():
    with pytest.raises(RuntimeError):
        UmlsConnector().source_relations("NCI", "C1")


def test_source_codes_for_cui_reads_codes_from_atoms():
    def transport(url, params):
        assert url.endswith("/CUI/C0021400/atoms") and params["sabs"] == "NCI"
        return {
            "result": [
                {"code": "https://uts-ws.nlm.nih.gov/rest/content/2026AA/source/NCI/C53482"},
                {"code": "https://uts-ws.nlm.nih.gov/rest/content/2026AA/source/NCI/C53482"},
            ]
        }

    assert _connector(transport).source_codes_for_cui("C0021400", "NCI") == ["C53482"]
