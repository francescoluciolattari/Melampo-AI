"""Tests for the EMA Product Management Service (PMS) connector."""

from melampo.connectors.pms_ema import (
    PmsEmaConfig,
    PmsEmaConnector,
    _passage_from_product,
)
from melampo.memory.literature_index import LiteratureIndex


def _fhir_resource(**overrides):
    base = {
        "id": "eu-med-00123",
        "resourceType": "MedicinalProductDefinition",
        "name": [{"productName": "Pembrolizumab EMA 25mg/ml"}],
        "ingredient": [{"substance": {"code": {"concept": {"text": "Pembrolizumab"}}}}],
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# Building a passage from a FHIR MedicinalProductDefinition resource
# --------------------------------------------------------------------------


def test_a_complete_resource_becomes_a_checkable_passage():
    passage = _passage_from_product(_fhir_resource())
    assert passage.source_id == "pms-ema:eu-med-00123"
    assert passage.is_independently_checkable is True


def test_active_ingredients_are_included():
    passage = _passage_from_product(_fhir_resource())
    assert "Pembrolizumab" in passage.text


def test_a_resource_with_no_ingredients_still_constructs():
    passage = _passage_from_product(_fhir_resource(ingredient=[]))
    assert passage is not None
    assert "Active ingredients" not in passage.text


def test_a_resource_with_no_id_is_not_constructed():
    assert _passage_from_product(_fhir_resource(id="")) is None


def test_a_resource_with_no_name_is_not_constructed():
    assert _passage_from_product(_fhir_resource(name=[])) is None


def test_a_malformed_ingredient_structure_does_not_crash_parsing():
    resource = _fhir_resource(ingredient=[{"substance": {}}])
    passage = _passage_from_product(resource)
    assert passage is not None
    assert "Active ingredients" not in passage.text


# --------------------------------------------------------------------------
# Availability: registration required, unlike the other three connectors
# --------------------------------------------------------------------------


def test_unconfigured_connector_reports_unavailable():
    connector = PmsEmaConnector()
    assert connector.availability().available is False
    assert connector.search("pembrolizumab") == []


def test_a_configured_connector_reports_available():
    connector = PmsEmaConnector(config=PmsEmaConfig(api_key="a-real-key"))
    assert connector.availability().available is True


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


def test_search_returns_passages():
    connector = PmsEmaConnector(config=PmsEmaConfig(api_key="k"))
    connector._fetch_search_page = lambda name: {"entry": [{"resource": _fhir_resource()}]}

    results = connector.search("pembrolizumab")

    assert len(results) == 1
    assert results[0].source_id == "pms-ema:eu-med-00123"


def test_search_stops_at_max_results():
    connector = PmsEmaConnector(config=PmsEmaConfig(api_key="k"))
    connector._fetch_search_page = lambda name: {
        "entry": [{"resource": _fhir_resource(id=f"id-{i}")} for i in range(5)]
    }
    results = connector.search("pembrolizumab", max_results=2)
    assert len(results) == 2


def test_a_failing_call_degrades_to_an_empty_list():
    connector = PmsEmaConnector(config=PmsEmaConfig(api_key="k"))

    def broken(name):
        raise RuntimeError("beta endpoint unavailable")

    connector._fetch_search_page = broken
    assert connector.search("pembrolizumab") == []


def test_search_for_concepts_queries_once_per_concept():
    connector = PmsEmaConnector(config=PmsEmaConfig(api_key="k"))
    calls = []

    def fake_search(name):
        calls.append(name)
        return {"entry": [{"resource": _fhir_resource(id=f"id-{name}")}]}

    connector._fetch_search_page = fake_search
    connector.search_for_concepts(["pembrolizumab", "nivolumab"])

    assert calls == ["pembrolizumab", "nivolumab"]


def test_populate_adds_results_to_an_index():
    connector = PmsEmaConnector(config=PmsEmaConfig(api_key="k"))
    connector._fetch_search_page = lambda name: {"entry": [{"resource": _fhir_resource()}]}
    index = LiteratureIndex()

    added = connector.populate(index, "pembrolizumab")

    assert added == 1
    assert len(index) == 1


def test_populate_forwards_graph_as_source_graph_to_a_graph_aware_index():
    class _RecordingIndex:
        def __init__(self):
            self.calls = []

        def add_many(self, passages, source_graph=None):
            self.calls.append((list(passages), source_graph))
            return len(passages)

    connector = PmsEmaConnector(config=PmsEmaConfig(api_key="k"))
    connector._fetch_search_page = lambda name: {"entry": [{"resource": _fhir_resource()}]}
    index = _RecordingIndex()
    sentinel_graph = object()

    connector.populate(index, "pembrolizumab", graph=sentinel_graph)

    assert index.calls[0][1] is sentinel_graph


def test_populate_persists_when_given_a_store(tmp_path):
    from melampo.memory.vector_memory import PersistentJsonlVectorStore

    connector = PmsEmaConnector(config=PmsEmaConfig(api_key="k"))
    connector._fetch_search_page = lambda name: {"entry": [{"resource": _fhir_resource()}]}
    store = PersistentJsonlVectorStore(path=tmp_path / "lit.jsonl")
    index = LiteratureIndex()

    connector.populate(index, "pembrolizumab", store=store)

    assert len(store.records) == 1


def test_pms_ema_config_from_env_reads_the_named_variable(monkeypatch):
    monkeypatch.setenv("PMS_EMA_API_KEY", "a-real-key-from-secrets")
    assert PmsEmaConfig.from_env().api_key == "a-real-key-from-secrets"


def test_pms_ema_config_from_env_with_nothing_set_yields_no_key(monkeypatch):
    monkeypatch.delenv("PMS_EMA_API_KEY", raising=False)
    assert PmsEmaConfig.from_env().api_key is None
