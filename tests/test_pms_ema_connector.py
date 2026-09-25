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


def _configured_connector(**overrides) -> PmsEmaConnector:
    config_kwargs = {"client_id": "a-client-id", "client_secret": "a-client-secret"}
    config_kwargs.update(overrides)
    return PmsEmaConnector(config=PmsEmaConfig(**config_kwargs))


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
    connector = _configured_connector()
    assert connector.availability().available is True


def test_a_client_id_without_a_secret_is_still_unavailable():
    connector = PmsEmaConnector(config=PmsEmaConfig(client_id="a-client-id"))
    assert connector.availability().available is False


def test_a_client_secret_without_an_id_is_still_unavailable():
    connector = PmsEmaConnector(config=PmsEmaConfig(client_secret="a-client-secret"))
    assert connector.availability().available is False


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


def test_search_returns_passages():
    connector = _configured_connector()
    connector._fetch_search_page = lambda name: {"entry": [{"resource": _fhir_resource()}]}

    results = connector.search("pembrolizumab")

    assert len(results) == 1
    assert results[0].source_id == "pms-ema:eu-med-00123"


def test_search_stops_at_max_results():
    connector = _configured_connector()
    connector._fetch_search_page = lambda name: {
        "entry": [{"resource": _fhir_resource(id=f"id-{i}")} for i in range(5)]
    }
    results = connector.search("pembrolizumab", max_results=2)
    assert len(results) == 2


def test_a_failing_call_degrades_to_an_empty_list():
    connector = _configured_connector()

    def broken(name):
        raise RuntimeError("beta endpoint unavailable")

    connector._fetch_search_page = broken
    assert connector.search("pembrolizumab") == []


def test_search_for_concepts_queries_once_per_concept():
    connector = _configured_connector()
    calls = []

    def fake_search(name):
        calls.append(name)
        return {"entry": [{"resource": _fhir_resource(id=f"id-{name}")}]}

    connector._fetch_search_page = fake_search
    connector.search_for_concepts(["pembrolizumab", "nivolumab"])

    assert calls == ["pembrolizumab", "nivolumab"]


def test_populate_adds_results_to_an_index():
    connector = _configured_connector()
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

    connector = _configured_connector()
    connector._fetch_search_page = lambda name: {"entry": [{"resource": _fhir_resource()}]}
    index = _RecordingIndex()
    sentinel_graph = object()

    connector.populate(index, "pembrolizumab", graph=sentinel_graph)

    assert index.calls[0][1] is sentinel_graph


def test_populate_persists_when_given_a_store(tmp_path):
    from melampo.memory.vector_memory import PersistentJsonlVectorStore

    connector = _configured_connector()
    connector._fetch_search_page = lambda name: {"entry": [{"resource": _fhir_resource()}]}
    store = PersistentJsonlVectorStore(path=tmp_path / "lit.jsonl")
    index = LiteratureIndex()

    connector.populate(index, "pembrolizumab", store=store)

    assert len(store.records) == 1


def test_pms_ema_config_from_env_reads_the_named_variables(monkeypatch):
    monkeypatch.setenv("PMS_EMA_CLIENT_ID", "cid-from-secrets")
    monkeypatch.setenv("PMS_EMA_CLIENT_SECRET", "secret-from-secrets")
    config = PmsEmaConfig.from_env()
    assert config.client_id == "cid-from-secrets"
    assert config.client_secret == "secret-from-secrets"


def test_pms_ema_config_from_env_with_nothing_set_yields_no_credentials(monkeypatch):
    monkeypatch.delenv("PMS_EMA_CLIENT_ID", raising=False)
    monkeypatch.delenv("PMS_EMA_CLIENT_SECRET", raising=False)
    config = PmsEmaConfig.from_env()
    assert config.client_id is None
    assert config.client_secret is None


# --------------------------------------------------------------------------
# OAuth2 client-credentials token exchange
#
# EMA moved from a single registered API key to a Microsoft Entra ID
# client-credentials flow (see the module docstring). These tests cover the
# caching/expiry logic in `_get_access_token`; the actual HTTP exchange in
# `_request_token` is a network call, exercised here only through injection,
# the same way `_fetch_search_page`'s network call is never itself invoked.
# --------------------------------------------------------------------------


def test_get_access_token_requests_once_and_caches():
    connector = _configured_connector()
    calls = []

    def fake_request_token():
        calls.append(1)
        return {"access_token": "tok-1", "expires_in": 3600}

    connector._request_token = fake_request_token

    assert connector._get_access_token() == "tok-1"
    assert connector._get_access_token() == "tok-1"
    assert len(calls) == 1


def test_get_access_token_refreshes_once_the_cached_token_is_stale():
    import time

    connector = _configured_connector()
    tokens = iter(["tok-1", "tok-2"])
    connector._request_token = lambda: {"access_token": next(tokens), "expires_in": 3600}

    first = connector._get_access_token()
    connector._token_expiry = time.monotonic() - 1  # force the cached token to look expired

    second = connector._get_access_token()

    assert first == "tok-1"
    assert second == "tok-2"


def test_get_access_token_refreshes_early_within_the_safety_margin():
    """A token is renewed slightly before its stated deadline, not exactly at it.

    expires_in=61 with a 60-second margin leaves the cached token valid for
    only ~1 second -- effectively already due for renewal by the time a
    second call is made.
    """
    import time

    connector = _configured_connector()
    tokens = iter(["tok-1", "tok-2"])
    connector._request_token = lambda: {"access_token": next(tokens), "expires_in": 61}

    connector._get_access_token()
    connector._token_expiry = time.monotonic() - 0.001

    assert connector._get_access_token() == "tok-2"


def test_get_access_token_defaults_expiry_when_the_response_omits_it():
    connector = _configured_connector()
    connector._request_token = lambda: {"access_token": "tok-1"}
    assert connector._get_access_token() == "tok-1"


def test_search_fetches_a_token_through_the_real_fetch_path(monkeypatch):
    """search() drives _fetch_search_page, which asks for a token -- verified
    end to end with only the two real network calls (_request_token and the
    HTTP GET inside _fetch_search_page) stubbed out."""
    connector = _configured_connector()
    connector._request_token = lambda: {"access_token": "tok-1", "expires_in": 3600}

    captured_auth = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self):
            import json

            return json.dumps({"entry": [{"resource": _fhir_resource()}]}).encode("utf-8")

    def fake_urlopen(request, timeout=30):
        captured_auth["Authorization"] = request.get_header("Authorization")
        return _FakeResponse()

    monkeypatch.setattr("melampo.connectors.pms_ema.urlopen", fake_urlopen)

    results = connector.search("pembrolizumab")

    assert len(results) == 1
    assert captured_auth["Authorization"] == "Bearer tok-1"
