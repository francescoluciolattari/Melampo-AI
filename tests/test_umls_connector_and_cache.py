"""Tests for the UMLS connector, its encrypted cache, and the bridge into the normalisation cascade."""

import tempfile
from pathlib import Path

from melampo.connectors.umls import (
    CrosswalkResult,
    UmlsConfig,
    UmlsConnector,
)
from melampo.memory.concept_normalisation import NormalisationCascade
from melampo.memory.concept_paths import ConceptEdge, InMemoryConceptGraph
from melampo.memory.concept_resolution import TermIndex
from melampo.memory.encrypted_store import EncryptedJsonlStore
from melampo.memory.umls_cache import UmlsCache, crosswalk_with_cache

_RTA_OBO = "[Term]\nid: HP:0001947\nname: Renal tubular acidosis\n"


class _FakeUmls:
    """A connector stand-in matching the confirmed NLM crosswalk example."""

    def __init__(self):
        self.calls = 0

    def crosswalk_from_hpo(self, hpo_id, target_source=None):
        self.calls += 1
        if hpo_id == "HP:0001947":
            return [CrosswalkResult("233604007", "Distal renal tubular acidosis", "SNOMEDCT_US", "C0022099")]
        return []


# --------------------------------------------------------------------------
# UmlsConnector: availability, search, crosswalk, atoms
# --------------------------------------------------------------------------


def test_unconfigured_connector_reports_unavailable():
    connector = UmlsConnector()
    assert connector.availability().available is False
    assert connector.crosswalk_from_hpo("HP:0001947") == []
    assert connector.search("insulin") == []
    assert connector.atoms_for_cui("C0009044") == []


def test_a_configured_connector_reports_available():
    connector = UmlsConnector(config=UmlsConfig(api_key="a-real-key"))
    assert connector.availability().available is True


def test_crosswalk_matches_the_confirmed_nlm_documentation_example():
    """https://uts-ws.nlm.nih.gov/rest/crosswalk/current/source/HPO/HP:0001947
    ?targetSource=SNOMEDCT_US -- NLM's own worked example, verified before
    building anything against it."""
    connector = UmlsConnector(config=UmlsConfig(api_key="k"))
    connector._get = lambda url, params: {
        "result": [
            {"ui": "233604007", "name": "Renal tubular acidosis", "rootSource": "SNOMEDCT_US",
             "concepts": [{"ui": "C0022099"}]}
        ]
    }

    results = connector.crosswalk_from_hpo("HP:0001947", target_source="SNOMEDCT_US")

    assert results[0].ui == "233604007"
    assert results[0].source_cui == "C0022099"


def test_crosswalk_with_no_results_returns_an_empty_list():
    connector = UmlsConnector(config=UmlsConfig(api_key="k"))
    connector._get = lambda url, params: {"result": []}
    assert connector.crosswalk_from_hpo("HP:9999999") == []


def test_a_failing_call_degrades_gracefully():
    connector = UmlsConnector(config=UmlsConfig(api_key="k"))

    def broken(url, params):
        raise RuntimeError("endpoint down")

    connector._get = broken
    assert connector.crosswalk_from_hpo("HP:0001947") == []
    assert connector.search("insulin") == []
    assert connector.atoms_for_cui("C0009044") == []


def test_search_returns_concepts_with_cuis():
    connector = UmlsConnector(config=UmlsConfig(api_key="k"))
    connector._get = lambda url, params: {
        "result": {"results": [{"ui": "C0021400", "name": "Insulin", "rootSource": "MSH"}]}
    }
    results = connector.search("insulin")
    assert results[0].cui == "C0021400"


def test_search_excludes_the_none_sentinel():
    """UMLS returns ui: 'NONE' for a query with no match at all -- not a
    real CUI, and must not be treated as one."""
    connector = UmlsConnector(config=UmlsConfig(api_key="k"))
    connector._get = lambda url, params: {"result": {"results": [{"ui": "NONE", "name": "no results"}]}}
    assert connector.search("nonsense query") == []


def test_atoms_for_cui_returns_every_synonym_name():
    connector = UmlsConnector(config=UmlsConfig(api_key="k"))
    connector._get = lambda url, params: {"result": [{"name": "Renal tubular acidosis"}, {"name": "RTA"}]}
    atoms = connector.atoms_for_cui("C0022099")
    assert set(atoms) == {"Renal tubular acidosis", "RTA"}


# --------------------------------------------------------------------------
# UmlsCache: served from the encrypted store, one call per distinct query
# --------------------------------------------------------------------------


def test_the_cache_serves_a_repeated_query_without_a_second_call():
    with tempfile.TemporaryDirectory() as directory:
        cache = UmlsCache(store=EncryptedJsonlStore(path=Path(directory) / "cache.jsonl", password="x"))
        fake = _FakeUmls()

        crosswalk_with_cache(fake, cache, "HP:0001947")
        crosswalk_with_cache(fake, cache, "HP:0001947")

    assert fake.calls == 1


def test_the_cache_distinguishes_never_queried_from_queried_and_empty():
    """An empty list is a real, worth-keeping answer ('UMLS has no
    crosswalk for this'), not the same as 'never looked this up'."""
    with tempfile.TemporaryDirectory() as directory:
        cache = UmlsCache(store=EncryptedJsonlStore(path=Path(directory) / "cache.jsonl", password="x"))
        assert cache.get("HP:0001947") is None

        cache.put("HP:0001947", None, [])
        assert cache.get("HP:0001947") == []


def test_the_cache_is_encrypted_on_disk():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cache.jsonl"
        cache = UmlsCache(store=EncryptedJsonlStore(path=path, password="x"))
        crosswalk_with_cache(_FakeUmls(), cache, "HP:0001947")

        raw = path.read_bytes()

    assert b"Distal renal tubular acidosis" not in raw


def test_different_target_sources_are_cached_separately():
    with tempfile.TemporaryDirectory() as directory:
        cache = UmlsCache(store=EncryptedJsonlStore(path=Path(directory) / "cache.jsonl", password="x"))
        fake = _FakeUmls()

        crosswalk_with_cache(fake, cache, "HP:0001947", target_source="SNOMEDCT_US")
        crosswalk_with_cache(fake, cache, "HP:0001947", target_source="RXNORM")

    assert fake.calls == 2


def test_a_cache_reopened_from_disk_still_serves_without_a_new_call():
    """The actual restart scenario: a fresh UmlsCache instance pointed at
    the same encrypted file."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cache.jsonl"
        first_process = UmlsCache(store=EncryptedJsonlStore(path=path, password="x"))
        crosswalk_with_cache(_FakeUmls(), first_process, "HP:0001947")

        second_process = UmlsCache(store=EncryptedJsonlStore(path=path, password="x"))
        fake_in_second_process = _FakeUmls()
        result = crosswalk_with_cache(fake_in_second_process, second_process, "HP:0001947")

    assert fake_in_second_process.calls == 0
    assert result[0].name == "Distal renal tubular acidosis"


# --------------------------------------------------------------------------
# Bridged into the normalisation cascade's lexical tier
# --------------------------------------------------------------------------


def test_a_umls_crosswalked_synonym_resolves_through_the_cascade():
    """The scenario that motivated this: a clinical phrase closer to SNOMED
    usage than to HPO's own wording, resolved via UMLS without this project
    needing a direct SNOMED license.

    "RTA" shares no words at all with "Renal tubular acidosis" -- chosen
    deliberately so this test cannot pass by accident through the existing
    word-set matching the way an earlier version of it did, when the phrase
    "distal renal tubular acidosis" turned out to already match via
    containment alone, isolating nothing about UMLS's actual contribution.
    """
    graph = InMemoryConceptGraph.from_edges(
        [ConceptEdge("some syndrome", "has_phenotype", "Renal tubular acidosis", 0.8)]
    )
    index = TermIndex.from_obo(_RTA_OBO.splitlines())
    umls = _FakeUmls()
    umls.crosswalk_from_hpo = lambda hpo_id, target_source=None: (
        [CrosswalkResult("233604007", "RTA", "SNOMEDCT_US", "C0022099")] if hpo_id == "HP:0001947" else []
    )
    cascade = NormalisationCascade(graph=graph, synonym_index=index, umls=umls)

    without_umls = NormalisationCascade(graph=graph, synonym_index=index).resolve(
        "rta", candidates=["Renal tubular acidosis"]
    )
    assert without_umls.resolved is False, "the phrase must not be resolvable without UMLS, or this test isolates nothing"

    result = cascade.resolve("rta", candidates=["Renal tubular acidosis"])

    assert result.concept == "Renal tubular acidosis"
    assert result.tier == "lexical"
    assert result.is_deterministic is True


def test_without_umls_configured_the_cascade_behaves_as_before():
    """A phrase that word-set matching alone cannot bridge -- if UMLS is not
    configured, resolution must fail, not fall back to some other path."""
    graph = InMemoryConceptGraph.from_edges(
        [ConceptEdge("some syndrome", "has_phenotype", "Renal tubular acidosis", 0.8)]
    )
    index = TermIndex.from_obo(_RTA_OBO.splitlines())
    cascade = NormalisationCascade(graph=graph, synonym_index=index)

    result = cascade.resolve("rta", candidates=["Renal tubular acidosis"])

    assert result.resolved is False
