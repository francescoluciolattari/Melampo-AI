"""Tests for LiteratureIndex persistence via the vector store, and the
tracked-concept refresh queue."""

import tempfile
from pathlib import Path

from melampo.connectors.clinical_trials import ClinicalTrialsConnector
from melampo.connectors.europe_pmc import EuropePmcConnector
from melampo.evaluation.enumeration_bench import differential_graph
from melampo.evaluation.vetting_bench import VETTING_CASES
from melampo.memory.literature_index import LiteratureIndex, LiteraturePassage
from melampo.memory.literature_persistence import (
    load_literature_index,
    metadata_to_passage,
    passage_to_metadata,
    persist_all,
    persist_passage,
)
from melampo.memory.tracked_concepts import (
    SOURCE_VETTING_BENCH,
    TrackedConceptStore,
    seed_from_vetting_bench,
)
from melampo.memory.vector_memory import PersistentJsonlVectorStore

_PASSAGE = LiteraturePassage(
    "p1", "Sarcoidosis causes hypercalcaemia via calcitriol excess.",
    "Calcium in sarcoidosis", "pmid:12345678", 2024, "Chest",
)


# --------------------------------------------------------------------------
# The core round trip: LiteraturePassage <-> the project's own vector store
# --------------------------------------------------------------------------


def test_a_passage_round_trips_through_metadata_losslessly():
    metadata = passage_to_metadata(_PASSAGE)
    rebuilt = metadata_to_passage(_PASSAGE.text, metadata)

    assert rebuilt == _PASSAGE


def test_persisting_a_passage_and_reloading_survives_a_fresh_store_instance():
    """The actual restart scenario: a second, independent store instance
    pointed at the same path, simulating a new process."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "literature.jsonl"

        first_process = PersistentJsonlVectorStore(path=path)
        persist_passage(first_process, _PASSAGE)

        second_process = PersistentJsonlVectorStore(path=path)
        reloaded_index = load_literature_index(second_process)

    assert len(reloaded_index) == 1
    assert reloaded_index.passages[0].source_id == "pmid:12345678"


def test_persist_all_writes_every_passage_in_an_index():
    index = LiteratureIndex()
    index.add(_PASSAGE)
    index.add(LiteraturePassage("p2", "Another passage.", "Another title", "pmid:99", 2023, "Lancet"))

    with tempfile.TemporaryDirectory() as directory:
        store = PersistentJsonlVectorStore(path=Path(directory) / "lit.jsonl")
        written = persist_all(store, index)

    assert written == 2
    assert len(store.records) == 2


def test_persisting_the_same_passage_twice_does_not_duplicate_it():
    """upsert deduplicates by record_id, set here to the passage's own id --
    a passage rediscovered by a later refresh updates in place."""
    with tempfile.TemporaryDirectory() as directory:
        store = PersistentJsonlVectorStore(path=Path(directory) / "lit.jsonl")
        persist_passage(store, _PASSAGE)
        persist_passage(store, _PASSAGE)

    assert len(store.records) == 1


def test_loaded_passages_are_still_searched_by_concept_not_by_embedding():
    """The design point: persistence changed, relevance ranking did not.
    literature_index.py deliberately avoids embedding-based retrieval, and
    loading through the vector store must not reintroduce it."""
    with tempfile.TemporaryDirectory() as directory:
        store = PersistentJsonlVectorStore(path=Path(directory) / "lit.jsonl")
        persist_passage(store, _PASSAGE)

        reloaded = load_literature_index(store)

    hits = reloaded.search(["sarcoidosis", "hypercalcaemia"], differential_graph())
    assert len(hits) == 1
    assert hits[0].passage.passage_id == "p1"


def test_the_origin_label_distinguishes_pubmed_from_clinical_trials():
    from melampo.memory.literature_persistence import _origin_for

    assert _origin_for("pmid:12345") == "europe_pmc"
    assert _origin_for("nct:NCT01234567") == "clinical_trials"


# --------------------------------------------------------------------------
# Both connectors write to the same store
# --------------------------------------------------------------------------


def test_europe_pmc_connector_persists_when_given_a_store():
    connector = EuropePmcConnector()
    connector._fetch_page = lambda q, c: {
        "resultList": {"result": [{"id": "1", "pmid": "1", "title": "T", "abstractText": "A"}]},
        "nextCursorMark": None,
    }
    with tempfile.TemporaryDirectory() as directory:
        store = PersistentJsonlVectorStore(path=Path(directory) / "lit.jsonl")
        index = LiteratureIndex()
        connector.populate(index, "query", store=store)

    assert len(store.records) == 1


def test_clinical_trials_connector_persists_to_the_same_kind_of_store():
    connector = ClinicalTrialsConnector()
    connector._fetch_page = lambda q: {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {"nctId": "NCT01234567", "briefTitle": "T"},
                    "descriptionModule": {"briefSummary": "S"},
                }
            }
        ]
    }
    with tempfile.TemporaryDirectory() as directory:
        store = PersistentJsonlVectorStore(path=Path(directory) / "lit.jsonl")
        index = LiteratureIndex()
        connector.populate(index, "query", store=store)

    assert len(store.records) == 1


def test_populate_without_a_store_behaves_exactly_as_before():
    """Additive: a caller not passing store gets the same behaviour that
    existed before persistence was added."""
    connector = EuropePmcConnector()
    connector._fetch_page = lambda q, c: {
        "resultList": {"result": [{"id": "1", "pmid": "1", "title": "T", "abstractText": "A"}]},
        "nextCursorMark": None,
    }
    index = LiteratureIndex()
    added = connector.populate(index, "query")

    assert added == 1
    assert len(index) == 1


# --------------------------------------------------------------------------
# TrackedConceptStore: a queue, not a bare list
# --------------------------------------------------------------------------


def test_tracking_a_new_concept_returns_true():
    store = TrackedConceptStore()
    assert store.track("sarcoidosis", source=SOURCE_VETTING_BENCH) is True
    assert len(store) == 1


def test_tracking_an_already_tracked_concept_is_a_no_op():
    """Re-adding a concept must not reset its refresh history."""
    store = TrackedConceptStore()
    store.track("sarcoidosis", source=SOURCE_VETTING_BENCH)
    store.mark_refreshed("sarcoidosis", passage_count=3)

    added_again = store.track("sarcoidosis", source="confirmed_case")

    assert added_again is False
    assert store.concepts["sarcoidosis"].passage_count == 3


def test_never_refreshed_concepts_are_due_first():
    store = TrackedConceptStore()
    store.track("a", source=SOURCE_VETTING_BENCH)
    store.track("b", source=SOURCE_VETTING_BENCH)
    store.mark_refreshed("a", passage_count=1)

    batch = store.next_batch(limit=1)

    assert batch[0].concept == "b"


def test_the_longest_unrefreshed_concept_comes_before_a_recently_refreshed_one():
    import time

    store = TrackedConceptStore()
    store.track("old", source=SOURCE_VETTING_BENCH)
    store.mark_refreshed("old", passage_count=1)
    time.sleep(0.01)
    store.track("older_refresh_needed", source=SOURCE_VETTING_BENCH)
    store.mark_refreshed("older_refresh_needed", passage_count=1)
    store.concepts["old"].last_refreshed_at = store.concepts["older_refresh_needed"].last_refreshed_at + 100

    batch = store.next_batch(limit=1)

    assert batch[0].concept == "older_refresh_needed"


def test_next_batch_respects_the_limit():
    """The real ceiling PubMed's unauthenticated rate limit imposes -- 3
    requests per second bounds how many concepts one run can touch."""
    store = TrackedConceptStore()
    for i in range(10):
        store.track(f"concept-{i}", source=SOURCE_VETTING_BENCH)

    assert len(store.next_batch(limit=3)) == 3


def test_seeding_from_the_vetting_bench_tracks_factor_and_target_of_every_case():
    store = TrackedConceptStore()
    added = seed_from_vetting_bench(store, VETTING_CASES)

    assert added > 0
    assert "sarcoidosis" in store.concepts
    assert store.concepts["sarcoidosis"].added_from == SOURCE_VETTING_BENCH


def test_the_store_persists_and_reloads():
    store = TrackedConceptStore()
    store.track("sarcoidosis", source=SOURCE_VETTING_BENCH)
    store.mark_refreshed("sarcoidosis", passage_count=5)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "concepts.json"
        store.save(path)
        reloaded = TrackedConceptStore.load(path)

    assert len(reloaded) == 1
    assert reloaded.concepts["sarcoidosis"].passage_count == 5


def test_loading_a_missing_file_yields_an_empty_store_not_an_error():
    with tempfile.TemporaryDirectory() as directory:
        store = TrackedConceptStore.load(Path(directory) / "never_written.json")
    assert len(store) == 0


def test_an_empty_concept_is_not_tracked():
    store = TrackedConceptStore()
    assert store.track("   ", source=SOURCE_VETTING_BENCH) is False
    assert len(store) == 0


def test_marking_an_untracked_concept_refreshed_does_not_crash():
    store = TrackedConceptStore()
    store.mark_refreshed("never tracked", passage_count=1)  # must not raise
    assert len(store) == 0
