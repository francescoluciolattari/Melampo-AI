import pytest

from melampo.memory.retrieval_contract import (
    COVERAGE_BASIS_CORPUS,
    COVERAGE_BASIS_NONE,
    COVERAGE_BASIS_TOPK,
    assert_coverage_comparable,
    coverage_basis,
)
from melampo.memory.retriever import MemoryRetriever
from melampo.reasoning.retrieval_reconciliation import (
    build_dual_path_payload,
    reconcile,
)
from melampo.safety.rails import ClinicalSafetyRails


def _offset_evidence(record_id: str | None, *, source: str = "clinical_note") -> dict:
    item = {
        "text": "documented fragment",
        "source": source,
        "grounding_score": 0.7,
        "provenance": {"document_id": "note_1", "char_start": 0, "char_end": 40},
    }
    if record_id:
        item["record_id"] = record_id
    return item


def test_one_shot_retrieval_declares_a_topk_coverage_basis():
    retriever = MemoryRetriever()
    payload = retriever.retrieve("dyspnoea")
    assert coverage_basis(payload) in {COVERAGE_BASIS_TOPK, COVERAGE_BASIS_NONE}
    assert "coverage_basis" in payload["retrieval_quality"]


def test_dual_path_declares_a_corpus_coverage_basis():
    one_shot = {
        "query": "q",
        "focus": "therapy",
        "target_areas": ["therapy"],
        "evidence": [_offset_evidence("note_1:0-40")],
        "retrieval_mode": "semantic_vector_memory",
    }
    recursive = {"query": "q", "evidence": [_offset_evidence("note_1:0-40")]}
    verdict = reconcile(one_shot, recursive)
    payload = build_dual_path_payload(one_shot, recursive, verdict, coverage={"coverage_ratio": 0.5})

    assert coverage_basis(payload) == COVERAGE_BASIS_CORPUS


def test_comparing_coverage_across_bases_raises_instead_of_returning_a_number():
    topk = {"retrieval_mode": "semantic_vector_memory", "retrieval_quality": {"coverage_basis": COVERAGE_BASIS_TOPK}}
    corpus = {"retrieval_mode": "rlm_environment", "retrieval_quality": {"coverage_basis": COVERAGE_BASIS_CORPUS}}

    with pytest.raises(ValueError):
        assert_coverage_comparable(topk, corpus)

    assert assert_coverage_comparable(corpus, corpus) == COVERAGE_BASIS_CORPUS


def test_empty_basis_does_not_block_comparison():
    corpus = {"retrieval_mode": "rlm_environment", "retrieval_quality": {"coverage_basis": COVERAGE_BASIS_CORPUS}}
    empty = {"retrieval_mode": "empty_memory_no_fallback", "retrieval_quality": {}}
    assert assert_coverage_comparable(corpus, empty) == COVERAGE_BASIS_CORPUS


def test_rails_accept_character_offsets_as_provenance():
    rails = ClinicalSafetyRails()
    decision = rails.evaluate_retrieval([_offset_evidence(None)])
    assert "retrieval_provenance_below_threshold" not in decision.reasons
    assert decision.metadata["provenance_fraction"] == 1.0


def test_rails_still_reject_evidence_without_any_trace():
    rails = ClinicalSafetyRails()
    decision = rails.evaluate_retrieval([{"text": "no trace at all", "source": "clinical_note"}])
    assert "retrieval_provenance_below_threshold" in decision.reasons


def test_rails_still_recognise_legacy_record_id_and_page_traces():
    rails = ClinicalSafetyRails()
    legacy = [
        {"text": "a", "source": "note", "record_id": "abc"},
        {"text": "b", "source": "note", "metadata": {"page": 3}},
        {"text": "c", "source": "note", "metadata": {"section": "history"}},
    ]
    decision = rails.evaluate_retrieval(legacy)
    assert decision.metadata["provenance_fraction"] == 1.0


def test_rails_still_flag_synthetic_candidates_used_as_fact():
    rails = ClinicalSafetyRails()
    synthetic = [
        {
            "text": "generated alternative",
            "source": "synthetic_dream_candidate",
            "record_id": "cand_1",
            "learning_status": "candidate",
            "metadata": {"source_type": "synthetic_dream_candidate"},
        }
    ]
    decision = rails.evaluate_retrieval(synthetic)
    assert any("synthetic" in reason for reason in decision.reasons)
