from melampo.memory.weaviate_adapter import WeaviateEnterpriseMemoryAdapter
from melampo.memory.weaviate_schema import (
    QUARANTINED_HYPOTHESIS_CLASS,
    MelampoWeaviateSchema,
)
from melampo.training.hypothesis_channel import (
    HypothesisChannel,
    HypothesisEnvelope,
)

# --------------------------------------------------------------------------
# Schema partition
# --------------------------------------------------------------------------


def test_the_candidate_class_is_not_among_the_evidence_classes():
    schema = MelampoWeaviateSchema()
    assert QUARANTINED_HYPOTHESIS_CLASS in schema.class_names()
    assert QUARANTINED_HYPOTHESIS_CLASS in schema.quarantined_class_names()
    assert QUARANTINED_HYPOTHESIS_CLASS not in schema.evidence_class_names()


def test_evidence_classes_are_not_quarantined():
    schema = MelampoWeaviateSchema()
    assert schema.evidence_class_names()
    for name in schema.evidence_class_names():
        assert schema.is_quarantined(name) is False


def test_the_candidate_class_declares_its_own_review_requirement():
    schema = MelampoWeaviateSchema()
    candidate = next(
        item for item in schema.classes() if item.name == QUARANTINED_HYPOTHESIS_CLASS
    )
    names = {prop.name for prop in candidate.properties}
    assert {
        "learning_status",
        "promotion_state",
        "human_review_before_clinical_use",
    } <= names
    assert "quarantined" in candidate.description.lower()


# --------------------------------------------------------------------------
# The negative test: the exit criterion for physical isolation
# --------------------------------------------------------------------------


def _adapter_with_a_candidate() -> WeaviateEnterpriseMemoryAdapter:
    """Store one synthetic candidate and one piece of clinical evidence."""
    adapter = WeaviateEnterpriseMemoryAdapter()
    adapter.fallback_store.upsert(
        text="amyloidosis considered as a synthetic alternative",
        metadata={
            "class_name": QUARANTINED_HYPOTHESIS_CLASS,
            "source_type": "synthetic_nexus_candidate",
        },
        learning_status="candidate",
    )
    adapter.fallback_store.upsert(
        text="amyloidosis documented in the admission report",
        metadata={"class_name": "Pathology", "source_type": "clinical_document"},
        learning_status="grounded",
    )
    return adapter


def test_a_candidate_is_not_retrievable_through_the_evidence_path():
    """Attempts retrieval and expects failure.

    Inspecting the schema proves how the store is configured, not that a
    candidate is unreachable. Only an attempt proves that.
    """
    adapter = _adapter_with_a_candidate()
    result = adapter.hybrid_search(query="amyloidosis", limit=10)

    texts = [hit.get("text", "") for hit in result["hits"]]
    assert any("admission report" in text for text in texts), (
        "evidence must still be reachable"
    )
    assert not any("synthetic alternative" in text for text in texts)


def test_omitting_a_filter_does_not_reopen_the_path():
    """Exclusion is applied after retrieval, so it cannot be bypassed by omission."""
    adapter = _adapter_with_a_candidate()
    unfiltered = adapter.hybrid_search(query="amyloidosis", limit=10, filters=None)
    assert all(
        hit.get("metadata", {}).get("class_name") != QUARANTINED_HYPOTHESIS_CLASS
        for hit in unfiltered["hits"]
    )


def test_asking_for_the_quarantined_class_is_refused_not_silently_empty():
    """An empty result would read as 'no such evidence' instead of 'not permitted'."""
    adapter = _adapter_with_a_candidate()
    result = adapter.hybrid_search(
        query="amyloidosis", class_name=QUARANTINED_HYPOTHESIS_CLASS
    )

    assert result["status"] == "refused"
    assert result["hits"] == []
    assert "quarantined" in result["reason"]
    assert "hypothesis_search" in result["hint"]


def test_the_dedicated_channel_reaches_candidates_and_marks_them():
    adapter = _adapter_with_a_candidate()
    result = adapter.hypothesis_search(query="amyloidosis")

    assert result["status"] == "completed"
    assert result["usable_as_evidence"] is False
    assert result["hits"], "the candidate is reachable through its own channel"
    for hit in result["hits"]:
        assert hit["usable_as_evidence"] is False
        assert hit["retrieval_channel"] == "hypothesis_channel"


def test_the_dedicated_channel_does_not_return_clinical_evidence():
    adapter = _adapter_with_a_candidate()
    texts = [
        hit.get("text", "")
        for hit in adapter.hypothesis_search(query="amyloidosis")["hits"]
    ]
    assert not any("admission report" in text for text in texts)


def test_quarantine_holds_when_the_candidate_outranks_the_evidence():
    """Ranking must not be the thing keeping candidates out."""
    adapter = WeaviateEnterpriseMemoryAdapter()
    adapter.fallback_store.upsert(
        text="rare pulmonary amyloidosis exact match",
        metadata={"class_name": QUARANTINED_HYPOTHESIS_CLASS},
        learning_status="candidate",
    )
    result = adapter.hybrid_search(
        query="rare pulmonary amyloidosis exact match", limit=5
    )
    assert result["hits"] == []


# --------------------------------------------------------------------------
# The gate opens when the caller says the diagnosis is unclear
# --------------------------------------------------------------------------


def test_the_channel_opens_only_when_the_caller_says_unclear():
    channel = HypothesisChannel(max_hypotheses=2)
    envelope = [HypothesisEnvelope(label="amyloidosis", novelty_score=0.8)]
    closed = channel.open_for(envelope, unclear=False)
    assert closed["channel_open"] is False and closed["hypotheses"] == []
    opened = channel.open_for(envelope, unclear=True, reasons=["two unexplained findings"])
    assert opened["channel_open"] is True
    assert opened["gate"]["reasons"] == ["two unexplained findings"]
