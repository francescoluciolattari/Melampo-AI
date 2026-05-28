from concurrent.futures import ThreadPoolExecutor

from melampo.memory.vector_memory import InMemoryVectorStore
from melampo.memory.visual_imprint import VisualRecognitionImprint, VisualImprintMorpher
from melampo.models.model_client import ModelClientConfig, SafeModelClient
from melampo.training.dream_candidate_store import DreamCandidateStore


def test_model_client_redacts_sensitive_payload_and_enforces_remote_allowlist():
    client = SafeModelClient(
        provider="provider",
        model_name="model",
        role="role",
        config=ModelClientConfig(
            mode="http_json",
            enabled=True,
            endpoint="http://not-loopback.example/api",
            allow_remote=True,
            allowed_endpoint_hosts=["approved.example"],
        ),
    )
    result = client.execute({"case_id": "case-1", "api_key": "secret-value"})

    assert result["status"] == "blocked"
    assert result["reason"] in {"endpoint_must_be_https_or_loopback", "endpoint_host_not_allowlisted"}
    assert result["payload"]["api_key"] == "[REDACTED]"
    assert client.trace.dump()[0]["metadata"]["payload"]["api_key"] == "[REDACTED]"


def test_model_client_blocks_local_subprocess_until_explicitly_allowed():
    client = SafeModelClient(
        provider="provider",
        model_name="model",
        role="role",
        config=ModelClientConfig(
            mode="local_subprocess",
            enabled=True,
            local_command=["python", "-c", "print('{}')"],
            allow_local_subprocess=False,
        ),
    )

    result = client.execute({"case_id": "case-1"})

    assert result["status"] == "blocked"
    assert result["reason"] == "local_subprocess_not_allowed"


def test_in_memory_vector_store_handles_concurrent_upserts():
    store = InMemoryVectorStore.enterprise_default()

    def upsert(index: int) -> str:
        return store.upsert_text(
            record_id=f"record-{index}",
            text=f"opacity fever case {index}",
            metadata={"case_id": str(index)},
        )["record_id"]

    with ThreadPoolExecutor(max_workers=8) as executor:
        record_ids = list(executor.map(upsert, range(40)))

    assert len(set(record_ids)) == 40
    assert store.describe()["record_count"] == 40
    assert len(store.search("opacity fever", limit=5)) == 5


def test_dream_candidate_store_handles_concurrent_candidate_creation():
    store = DreamCandidateStore()

    def create(index: int) -> str:
        return store.create_candidate({"text": f"candidate {index}"}, case_id=f"case-{index}").candidate_id

    with ThreadPoolExecutor(max_workers=8) as executor:
        candidate_ids = list(executor.map(create, range(30)))

    assert len(set(candidate_ids)) == 30
    assert len(store.list_by_status(["candidate"])) == 30


def _imprint(concept: str, label: str, vector: list[float]) -> dict:
    return VisualRecognitionImprint.from_payload(
        {
            "semantic_concept": concept,
            "variant_label": label,
            "vector": vector,
            "learning_status": "candidate",
        }
    ).as_dict()


def test_visual_imprint_morpher_enforces_pair_budget_and_vector_suppression():
    imprints = [
        _imprint("opacity", f"variant-{index}", [1.0 - index * 0.05, index * 0.05, 0.0, 0.0])
        for index in range(5)
    ]

    result = VisualImprintMorpher(max_pairs=2, return_vectors=False, min_similarity=0.0).dream_morph(
        concept_imprints=imprints,
        diagnostic_imprints=imprints[:1],
    )

    assert result["evaluated_pair_count"] == 2
    assert result["pair_budget_exhausted"] is True
    assert result["governance"]["pair_budget_enforced"] is True
    assert "vector" not in result["visual_morph_candidates"][0]
