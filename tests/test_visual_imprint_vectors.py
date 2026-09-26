"""Regression tests for the visual imprint vector input (memory/visual_imprint.py).

Defect found on 2026-09-26 (docs/imaging_decision_record.md): every vector went
through _matrix_to_vector -- first 256 values only, folded into 64 buckets,
squashed by tanh -- so two different 1152-value embeddings, and a 1,000 mm3 and
a 30,000 mm3 lesion, all scored cosine 1.000. Each test below reproduces one
face of that defect with the numbers that exposed it.
"""

import math
import random

import numpy as np
import pytest

from melampo.memory.visual_imprint import (
    VECTOR_KIND_NUMERIC,
    VECTOR_KIND_SIGNATURE,
    VisualImprintMorpher,
    VisualRecognitionImprint,
    _cosine,
)


def _true_cosine(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def _imprint(vector, concept="nodulo", **extra):
    return VisualRecognitionImprint.from_payload({"semantic_concept": concept, "vector": vector, **extra})


def _embedding_pair():
    rng = random.Random(0)
    a = [rng.gauss(0, 1) for _ in range(1152)]
    b = a[:256] + [rng.gauss(0, 1) for _ in range(1152 - 256)]
    return a, b


def test_an_embedding_is_kept_whole_and_exactly():
    a, _ = _embedding_pair()
    imprint = _imprint(a)
    assert imprint.vector_kind == VECTOR_KIND_NUMERIC
    assert len(imprint.vector) == 1152
    norm = math.sqrt(sum(x * x for x in a))
    assert imprint.vector[1000] == round(a[1000] / norm, 6)


def test_embeddings_differing_after_position_256_are_no_longer_identical():
    a, b = _embedding_pair()
    similarity = _cosine(_imprint(a).vector, _imprint(b).vector)
    assert similarity == pytest.approx(_true_cosine(a, b), abs=1e-4)
    assert similarity < 0.5  # it was 1.000


def test_magnitudes_are_not_saturated_but_cosine_still_cannot_compare_sizes():
    # tanh used to turn 1,000 and 30,000 into the same 1.0: the vectors were identical.
    small = _imprint([1000.0, 0.9])
    large = _imprint([30000.0, 0.9])
    assert small.vector != large.vector
    assert small.vector == pytest.approx([1000.0 / math.hypot(1000.0, 0.9), 0.9 / math.hypot(1000.0, 0.9)], abs=1e-6)
    # Known limit, pinned so nobody mistakes the fix for more than it is: cosine
    # ignores scale by definition, so raw measurement vectors (size, shape) must
    # be compared with standardised features and a distance, not through imprints.
    assert _cosine(small.vector, large.vector) > 0.9999


def test_numpy_arrays_are_numeric_embeddings():
    imprint = _imprint(np.arange(1, 9, dtype=np.float32))
    assert imprint.vector_kind == VECTOR_KIND_NUMERIC and len(imprint.vector) == 8


def test_structured_payloads_are_marked_as_signatures():
    imprint = _imprint({"volume_mm3": 1000.0, "sphericity": 0.9})
    assert imprint.vector_kind == VECTOR_KIND_SIGNATURE and len(imprint.vector) == 64
    assert _imprint(["a", "b"]).vector_kind == VECTOR_KIND_SIGNATURE
    assert _imprint([1.0, float("nan")]).vector_kind == VECTOR_KIND_SIGNATURE


def test_a_round_trip_through_as_dict_changes_nothing():
    for original in (_imprint(_embedding_pair()[0]), _imprint({"x": 1.0})):
        again = VisualRecognitionImprint.from_payload(original.as_dict())
        assert (again.vector_kind, again.vector, again.imprint_id) == (original.vector_kind, original.vector, original.imprint_id)


def test_vectors_of_different_length_are_not_comparable():
    # They used to be truncated to the shorter length and compared anyway.
    assert _cosine([1.0, 0.0, 0.0], [1.0, 0.0]) == 0.0


def test_the_morpher_never_pairs_different_kinds_or_dimensions():
    a, b = _embedding_pair()
    payloads = [
        _imprint(a, variant_label="a").as_dict(),
        _imprint(b, variant_label="b").as_dict(),
        _imprint({"signal": 1.0}, variant_label="sig").as_dict(),
        _imprint([1.0, 2.0, 3.0], variant_label="short").as_dict(),
    ]
    result = VisualImprintMorpher(min_similarity=0.0).nexus_morph(concept_imprints=payloads)
    assert result["evaluated_pair_count"] == 6
    assert result["incomparable_pair_count"] == 5  # only the two 1152-value embeddings form a pair
    assert result["morph_count"] == 1
    [morph] = result["visual_morph_candidates"]
    assert morph["vector_kind"] == VECTOR_KIND_NUMERIC and len(morph["vector"]) == 1152
    assert result["governance"]["incomparable_vectors_never_mixed"] is True


def test_the_morpher_ignores_diagnostic_targets_it_cannot_compare():
    left = _imprint([0.9, 0.1, 0.0, 0.0], concept="opacity", variant_label="l").as_dict()
    right = _imprint([0.7, 0.3, 0.0, 0.0], concept="opacity", variant_label="r").as_dict()
    signature_target = _imprint({"opacity": 1.0}, concept="opacity", variant_label="t").as_dict()
    result = VisualImprintMorpher(min_similarity=0.0).nexus_morph(concept_imprints=[left, right], diagnostic_imprints=[signature_target])
    [morph] = result["visual_morph_candidates"]
    assert morph["target_imprint_id"] == "none" and morph["target_similarity"] == 0.0


# ---------------------------------------------------------------------------
# Found by review of the first fix (same day)
# ---------------------------------------------------------------------------


def test_numpy_scalar_elements_and_a_batch_dimension_are_still_embeddings():
    # list(np.ones(300, np.float32)) holds numpy scalars, not Python floats:
    # rejecting them sent the embedding down the lossy signature path.
    assert _imprint(list(np.ones(300, np.float32))).vector_kind == VECTOR_KIND_NUMERIC
    assert len(_imprint(list(np.arange(300))).vector) == 300
    batched = _imprint(np.ones((1, 300), np.float32))
    assert batched.vector_kind == VECTOR_KIND_NUMERIC and len(batched.vector) == 300
    assert _imprint([True, False, True]).vector_kind == VECTOR_KIND_SIGNATURE


@pytest.mark.parametrize("dimension", [3, 8, 300])
def test_round_trips_are_exact_at_every_dimension(dimension):
    # Re-normalising an already rounded vector moved its 6th decimal on ~1% of reads.
    rng = random.Random(dimension)
    for _ in range(300):
        original = _imprint([rng.gauss(0, 1) for _ in range(dimension)])
        stored = original.as_dict()
        again = VisualRecognitionImprint.from_payload(stored).as_dict()
        assert again["vector"] == stored["vector"]
        assert again["matrix_signature_hash"] == stored["matrix_signature_hash"]


def test_the_vector_key_is_chosen_as_the_old_or_chain_chose_it():
    # A falsy scalar in "vector" is skipped in favour of "embedding", as before.
    chosen = VisualRecognitionImprint.from_payload({"semantic_concept": "x", "vector": 0.0, "embedding": [3.0, 4.0]})
    assert chosen.vector == [0.6, 0.8]
    # A 0-d numpy array is a scalar: no crash.
    VisualRecognitionImprint.from_payload({"semantic_concept": "x", "vector": np.array(3.0)})
    # An empty list is skipped too.
    assert VisualRecognitionImprint.from_payload({"semantic_concept": "x", "vector": [], "embedding": [1.0, 0.0]}).vector == [1.0, 0.0]


def test_skipped_pairs_are_reported_as_a_warning_not_only_a_counter():
    signature = _imprint({"signal": 1.0}, concept="opacity", variant_label="s").as_dict()
    embedding = _imprint([0.9, 0.1], concept="opacity", variant_label="e").as_dict()
    result = VisualImprintMorpher(min_similarity=0.0).nexus_morph(concept_imprints=[signature, embedding])
    assert result["morph_count"] == 0
    assert result["warnings"] == ["incomparable_vectors_skipped:1 pairs of different vector_kind or dimension"]


def test_weaviate_keeps_embeddings_and_signatures_apart_and_one_embedding_dimension():
    from melampo.memory.weaviate_adapter import WeaviateEnterpriseMemoryAdapter

    adapter = WeaviateEnterpriseMemoryAdapter()
    stored = adapter.upsert_visual_imprint(_imprint([0.9, 0.1, 0.0], variant_label="a").as_dict())
    assert stored["status"] == "completed"
    signature = adapter.upsert_visual_imprint(_imprint({"signal": 1.0}, variant_label="b").as_dict())
    assert signature["status"] == "completed"
    vectors_by_kind = {
        record["properties"]["vector_kind"]: set(record["vectors"])
        for record in adapter.object_graph.values()
        if record["class_name"] == "VisualRecognitionImprint"
    }
    assert "numeric_embedding_vector" in vectors_by_kind[VECTOR_KIND_NUMERIC]
    assert "recognition_matrix_vector" in vectors_by_kind[VECTOR_KIND_SIGNATURE]
    # Weaviate fixes a named vector's dimension at the first insert.
    refused = adapter.upsert_visual_imprint(_imprint([0.5, 0.5, 0.5, 0.5], variant_label="c").as_dict())
    assert refused["status"] == "rejected_embedding_dimension_mismatch"
    assert (refused["expected_dimension"], refused["received_dimension"]) == (3, 4)
