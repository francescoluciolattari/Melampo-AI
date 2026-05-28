from melampo.memory.visual_imprint import VisualImprintBuilder, VisualImprintMorpher, VisualRecognitionImprint
from melampo.memory.weaviate_adapter import WeaviateEnterpriseMemoryAdapter
from melampo.memory.weaviate_schema import MelampoWeaviateSchema
from melampo.models.quantum_belief_layer import QuantumBeliefLayer
from melampo.training.counterfactual_sampler import CounterfactualSampler
from melampo.training.dream_trainer import DreamTrainer
from melampo.training.replay_filter import ReplayFilter


def _imprint(concept: str, label: str, vector: list[float], ontology_refs: list[str] | None = None) -> dict:
    return VisualRecognitionImprint.from_payload(
        {
            "semantic_concept": concept,
            "variant_label": label,
            "source_object_id": f"study-{label}",
            "vector": vector,
            "salience": 0.8,
            "uncertainty": 0.2,
            "ontology_refs": ontology_refs or [],
            "provenance": {"source": "unit_test", "license": "synthetic"},
        }
    ).as_dict()


def test_visual_imprint_builder_creates_governed_semantic_footprints():
    imprints = VisualImprintBuilder().from_visual_area(
        signal={
            "claims": [{"normalized_entity": "ground glass opacity"}],
            "salience_score": 0.7,
            "uncertainty_score": 0.3,
        },
        volume_features={"study_id": "study-1", "input_kind": "ct", "metadata": {"modality": "CT"}},
    )

    assert imprints[0]["semantic_concept"] == "ground glass opacity"
    assert imprints[0]["matrix_signature_hash"]
    assert imprints[0]["provenance"]["hidden_network_call"] is False


def test_visual_imprint_morpher_supports_partial_semantic_concept_overlap():
    memory_variants = [
        _imprint("ground glass opacity", "variant-a", [0.9, 0.1, 0.0, 0.0]),
        _imprint("peripheral opacity pattern", "variant-b", [0.65, 0.35, 0.0, 0.0]),
    ]
    diagnostic = [_imprint("opacity", "diagnostic", [0.78, 0.22, 0.0, 0.0])]

    result = VisualImprintMorpher(min_similarity=0.1, min_semantic_overlap=0.2).dream_morph(
        concept_imprints=memory_variants,
        diagnostic_imprints=diagnostic,
        area_dynamics={
            "neuro_dynamic_metrics": {
                "pi_score": 0.72,
                "prediction_error": 0.12,
                "mismatch_index": 0.15,
                "dream_plasticity": 0.55,
                "action_potential_gate": 0.65,
            }
        },
    )

    candidate = result["visual_morph_candidates"][0]
    assert candidate["morphing_mode"] == "inferential_semantic_matrix_morphing"
    assert candidate["semantic_match_type"] == "partial_semantic_concept"
    assert "opacity" in candidate["shared_semantic_terms"]
    assert candidate["semantic_relation_score"] > 0.0
    assert result["governance"]["morphs_total_or_partial_semantic_concepts"] is True
    assert result["neuroquantum_trace"]["supports_total_or_partial_semantic_concepts"] is True


def test_visual_imprint_morpher_links_same_semantic_concept_to_diagnostic_imprint():
    memory_variants = [
        _imprint("ground glass opacity", "variant-a", [0.9, 0.1, 0.0, 0.0]),
        _imprint("ground glass opacity", "variant-b", [0.7, 0.3, 0.0, 0.0]),
    ]
    diagnostic = [_imprint("ground glass opacity", "diagnostic", [0.8, 0.2, 0.0, 0.0])]

    result = VisualImprintMorpher(min_similarity=0.1).dream_morph(
        concept_imprints=memory_variants,
        diagnostic_imprints=diagnostic,
        area_dynamics={
            "neuro_dynamic_metrics": {
                "pi_score": 0.75,
                "prediction_error": 0.1,
                "mismatch_index": 0.1,
                "dream_plasticity": 0.6,
                "action_potential_gate": 0.7,
            }
        },
    )

    assert result["morph_count"] == 1
    assert result["semantic_links"]
    assert result["visual_prediction_link_score"] > 0.0
    assert result["governance"]["does_not_generate_clinical_images"] is True
    assert result["neuroquantum_trace"]["formalism"] == "quantum_like_latent_interference_not_physical_quantum_claim"


def test_dream_trainer_uses_visual_morphing_in_rehearsal_and_belief_update():
    memory_variants = [
        _imprint("opacity", "variant-a", [0.9, 0.1, 0.0, 0.0]),
        _imprint("opacity", "variant-b", [0.6, 0.4, 0.0, 0.0]),
    ]
    diagnostic = [_imprint("opacity", "diagnostic", [0.75, 0.25, 0.0, 0.0])]
    trainer = DreamTrainer(
        replay_filter=ReplayFilter(min_coherence=0.7, max_risk=0.3),
        sampler=CounterfactualSampler(),
        belief_layer=QuantumBeliefLayer(),
    )

    result = trainer.run(
        case_context={
            "case_id": "case-visual",
            "bundle_keys": ["ImagingStudy"],
            "report_text": "opacity",
            "visual_imprints": diagnostic,
            "diagnostic_visual_imprints": diagnostic,
            "concept_memory_imprints": memory_variants,
            "area_dynamics": {
                "mismatch_score": 0.1,
                "coherence_pairs": [("language_listening", "visual_diagnostic")],
                "neuro_dynamic_metrics": {
                    "pi_score": 0.78,
                    "convergence_index": 0.7,
                    "revision_pressure": 0.2,
                    "dream_plasticity": 0.6,
                    "prediction_error": 0.1,
                    "mismatch_index": 0.1,
                    "action_potential_gate": 0.7,
                },
            },
        },
        coherence=0.9,
        risk=0.1,
    )

    assert result["visual_morphing"]["morph_count"] >= 1
    assert result["rehearsal_profile"]["visual_morphing_active"] is True
    assert result["belief"]["visual_morph_intuition_gain"] > 0.0
    assert any(item["kind"] == "visual_semantic_morph_correlation" for item in result["alternative_hypotheses"])


def test_weaviate_schema_and_adapter_store_visual_imprint_contract_locally():
    schema = MelampoWeaviateSchema().as_dict()
    class_names = {class_schema["class"] for class_schema in schema["classes"]}
    assert {"VisualConcept", "VisualRecognitionImprint"}.issubset(class_names)

    adapter = WeaviateEnterpriseMemoryAdapter()
    result = adapter.upsert_visual_imprint(_imprint("opacity", "variant-a", [0.9, 0.1, 0.0, 0.0]))

    assert result["status"] == "completed"
    assert result["governance"]["semantic_object_imprint_association"] is True
    assert result["imprint_result"]["hidden_network_call"] is False
    assert "VisualRecognitionImprint" in result["imprint_result"]["object_key"]
