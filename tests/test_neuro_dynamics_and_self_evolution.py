from melampo.memory.semantic_memory import SemanticMemoryStore
from melampo.reasoning.area_coherence import AreaCoherenceAnalyzer
from melampo.reasoning.neuro_dynamics import NeuroDynamicMetrics
from melampo.training.self_evolution import NexusSelfEvolutionLoop


def test_neuro_dynamic_metrics_are_emitted_by_area_coherence():
    area_signals = {
        "visual_diagnostic": {"salience_score": 0.8, "signal_count": 3},
        "language_listening": {"salience_score": 0.6, "signal_count": 2},
        "case_context": {"salience_score": 0.3, "signal_count": 2},
        "epidemiology": {"salience_score": 0.4, "signal_count": 2},
    }
    dynamics = AreaCoherenceAnalyzer().analyze(area_signals)
    neuro = dynamics["neuro_dynamic_metrics"]
    assert 0.0 <= neuro["pi_score"] <= 1.0
    assert 0.0 <= neuro["prediction_error"] <= 1.0
    assert 0.0 <= neuro["precision_weighted_coherence"] <= 1.0
    assert dynamics["pi_score"] == neuro["pi_score"]


def test_neuro_dynamic_metrics_direct_compute_contract():
    metrics = NeuroDynamicMetrics().compute(
        pair_profiles=[
            {"status": "coherent", "pair_salience": 1.0, "pair_signal_count": 4},
            {"status": "mismatch", "pair_salience": 0.2, "pair_signal_count": 1},
        ],
        coherence_score=0.8,
        mismatch_score=0.2,
        total_salience=1.2,
    )
    assert metrics["pi_score"] > 0.0
    assert metrics["deductive_gate"] >= 0.0
    assert metrics["interpretation"] == "computational_abstraction_not_literal_neurobiology"


def test_semantic_memory_indexes_documents_in_vector_memory():
    memory = SemanticMemoryStore()
    memory.add_document({
        "id": "doc-1",
        "text": "pulmonary opacity cough fever differential pneumonia",
        "source": "test_fixture",
        "learning_status": "promoted",
    })
    hits = memory.semantic_search("cough pneumonia", limit=1, promoted_only=True)
    assert hits
    assert hits[0]["record_id"] == "doc-1"
    assert memory.describe()["vector_store"]["record_count"] == 1


def test_the_real_governed_chain_holds_a_favorable_candidate_at_needs_review_by_default():
    """Replaces a test that exercised rehearse(), removed together with
    evaluate_candidate() -- neither was ever called by anything except each
    other and this test; nexus_scheduler.py._execute_job() (the real,
    live-connected flow) only ever calls generate_candidate(), then routes
    through NexusCandidateStore, RationalControlValidator and
    PromotionPolicy. This exercises that real chain directly, on a
    genuinely favorable candidate, and confirms this project's actual
    default settings (PromotionPolicy.allow_automatic_promotion=False,
    require_human_review_for_promoted=True) hold it at needs_review rather
    than auto-promoting -- the property the whole redesign discussion
    depended on being true, not assumed.
    """
    from melampo.training.nexus_candidate_store import NexusCandidateStore
    from melampo.training.promotion_policy import PromotionPolicy
    from melampo.training.rational_control_validator import RationalControlValidator

    loop = NexusSelfEvolutionLoop()
    area_dynamics = _area_dynamics(pi_score=0.9, convergence_index=0.9, nexus_plasticity=0.9)
    candidate = loop.generate_candidate(
        case_context={"case_id": "case-good", "report_text": "cough opacity", "patient_complaints": "fever"},
        area_dynamics=area_dynamics,
        nexus={"visual_morphing": {"visual_morph_intuition_gain": 0.9}},
        governance_scores={"risk": 0.0},
    )

    store = NexusCandidateStore()
    record = store.create_candidate(payload=candidate, case_id="case-good", source="test", learning_status="candidate")
    validation = RationalControlValidator().evaluate(
        candidate=candidate, area_dynamics=area_dynamics, retrieval_context={}, governance_scores={"risk": 0.0},
    )
    decision = PromotionPolicy().decide(candidate=store.attach_validation(record.candidate_id, validation), validation=validation)

    assert candidate["metadata"]["candidate_score"] > 0.5
    assert decision["target_learning_status"] == "needs_review"
    assert decision["policy"]["allow_automatic_promotion"] is False


# --------------------------------------------------------------------------
# candidate_score: consolidated here from NexusTrainer's former
# _auto_evolution_plan(), the only field promotion_policy.decide() and
# rational_control_validator.py actually read from it -- computed fresh
# from area_dynamics, nexus (visual_morphing), and the newly added
# governance_scores parameter.
# --------------------------------------------------------------------------


def _area_dynamics(pi_score=0.6, convergence_index=0.5, nexus_plasticity=0.4):
    return {
        "coherence_pairs": [],
        "mismatch_pairs": [],
        "neuro_dynamic_metrics": {
            "pi_score": pi_score,
            "convergence_index": convergence_index,
            "nexus_plasticity": nexus_plasticity,
            "prediction_error": 0.1,
            "bias_suppression_score": 0.8,
        },
    }


def test_candidate_score_is_computed_from_area_dynamics_and_nexus():
    loop = NexusSelfEvolutionLoop()
    candidate = loop.generate_candidate(
        case_context={"case_id": "c1"},
        area_dynamics=_area_dynamics(pi_score=0.8, convergence_index=0.7, nexus_plasticity=0.6),
        nexus={"visual_morphing": {"visual_morph_intuition_gain": 0.5}},
        governance_scores={"risk": 0.1},
    )
    expected = round(0.8 * 0.32 + 0.7 * 0.27 + 0.6 * 0.18 + 0.5 * 0.08 - 0.1 * 0.15, 3)
    assert candidate["metadata"]["candidate_score"] == expected


def test_candidate_score_defaults_to_zero_inputs_without_governance_scores():
    loop = NexusSelfEvolutionLoop()
    candidate = loop.generate_candidate(
        case_context={"case_id": "c1"}, area_dynamics={"neuro_dynamic_metrics": {}}, nexus={},
    )
    assert candidate["metadata"]["candidate_score"] == 0.0


def test_a_higher_risk_lowers_the_candidate_score():
    loop = NexusSelfEvolutionLoop()
    low_risk = loop.generate_candidate(
        case_context={"case_id": "c1"}, area_dynamics=_area_dynamics(), nexus={}, governance_scores={"risk": 0.1},
    )
    high_risk = loop.generate_candidate(
        case_context={"case_id": "c1"}, area_dynamics=_area_dynamics(), nexus={}, governance_scores={"risk": 0.9},
    )
    assert high_risk["metadata"]["candidate_score"] < low_risk["metadata"]["candidate_score"]


def test_candidate_score_flows_through_to_promotion_policy_via_metadata():
    """The end-to-end property this whole consolidation depends on:
    promotion_policy.decide() and rational_control_validator.py read
    candidate_score via a fallback chain (auto_plan -> metadata -> default)
    that was never changed -- only where the value now originates."""
    from melampo.training.promotion_policy import PromotionPolicy

    loop = NexusSelfEvolutionLoop()
    candidate = loop.generate_candidate(
        case_context={"case_id": "c1"},
        area_dynamics=_area_dynamics(pi_score=0.9, convergence_index=0.9, nexus_plasticity=0.9),
        nexus={"visual_morphing": {"visual_morph_intuition_gain": 0.9}},
        governance_scores={"risk": 0.0},
    )
    policy = PromotionPolicy()
    decision = policy.decide(candidate={"metadata": candidate["metadata"]}, validation={"allowed_for_promotion": True})

    assert candidate["metadata"]["candidate_score"] > 0.0
    assert "candidate_score_below_promotion_policy_threshold" not in decision.get("reasons", [])
