from melampo.models.quantum_belief_layer import QuantumBeliefLayer
from melampo.training.counterfactual_sampler import CounterfactualSampler
from melampo.training.dream_trainer import DreamTrainer
from melampo.training.replay_filter import ReplayFilter


def test_dream_trainer_runs():
    trainer = DreamTrainer(
        replay_filter=ReplayFilter(min_coherence=0.7, max_risk=0.3),
        sampler=CounterfactualSampler(),
        belief_layer=QuantumBeliefLayer(),
    )
    result = trainer.run(
        case_context={
            "case_id": "x",
            "bundle_keys": ["a", "b"],
            "report_text": "possible lesion",
            "patient_complaints": "cough",
        },
        coherence=0.9,
        risk=0.1,
    )
    assert result["accepted"] is True
    assert result["belief"]["mode"] == "quantum_like_belief_update"
    assert "filter_assessment" in result
    assert result["filter_assessment"]["replay_mode"] in ["stabilizing_replay", "boundary_replay", "corrective_replay"]
    assert "sampled" in result
    assert result["sampled"]["variant_focus"] in ["context", "language_listening", "epidemiology", "cross_area_alignment"]
    assert "rehearsal_profile" in result
    assert result["rehearsal_profile"]["revision_bias"] in ["exploratory", "conservative"]
    assert result["rehearsal_profile"]["post_error_adjustment"] in ["re-rank_alternatives", "stabilize_primary"]
    assert len(result["alternative_hypotheses"]) >= 2
