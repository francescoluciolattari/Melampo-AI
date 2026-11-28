from melampo.training.counterfactual_sampler import CounterfactualSampler
from melampo.training.nexus_trainer import NexusTrainer
from melampo.training.replay_filter import ReplayFilter


def test_nexus_trainer_runs():
    trainer = NexusTrainer(
        replay_filter=ReplayFilter(min_coherence=0.7, max_risk=0.3),
        sampler=CounterfactualSampler(),
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
    assert "belief" not in result
    # closed (28 November 2026): every case held until the lateral cycle has measured criteria
    assert result["auto_evolution_plan"] == {"status": "hold_for_more_evidence",
                                             "reason": "no_measured_promotion_criteria"}
    assert "filter_assessment" in result
    assert result["filter_assessment"]["replay_mode"] in [
        "stabilizing_replay",
        "boundary_replay",
        "corrective_replay",
    ]
    assert "sampled" in result
    assert result["sampled"]["variant_focus"] in [
        "context",
        "language_listening",
        "epidemiology",
        "cross_area_alignment",
    ]
    assert "rehearsal_profile" in result
    assert result["rehearsal_profile"]["revision_bias"] in [
        "exploratory",
        "conservative",
    ]
    assert result["rehearsal_profile"]["post_error_adjustment"] in [
        "re-rank_alternatives",
        "stabilize_primary",
    ]
    assert len(result["alternative_hypotheses"]) >= 2


def test_the_case_discordances_turn_rehearsal_conservative_and_add_a_resolution():
    trainer = NexusTrainer(replay_filter=ReplayFilter(), sampler=CounterfactualSampler())
    calm = trainer.run(case_context={"case_id": "x", "report_text": "t"}, coherence=0.9, risk=0.1)
    tense = trainer.run(
        case_context={"case_id": "x", "report_text": "t", "discordances": {"count": 2}},
        coherence=0.9,
        risk=0.1,
    )
    assert calm["rehearsal_profile"]["revision_bias"] == "exploratory"
    assert tense["rehearsal_profile"]["revision_bias"] == "conservative"
    assert tense["rehearsal_profile"]["contradiction_rehearsal"] is True
    assert tense["sampled"]["variant_focus"] == "cross_area_alignment"
    assert any(h["kind"] == "discordance_resolution" for h in tense["alternative_hypotheses"])
    assert not any(h["kind"] == "discordance_resolution" for h in calm["alternative_hypotheses"])
