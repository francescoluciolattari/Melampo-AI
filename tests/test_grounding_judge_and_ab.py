from melampo.evaluation.grounding_judge import (
    VERDICT_GROUNDED,
    VERDICT_OVERREACH,
    GroundingJudge,
)


def test_claim_restating_a_single_fragment_is_grounded():
    judge = GroundingJudge()
    fragments = [
        {
            "text": "Chest radiograph shows bibasilar opacities consistent with pulmonary oedema."
        }
    ]
    assessment = judge.assess("Chest radiograph shows bibasilar opacities.", fragments)
    assert assessment.verdict == VERDICT_GROUNDED
    assert assessment.unsupported_terms == []


def test_relation_across_two_fragments_is_flagged_although_every_term_is_supported():
    """The failure term-overlap faithfulness cannot see.

    Both entities appear in the evidence, so term overlap is complete and every
    citation is real. The relation between them appears in no fragment.
    """
    judge = GroundingJudge()
    fragments = [
        {"text": "Chest radiograph shows bibasilar opacities."},
        {"text": "The patient has a documented history of congestive cardiac failure."},
    ]
    claim = "The bibasilar opacities are caused by the congestive cardiac failure."

    assessment = judge.assess(claim, fragments)

    assert "caused by" in assessment.unsupported_relations
    assert assessment.verdict != VERDICT_GROUNDED
    assert any("never co-occur" in note for note in assessment.notes)


def test_relation_present_within_one_fragment_is_not_flagged():
    judge = GroundingJudge()
    fragments = [
        {
            "text": "Bibasilar opacities caused by congestive cardiac failure were documented on admission."
        },
    ]
    assessment = judge.assess(
        "The bibasilar opacities are caused by congestive cardiac failure.", fragments
    )
    assert assessment.unsupported_relations == []


def test_fabricated_entities_are_reported_as_unsupported_terms():
    judge = GroundingJudge()
    fragments = [{"text": "Chest radiograph shows bibasilar opacities."}]
    assessment = judge.assess(
        "Chest radiograph shows bibasilar opacities and splenomegaly.", fragments
    )
    assert "splenomegaly" in assessment.unsupported_terms


def test_asserting_what_the_source_hedges_is_flagged():
    judge = GroundingJudge()
    fragments = [
        {
            "text": "The appearance may represent an early consolidation; infection cannot be excluded."
        }
    ]
    assessment = judge.assess(
        "The appearance is diagnostic of an early consolidation.", fragments
    )
    assert assessment.modality_escalations
    assert any("hedges" in note for note in assessment.notes)


def test_hedged_claim_from_hedged_source_is_not_flagged():
    judge = GroundingJudge()
    fragments = [{"text": "The appearance may represent an early consolidation."}]
    assessment = judge.assess(
        "The appearance may represent an early consolidation.", fragments
    )
    assert assessment.modality_escalations == []


def test_claim_without_any_cited_fragment_is_maximal_overreach():
    judge = GroundingJudge()
    assessment = judge.assess("The patient has pulmonary oedema.", [])
    assert assessment.verdict == VERDICT_OVERREACH
    assert assessment.overreach_score == 1.0


