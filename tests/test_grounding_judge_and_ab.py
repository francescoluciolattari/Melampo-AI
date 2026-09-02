from melampo.evaluation.dual_path_ab import (
    CLAIM_ID,
    OUTCOME_CORROBORATED,
    OUTCOME_INCONCLUSIVE,
    OUTCOME_REFUTED,
    DualPathComparison,
    PairedCaseResult,
)
from melampo.evaluation.falsification_program import CLAIM_OPEN, FalsificationProgram
from melampo.evaluation.grounding_judge import (
    VERDICT_GROUNDED,
    VERDICT_OVERREACH,
    GroundingJudge,
)


def test_claim_restating_a_single_fragment_is_grounded():
    judge = GroundingJudge()
    fragments = [{"text": "Chest radiograph shows bibasilar opacities consistent with pulmonary oedema."}]
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
        {"text": "Bibasilar opacities caused by congestive cardiac failure were documented on admission."},
    ]
    assessment = judge.assess("The bibasilar opacities are caused by congestive cardiac failure.", fragments)
    assert assessment.unsupported_relations == []


def test_fabricated_entities_are_reported_as_unsupported_terms():
    judge = GroundingJudge()
    fragments = [{"text": "Chest radiograph shows bibasilar opacities."}]
    assessment = judge.assess("Chest radiograph shows bibasilar opacities and splenomegaly.", fragments)
    assert "splenomegaly" in assessment.unsupported_terms


def test_asserting_what_the_source_hedges_is_flagged():
    judge = GroundingJudge()
    fragments = [{"text": "The appearance may represent an early consolidation; infection cannot be excluded."}]
    assessment = judge.assess("The appearance is diagnostic of an early consolidation.", fragments)
    assert assessment.modality_escalations
    assert any("hedges" in note for note in assessment.notes)


def test_hedged_claim_from_hedged_source_is_not_flagged():
    judge = GroundingJudge()
    fragments = [{"text": "The appearance may represent an early consolidation."}]
    assessment = judge.assess("The appearance may represent an early consolidation.", fragments)
    assert assessment.modality_escalations == []


def test_claim_without_any_cited_fragment_is_maximal_overreach():
    judge = GroundingJudge()
    assessment = judge.assess("The patient has pulmonary oedema.", [])
    assert assessment.verdict == VERDICT_OVERREACH
    assert assessment.overreach_score == 1.0


def _pairs(faithfulness_delta: float, recall_delta: float, count: int = 30) -> list[PairedCaseResult]:
    return [
        PairedCaseResult(
            case_id=f"case_{index}",
            one_shot_faithfulness=0.70,
            dual_path_faithfulness=0.70 + faithfulness_delta,
            one_shot_recall=0.60,
            dual_path_recall=0.60 + recall_delta,
        )
        for index in range(count)
    ]


def test_faithfulness_regression_refutes_the_claim():
    comparison = DualPathComparison()
    report = comparison.evaluate(_pairs(faithfulness_delta=-0.10, recall_delta=0.10))
    assert report.outcome == OUTCOME_REFUTED
    assert any("below the one-shot baseline" in reason for reason in report.reasons)


def test_recall_gain_indistinguishable_from_zero_refutes_the_claim():
    comparison = DualPathComparison()
    report = comparison.evaluate(_pairs(faithfulness_delta=0.02, recall_delta=0.0))
    assert report.outcome == OUTCOME_REFUTED
    assert any("not distinguishable" in reason for reason in report.reasons)


def test_positive_recall_without_faithfulness_regression_corroborates():
    comparison = DualPathComparison()
    report = comparison.evaluate(_pairs(faithfulness_delta=0.03, recall_delta=0.12))
    assert report.outcome == OUTCOME_CORROBORATED
    assert report.recall_interval[0] > 0


def test_too_few_cases_is_inconclusive_rather_than_a_verdict():
    comparison = DualPathComparison(min_cases=20)
    report = comparison.evaluate(_pairs(faithfulness_delta=0.05, recall_delta=0.20, count=5))
    assert report.outcome == OUTCOME_INCONCLUSIVE
    assert report.case_count == 5


def test_bootstrap_is_deterministic_across_runs():
    pairs = _pairs(faithfulness_delta=0.03, recall_delta=0.09)
    first = DualPathComparison().evaluate(pairs)
    second = DualPathComparison().evaluate(pairs)
    assert first.recall_interval == second.recall_interval
    assert first.faithfulness_interval == second.faithfulness_interval


def test_inconclusive_run_leaves_the_claim_open():
    program = FalsificationProgram()
    comparison = DualPathComparison()
    report = comparison.evaluate(_pairs(faithfulness_delta=0.05, recall_delta=0.20, count=5))

    assert comparison.resolve_claim(report, program, run_id="run_1") is None
    assert program.get(CLAIM_ID).status == CLAIM_OPEN
    assert CLAIM_ID in {claim.claim_id for claim in program.blocking_claims()}


def test_refuted_run_records_evidence_against_the_claim():
    program = FalsificationProgram()
    comparison = DualPathComparison()
    report = comparison.evaluate(_pairs(faithfulness_delta=-0.10, recall_delta=0.10))

    comparison.resolve_claim(report, program, run_id="run_2")
    claim = program.get(CLAIM_ID)
    assert claim.status == "refuted"
    assert claim.evidence and claim.evidence[0].startswith("run_2:")
