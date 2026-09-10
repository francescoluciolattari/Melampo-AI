"""Tests for running two root models over the same case and comparing them."""

from melampo.memory.context_environment import EnvironmentDocument
from melampo.reasoning.root_model_cross_check import (
    ALTERNATIVE_PAIR,
    ANSWER_AGREEMENT_THRESHOLD,
    DEFAULT_PAIR,
    CrossCheckReport,
    CrossCheckResult,
    answer_similarity,
    cross_check,
    cross_check_cases,
    normalise_answer,
)

DOC = EnvironmentDocument(
    "report_1", "Prednisone 40 mg daily was started.", metadata={"data_class": "synthetic"}
)


def _scripted(*outputs: str):
    queue = iter(outputs)
    return lambda prompt: next(queue, "final(fallback)")


# --------------------------------------------------------------------------
# Answer comparison: the part that decides whether two models agree
# --------------------------------------------------------------------------


def test_identical_answers_agree():
    assert answer_similarity("40 mg daily", "40 mg daily") == 1.0


def test_the_same_answer_stated_more_verbosely_agrees():
    """Containment, not sequence ratio: "40 mg daily" inside "prednisone 40 mg
    daily" is the same answer, one more verbose, and scores only 0.67 on
    sequence ratio alone -- below the threshold, wrongly flagged."""
    assert answer_similarity("40 mg daily", "prednisone 40 mg daily") == 1.0
    assert answer_similarity("pulmonary embolism", "confirmed pulmonary embolism") == 1.0


def test_clinically_opposite_findings_do_not_agree():
    """The case that proves sequence ratio alone is unusable here: these two
    findings are opposites (one advanced bench case exists to distinguish
    them) and score 0.71 on raw ratio -- HIGHER than a correct paraphrase."""
    similarity = answer_similarity("pulmonary embolism", "pulmonary oedema")
    assert similarity < ANSWER_AGREEMENT_THRESHOLD


def test_genuinely_different_answers_do_not_agree():
    assert answer_similarity("40 mg daily", "10 mg twice weekly") < ANSWER_AGREEMENT_THRESHOLD


def test_two_empty_answers_score_zero_not_one():
    """Two models that said nothing have not agreed on anything -- scoring
    this 1.0 would make the worst outcome the best-looking number."""
    assert answer_similarity(None, None) == 0.0
    assert answer_similarity("", "") == 0.0


def test_one_empty_answer_scores_zero():
    assert answer_similarity("40 mg daily", None) == 0.0


def test_normalisation_ignores_case_whitespace_and_trailing_punctuation():
    assert normalise_answer("  40 MG   Daily.  ") == "40 mg daily"


# --------------------------------------------------------------------------
# The four dispositions, kept distinct because they need different responses
# --------------------------------------------------------------------------


def test_both_completing_with_the_same_answer_is_agreement():
    result = cross_check(
        "c1", [DOC], "dose?",
        _scripted("grep(Prednisone)", "final(40 mg daily)"),
        _scripted("grep(Prednisone)", "final(prednisone 40 mg daily)"),
    )
    assert result.disposition == "agreed"
    assert result.answers_agree is True
    assert result.needs_review is False


def test_both_completing_with_different_answers_is_disagreement():
    result = cross_check(
        "c2", [DOC], "dose?",
        _scripted("grep(Prednisone)", "final(40 mg daily)"),
        _scripted("grep(Prednisone)", "final(10 mg twice weekly)"),
    )
    assert result.disposition == "disagreed"
    assert result.needs_review is True
    assert any("neither is preferred" in note for note in result.notes)


def test_neither_answer_is_silently_preferred_on_disagreement():
    """The whole point: a disagreement records both and picks neither."""
    result = cross_check(
        "c3", [DOC], "dose?",
        _scripted("grep(Prednisone)", "final(40 mg daily)"),
        _scripted("grep(Prednisone)", "final(10 mg twice weekly)"),
    )
    assert result.primary_answer == "40 mg daily"
    assert result.secondary_answer == "10 mg twice weekly"


def test_only_one_model_completing_is_not_agreement():
    """One answer is not a second opinion -- there is nothing to confirm it."""
    result = cross_check(
        "c4", [DOC], "dose?",
        _scripted("grep(Prednisone)", "final(40 mg daily)"),
        lambda prompt: "I cannot answer this.",
    )
    assert result.disposition == "single_answer_only"
    assert result.answers_agree is False
    assert result.needs_review is True


def test_neither_completing_is_its_own_disposition():
    result = cross_check("c5", [DOC], "dose?", lambda p: "prose", lambda p: "more prose")
    assert result.disposition == "neither_completed"
    assert result.needs_review is True


def test_every_non_agreed_disposition_needs_review():
    """Deliberately inclusive: one model failing is itself a reason not to
    trust the other's answer unexamined."""
    for disposition in ("disagreed", "single_answer_only", "neither_completed"):
        result = CrossCheckResult(case_id="c", primary_model="a", secondary_model="b")
        # Construct each disposition directly rather than through a run.
        if disposition == "disagreed":
            result.primary_completed = result.secondary_completed = True
            result.primary_answer, result.secondary_answer = "x", "totally different"
        elif disposition == "single_answer_only":
            result.primary_completed = True
            result.primary_answer = "x"
        assert result.disposition == disposition
        assert result.needs_review is True


# --------------------------------------------------------------------------
# Evidence overlap: the same answer from different fragments means more
# --------------------------------------------------------------------------


def test_evidence_agreement_ratio_is_reported():
    result = cross_check(
        "c6", [DOC], "dose?",
        _scripted("grep(Prednisone)", "final(40 mg daily)"),
        _scripted("grep(Prednisone)", "final(40 mg daily)"),
    )
    assert 0.0 <= result.evidence_agreement_ratio <= 1.0


def test_agreement_from_different_evidence_is_noted_as_stronger():
    """Two models reaching the same answer from different fragments is
    genuine independent corroboration; from identical fragments it may just
    be two models finding the one obvious passage."""
    result = CrossCheckResult(case_id="c", primary_model="a", secondary_model="b")
    result.primary_completed = result.secondary_completed = True
    result.primary_answer = result.secondary_answer = "same"
    result.answer_similarity = 1.0
    result.shared_evidence_ids = ["x"]
    result.primary_only_evidence_ids = ["y", "z"]
    result.secondary_only_evidence_ids = ["w"]
    assert result.evidence_agreement_ratio < 0.5


def test_evidence_ratio_of_a_result_with_no_evidence_is_zero_not_a_crash():
    result = CrossCheckResult(case_id="c", primary_model="a", secondary_model="b")
    assert result.evidence_agreement_ratio == 0.0


# --------------------------------------------------------------------------
# Independence: neither run may influence the other
# --------------------------------------------------------------------------


def test_each_model_gets_its_own_budget_instance():
    """A shared Budget would let the first model's iteration count constrain
    the second, defeating the independence the comparison depends on."""
    from melampo.reasoning.rlm_engine import Budget

    handed_out = []

    def tracking_factory():
        budget = Budget(max_iterations=5)
        handed_out.append(budget)
        return budget

    cross_check(
        "c7", [DOC], "dose?",
        _scripted("final(a)"), _scripted("final(b)"),
        budget_factory=tracking_factory,
    )
    assert len(handed_out) == 2
    assert handed_out[0] is not handed_out[1]


def test_the_second_model_does_not_see_the_first_models_prompts():
    """Independence is only meaningful if the second model navigates from
    scratch, not from a context the first one shaped."""
    primary_prompts, secondary_prompts = [], []

    def primary(prompt):
        primary_prompts.append(prompt)
        return "final(a)"

    def secondary(prompt):
        secondary_prompts.append(prompt)
        return "final(b)"

    cross_check("c8", [DOC], "dose?", primary, secondary)
    assert secondary_prompts[0] == primary_prompts[0], "both start from the identical initial prompt"


# --------------------------------------------------------------------------
# Named pairings
# --------------------------------------------------------------------------


def test_the_default_pair_is_the_two_most_efficient_benched_candidates():
    assert DEFAULT_PAIR == ("nemotron-3-super", "gemma-3-27b")


def test_the_alternative_pair_swaps_in_mistral():
    assert ALTERNATIVE_PAIR == ("nemotron-3-super", "mistral-large-openrouter")
    assert ALTERNATIVE_PAIR[0] == DEFAULT_PAIR[0], "same primary, only the second opinion changes"


def test_model_names_are_recorded_in_the_result():
    result = cross_check(
        "c9", [DOC], "dose?", _scripted("final(a)"), _scripted("final(b)"),
        primary_name="model-one", secondary_name="model-two",
    )
    assert result.primary_model == "model-one"
    assert result.secondary_model == "model-two"


# --------------------------------------------------------------------------
# Aggregate reporting
# --------------------------------------------------------------------------


def test_cross_check_cases_aggregates_across_a_sequence():
    cases = [
        ("c1", [DOC], "dose?"),
        ("c2", [DOC], "what drug?"),
    ]
    report = cross_check_cases(cases, _scripted("final(same)"), _scripted("final(same)"))
    assert len(report.results) == 2


def test_the_report_separates_agreement_rate_from_review_rate():
    report = CrossCheckReport()
    agreed = CrossCheckResult(case_id="a", primary_model="p", secondary_model="s")
    agreed.primary_completed = agreed.secondary_completed = True
    agreed.answer_similarity = 1.0
    disagreed = CrossCheckResult(case_id="b", primary_model="p", secondary_model="s")
    disagreed.primary_completed = disagreed.secondary_completed = True
    disagreed.answer_similarity = 0.1
    report.results = [agreed, disagreed]

    assert report.agreement_rate == 0.5
    assert report.review_rate == 0.5
    assert report.by_disposition() == {"agreed": 1, "disagreed": 1}


def test_flagged_returns_only_cases_needing_review():
    report = CrossCheckReport()
    agreed = CrossCheckResult(case_id="a", primary_model="p", secondary_model="s")
    agreed.primary_completed = agreed.secondary_completed = True
    agreed.answer_similarity = 1.0
    failed = CrossCheckResult(case_id="b", primary_model="p", secondary_model="s")
    report.results = [agreed, failed]

    assert [item.case_id for item in report.flagged()] == ["b"]


def test_an_empty_report_reports_zero_rather_than_dividing_by_zero():
    report = CrossCheckReport()
    assert report.agreement_rate == 0.0
    assert report.review_rate == 0.0
    assert report.by_disposition() == {}


def test_as_dict_carries_the_fields_a_reviewer_needs():
    result = cross_check(
        "c10", [DOC], "dose?",
        _scripted("grep(Prednisone)", "final(40 mg daily)"),
        _scripted("grep(Prednisone)", "final(10 mg weekly)"),
    )
    payload = result.as_dict()
    for key in (
        "primary_answer", "secondary_answer", "answers_agree", "answer_similarity",
        "disposition", "needs_review", "evidence_agreement_ratio",
    ):
        assert key in payload


# --------------------------------------------------------------------------
# Frame-structured comparison: the fix for what character comparison got wrong
# --------------------------------------------------------------------------


def test_frame_comparison_replaces_character_similarity_when_a_frame_is_given():
    from melampo.reasoning.frame_answer import FRAME_FINDING

    result = cross_check(
        "c-frame", [DOC], "finding?",
        _scripted("final(embolism | pulmonary | affirmed)"),
        _scripted("final(oedema | pulmonary | affirmed)"),
        frame=FRAME_FINDING,
    )
    assert result.frame_comparison is not None
    assert result.disposition == "disagreed"
    assert result.frame_comparison.conflicting_slots == ["finding"]


def test_the_character_path_is_used_when_no_frame_is_given():
    """Optional, not mandatory: a caller that did not ask its models for slot
    format must not have unstructured answers parsed into empty slots."""
    result = cross_check(
        "c-nof", [DOC], "dose?",
        _scripted("final(40 mg daily)"), _scripted("final(40 mg daily)"),
    )
    assert result.frame_comparison is None
    assert result.answers_agree is True


def test_a_polarity_conflict_gets_its_own_explicit_note():
    """Two models asserting opposites about the same finding is categorically
    worse than naming two different findings, and must not read as an
    ordinary disagreement."""
    from melampo.reasoning.frame_answer import FRAME_FINDING

    result = cross_check(
        "c-pol", [DOC], "finding?",
        _scripted("final(embolism | pulmonary | affirmed)"),
        _scripted("final(embolism | pulmonary | negated)"),
        frame=FRAME_FINDING,
    )
    assert result.frame_comparison.polarity_conflict is True
    assert any("POLARITY" in note for note in result.notes)


def test_frame_agreement_survives_verbosity_differences_that_broke_the_ratio():
    from melampo.reasoning.frame_answer import FRAME_MEDICATION

    result = cross_check(
        "c-verb", [DOC], "dose?",
        _scripted("final(prednisone | 40 mg | daily | affirmed)"),
        _scripted("final(prednisone | 40 mg | once daily | affirmed)"),
        frame=FRAME_MEDICATION,
    )
    assert result.disposition == "agreed"


def test_as_dict_carries_the_frame_comparison_when_present():
    from melampo.reasoning.frame_answer import FRAME_FINDING

    result = cross_check(
        "c-dict", [DOC], "finding?",
        _scripted("final(embolism | pulmonary | affirmed)"),
        _scripted("final(embolism | pulmonary | affirmed)"),
        frame=FRAME_FINDING,
    )
    payload = result.as_dict()
    assert payload["frame_comparison"] is not None
    assert "polarity_conflict" in payload["frame_comparison"]


# --------------------------------------------------------------------------
# Mechanism verification wired into the real entry point, not only the
# standalone module: this is the gap a status check found -- the module
# existed and was tested, but cross_check() never called it.
# --------------------------------------------------------------------------


def _marfan_graph():
    from melampo.memory.concept_paths import ConceptEdge, InMemoryConceptGraph

    return InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("marfan syndrome", "causes", "connective tissue weakness", 0.9),
            ConceptEdge("connective tissue weakness", "causes", "aortic root dilation", 0.85),
        ]
    )


def test_without_a_concept_graph_mechanism_check_is_not_populated():
    """The documented fallback: a caller with no graph handy still gets a
    usable, if less precise, comparison rather than an error."""
    from melampo.reasoning.frame_answer import FRAME_RELEVANCE

    result = cross_check(
        "c-nograph", [DOC], "does X bear on Y?",
        _scripted("final(marfan syndrome | aortic root dilation | yes | cosmic ray exposure)"),
        _scripted("final(marfan syndrome | aortic root dilation | yes | cosmic ray exposure)"),
        frame=FRAME_RELEVANCE,
    )
    assert result.mechanism_check is None


def test_two_models_agreeing_on_an_invented_mechanism_is_caught_through_cross_check_itself():
    """The exact gap found in a status audit: mechanism_verification.py
    existed and was tested in isolation, but cross_check() -- the real entry
    point -- never called it, so this case looked like clean agreement right
    up until concept_graph was actually wired through."""
    from melampo.reasoning.frame_answer import FRAME_RELEVANCE

    result = cross_check(
        "c-danger", [DOC], "does X bear on Y?",
        _scripted("final(marfan syndrome | aortic root dilation | yes | cosmic ray exposure)"),
        _scripted("final(marfan syndrome | aortic root dilation | yes | cosmic ray exposure)"),
        frame=FRAME_RELEVANCE,
        concept_graph=_marfan_graph(),
    )
    assert result.mechanism_check is not None
    assert result.mechanism_check.disposition == "agreed_but_ungrounded"
    assert result.needs_review is True, "must not be masked by frame_comparison's own clean agreement"


def test_needs_review_is_a_union_not_an_override():
    """A relevance case can pass the plain slot comparison (same words) while
    the graph finds neither claim grounded -- the union catches this, an
    override keyed only on mechanism_check would not distinguish it from a
    case where the slot comparison itself already disagreed."""
    from melampo.reasoning.frame_answer import FRAME_RELEVANCE

    result = cross_check(
        "c-union", [DOC], "does X bear on Y?",
        _scripted("final(marfan syndrome | aortic root dilation | yes | cosmic ray exposure)"),
        _scripted("final(marfan syndrome | aortic root dilation | yes | cosmic ray exposure)"),
        frame=FRAME_RELEVANCE,
        concept_graph=_marfan_graph(),
    )
    assert result.frame_comparison.agrees is True, "the slot comparison itself sees clean agreement"
    assert result.needs_review is True, "but the union with mechanism_check still flags it"


def test_a_grounded_agreed_mechanism_needs_no_review():
    from melampo.reasoning.frame_answer import FRAME_RELEVANCE

    result = cross_check(
        "c-clean", [DOC], "does X bear on Y?",
        _scripted("final(marfan syndrome | aortic root dilation | yes | connective tissue weakness)"),
        _scripted("final(marfan syndrome | aortic root dilation | yes | connective tissue weakness)"),
        frame=FRAME_RELEVANCE,
        concept_graph=_marfan_graph(),
    )
    assert result.mechanism_check.disposition == "agreed_and_grounded"
    assert result.needs_review is False


def test_mechanism_check_is_only_attempted_for_the_relevance_frame():
    """concept_graph passed with a different frame must not attempt mechanism
    verification against slots that frame does not have."""
    from melampo.reasoning.frame_answer import FRAME_MEDICATION

    result = cross_check(
        "c-otherframe", [DOC], "dose?",
        _scripted("final(prednisone | 40 mg | daily | affirmed)"),
        _scripted("final(prednisone | 40 mg | daily | affirmed)"),
        frame=FRAME_MEDICATION,
        concept_graph=_marfan_graph(),
    )
    assert result.mechanism_check is None


def test_as_dict_carries_mechanism_check_when_present():
    from melampo.reasoning.frame_answer import FRAME_RELEVANCE

    result = cross_check(
        "c-dict2", [DOC], "does X bear on Y?",
        _scripted("final(marfan syndrome | aortic root dilation | yes | connective tissue weakness)"),
        _scripted("final(marfan syndrome | aortic root dilation | yes | connective tissue weakness)"),
        frame=FRAME_RELEVANCE,
        concept_graph=_marfan_graph(),
    )
    payload = result.as_dict()
    assert payload["mechanism_check"] is not None
    assert "disposition" in payload["mechanism_check"]
