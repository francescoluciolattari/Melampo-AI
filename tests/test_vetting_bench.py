"""Tests for the vetting bench -- measuring a candidate on vetting, not navigation."""

from melampo.evaluation.vetting_bench import (
    VETTING_CASES,
    VettingCase,
    VettingResult,
    bench_vetting,
    rank_vetting_results,
    vetting_graph,
    vetting_table,
)
from melampo.reasoning.frame_answer import FRAME_RELEVANCE, parse_frame_answer

_MECHANISMS = {
    "marfan": ("marfan syndrome", "aortic root dilation", "connective tissue weakness"),
    "kidney": ("chronic kidney disease", "renal osteodystrophy", "secondary hyperparathyroidism"),
    "coeliac": ("coeliac disease", "iron malabsorption", "villous atrophy"),
    "sarcoid": ("sarcoidosis", "hypercalcaemia", "granuloma formation"),
}


def _correct_model(prompt: str) -> str:
    lowered = prompt.lower()
    for key, (factor, target, mechanism) in _MECHANISMS.items():
        if key in lowered:
            return f"{factor} | {target} | yes | {mechanism}"
    return "unknown | unknown | no | unknown"


def _inventing_model(prompt: str) -> str:
    return _correct_model(prompt).rsplit("|", 1)[0] + "| cosmic ray exposure"


def _prose_model(prompt: str) -> str:
    return "Yes, I think the history is relevant here."


# --------------------------------------------------------------------------
# The bench must separate the three ways a candidate can behave
# --------------------------------------------------------------------------


def test_a_correct_candidate_scores_fully_grounded():
    result = bench_vetting("correct", _correct_model, VETTING_CASES, vetting_graph(), table=vetting_table())
    assert result.format_rate == 1.0
    assert result.grounding_rate == 1.0


def test_a_candidate_inventing_mechanisms_formats_well_but_is_not_grounded():
    """The failure this bench exists to catch: a well-formed answer whose
    mechanism the graph does not support."""
    result = bench_vetting("inventing", _inventing_model, VETTING_CASES, vetting_graph(), table=vetting_table())
    assert result.format_rate == 1.0, "the format is fine"
    assert result.grounding_rate == 0.0, "the substance is not"


def test_a_candidate_producing_prose_fails_on_format_not_grounding():
    """Distinguishing these matters: a format failure is prompt work, a
    grounding failure is a reason to prefer a different candidate."""
    result = bench_vetting("prose", _prose_model, VETTING_CASES, vetting_graph(), table=vetting_table())
    assert result.format_rate == 0.0
    assert result.grounding_rate == 0.0
    assert result.malformed_answers, "the unparseable output is recorded for inspection"


def test_naming_a_wrong_mechanism_between_connected_concepts_still_counts_as_useful():
    """Wrong in a recoverable way -- the graph knows a connection and can
    offer what it is -- as distinct from claiming a connection that does not
    exist at all."""
    result = bench_vetting("inventing", _inventing_model, VETTING_CASES, vetting_graph(), table=vetting_table())
    assert result.useful_rate == 1.0
    assert result.connection_but_wrong_mechanism == len(VETTING_CASES)


def test_claiming_a_connection_the_graph_does_not_know_is_counted_separately():
    unconnected = (
        VettingCase("unrelated", "sarcoidosis", "villous atrophy", "none", "Are these related?"),
    )

    def confident_model(prompt: str) -> str:
        return "sarcoidosis | villous atrophy | yes | some invented link"

    result = bench_vetting("confident", confident_model, unconnected, vetting_graph(), table=vetting_table())
    assert result.no_connection_claimed == 1
    assert result.useful_rate == 0.0


# --------------------------------------------------------------------------
# Scoring choices that were deliberate
# --------------------------------------------------------------------------


def test_grounding_rate_is_over_all_runs_not_only_well_formed_ones():
    """A candidate with three perfect answers and seventeen malformed ones
    has not earned 100%."""
    cases = VETTING_CASES
    call_count = {"n": 0}

    def half_broken(prompt: str) -> str:
        call_count["n"] += 1
        return _correct_model(prompt) if call_count["n"] == 1 else "unparseable"

    result = bench_vetting("half", half_broken, cases, vetting_graph(), table=vetting_table())
    assert result.grounded == 1
    assert result.grounding_rate == 1 / len(cases)
    assert result.grounding_rate < result.format_rate or result.format_rate == result.grounding_rate


def test_ranking_puts_grounding_ahead_of_format():
    """A candidate that grounds well and formats poorly is a prompt problem;
    one that formats perfectly and grounds badly is a candidate problem."""
    results = [
        bench_vetting("inventing", _inventing_model, VETTING_CASES, vetting_graph(), table=vetting_table()),
        bench_vetting("correct", _correct_model, VETTING_CASES, vetting_graph(), table=vetting_table()),
        bench_vetting("prose", _prose_model, VETTING_CASES, vetting_graph(), table=vetting_table()),
    ]
    ranked = rank_vetting_results(results)
    assert [item.model_name for item in ranked] == ["correct", "inventing", "prose"]


def test_an_empty_result_reports_zero_rather_than_dividing_by_zero():
    empty = VettingResult(model_name="never-run")
    assert empty.format_rate == 0.0
    assert empty.grounding_rate == 0.0
    assert empty.useful_rate == 0.0


# --------------------------------------------------------------------------
# The fixture itself must be well-formed
# --------------------------------------------------------------------------


def test_every_shipped_case_has_a_mechanism_the_graph_actually_connects():
    """If a case's expected mechanism were not in the graph, the bench would
    be measuring the fixture's gaps rather than the candidate."""
    from melampo.reasoning.mechanism_verification import verify_mechanism

    graph, table = vetting_graph(), vetting_table()
    for case in VETTING_CASES:
        verification = verify_mechanism(graph, case.factor, case.target, case.expected_mechanism, table=table)
        assert verification.is_grounded, f"{case.case_id}: the fixture's own expected mechanism must be grounded"


def test_the_prompt_carries_the_frame_instruction():
    """A candidate cannot be scored on a format it was never told to use."""
    prompt = VETTING_CASES[0].prompt()
    assert "bears_on" in prompt
    assert "mechanism" in prompt


def test_the_prompt_does_not_leak_the_expected_mechanism():
    """The expected mechanism is for asserting the fixture is sound, never
    for showing the candidate the answer."""
    for case in VETTING_CASES:
        assert case.expected_mechanism not in case.prompt()


def test_every_shipped_case_parses_as_a_relevance_frame_when_answered_correctly():
    for case in VETTING_CASES:
        parsed = parse_frame_answer(FRAME_RELEVANCE, _correct_model(case.prompt()))
        assert parsed.value("factor")
        assert parsed.value("mechanism")
