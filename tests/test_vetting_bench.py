"""Tests for the vetting bench v2 -- broader case set, restraint scoring, Wilson intervals."""

from melampo.evaluation.vetting_bench import (
    VETTING_CASES,
    VettingCase,
    VettingResult,
    bench_vetting,
    rank_vetting_results,
    vetting_graph,
    vetting_table,
)
from melampo.memory.concept_paths import resolve_concept
from melampo.reasoning.frame_answer import FRAME_RELEVANCE, parse_frame_answer
from melampo.reasoning.mechanism_verification import verify_mechanism


def _correct_model(prompt: str) -> str:
    """Answers every case correctly, including declining every restraint case outright."""
    case = _case_for_prompt(prompt)
    if case.is_restraint_case:
        return f"{case.factor} | {case.target} | no | none"
    return f"{case.factor} | {case.target} | yes | {case.expected_mechanism}"


def _inventing_model(prompt: str) -> str:
    """Claims a connection on every case, real or not, with a mechanism the graph never taught it."""
    case = _case_for_prompt(prompt)
    return f"{case.factor} | {case.target} | yes | cosmic ray exposure"


def _prose_model(prompt: str) -> str:
    return "Yes, I think the history is relevant here."


def _grounded_but_reckless_model(prompt: str) -> str:
    """Correct on every real connection, but never declines a restraint case --
    the exact failure grounding_rate alone cannot see."""
    case = _case_for_prompt(prompt)
    if case.is_restraint_case:
        return f"{case.factor} | {case.target} | yes | unrelated coincidental finding"
    return f"{case.factor} | {case.target} | yes | {case.expected_mechanism}"


def _case_for_prompt(prompt: str) -> VettingCase:
    for case in VETTING_CASES:
        if case.question[:40] in prompt:
            return case
    raise AssertionError("prompt did not match any shipped case")


# --------------------------------------------------------------------------
# The bench separates the behaviours it exists to separate
# --------------------------------------------------------------------------


def test_a_correct_candidate_scores_fully_grounded_and_restrained():
    result = bench_vetting("correct", _correct_model, VETTING_CASES, vetting_graph(), table=vetting_table(), measure_latency=False)
    assert result.format_rate == 1.0
    assert result.grounding_rate == 1.0
    assert result.restraint_rate == 1.0


def test_a_candidate_inventing_mechanisms_formats_well_but_is_not_grounded():
    result = bench_vetting("inventing", _inventing_model, VETTING_CASES, vetting_graph(), table=vetting_table(), measure_latency=False)
    assert result.format_rate == 1.0
    assert result.grounding_rate == 0.0
    assert result.restraint_rate == 0.0, "it never declines the restraint cases either"


def test_a_candidate_producing_prose_fails_on_format_not_grounding():
    result = bench_vetting("prose", _prose_model, VETTING_CASES, vetting_graph(), table=vetting_table(), measure_latency=False)
    assert result.format_rate == 0.0
    assert result.grounding_rate == 0.0
    assert result.malformed_answers


def test_a_candidate_that_grounds_perfectly_but_never_declines_is_ranked_below_a_fully_correct_one():
    """The scenario this version of the bench was built to catch: identical
    grounding_rate, but one candidate invents connections where none exist."""
    reckless = bench_vetting("reckless", _grounded_but_reckless_model, VETTING_CASES, vetting_graph(), table=vetting_table(), measure_latency=False)
    correct = bench_vetting("correct", _correct_model, VETTING_CASES, vetting_graph(), table=vetting_table(), measure_latency=False)

    assert reckless.grounding_rate == correct.grounding_rate == 1.0, "grounding alone cannot tell them apart"
    assert reckless.restraint_rate < correct.restraint_rate

    ranked = rank_vetting_results([reckless, correct])
    assert ranked[0].model_name == "correct"


# --------------------------------------------------------------------------
# Restraint scoring: the three real outcomes
# --------------------------------------------------------------------------


def test_declining_a_restraint_case_outright_is_correct():
    def model(prompt):
        case = _case_for_prompt(prompt)
        return f"{case.factor} | {case.target} | no | none"

    result = bench_vetting("decliner", model, VETTING_CASES, vetting_graph(), table=vetting_table(), measure_latency=False)
    assert result.restraint_rate == 1.0


def test_inventing_a_mechanism_on_a_restraint_case_is_a_failure():
    def model(prompt):
        case = _case_for_prompt(prompt)
        if case.is_restraint_case:
            return f"{case.factor} | {case.target} | yes | some invented pathway"
        return f"{case.factor} | {case.target} | yes | {case.expected_mechanism}"

    result = bench_vetting("inventor", model, VETTING_CASES, vetting_graph(), table=vetting_table(), measure_latency=False)
    assert result.restraint_rate == 0.0
    outcomes = {c["outcome"] for c in result.per_case if "restraint" in c["case_id"]}
    assert "restraint_failed_invented_connection" in outcomes


def test_a_restraint_case_the_graph_unexpectedly_grounds_counts_as_correct_not_a_candidate_failure():
    """If the graph itself finds the proposed mechanism grounded, the fixture
    was not a restraint case after all -- not something to penalise the
    candidate for."""
    graph = vetting_graph()
    # Use the real expected mechanism of a *different*, genuinely connected
    # case as the "restraint" case's claim, to construct a graph-grounded
    # answer deliberately.
    case = VettingCase("synthetic_restraint", "chronic kidney disease", "renal osteodystrophy", None, "does X bear on Y?")

    def model(prompt):
        return "chronic kidney disease | renal osteodystrophy | yes | secondary hyperparathyroidism"

    result = bench_vetting("lucky", model, [case], graph, table=vetting_table(), measure_latency=False)
    assert result.restraint_correct == 1
    assert result.per_case[0]["outcome"] == "restraint_case_actually_grounded"


# --------------------------------------------------------------------------
# Denominators: restraint cases must not dilute grounding_rate
# --------------------------------------------------------------------------


def test_grounding_rate_excludes_restraint_cases_from_its_denominator():
    result = bench_vetting("correct", _correct_model, VETTING_CASES, vetting_graph(), table=vetting_table(), measure_latency=False)
    restraint_count = sum(1 for c in VETTING_CASES if c.is_restraint_case)
    assert result.conclusive_runs == len(VETTING_CASES) - restraint_count
    assert result.grounding_rate == 1.0, "restraint cases must not dilute this even though they were all declined"


def test_useful_rate_also_excludes_restraint_cases():
    result = bench_vetting("correct", _correct_model, VETTING_CASES, vetting_graph(), table=vetting_table(), measure_latency=False)
    assert result.useful_rate == 1.0


# --------------------------------------------------------------------------
# The Wilson interval: a confidence bound, not just a point estimate
# --------------------------------------------------------------------------


def test_grounding_wilson_interval_is_wider_for_a_smaller_sample():
    """The reason this bench was widened: the same point estimate from a
    smaller sample deserves a wider interval, not equal confidence."""
    small = VettingResult(model_name="small", grounded=3, restraint_expected=0)
    small.runs = 4
    large = VettingResult(model_name="large", grounded=12, restraint_expected=0)
    large.runs = 16

    small_lo, small_hi = small.grounding_wilson_interval
    large_lo, large_hi = large.grounding_wilson_interval

    assert small.grounding_rate == large.grounding_rate == 0.75
    assert (small_hi - small_lo) > (large_hi - large_lo)


def test_ranking_prefers_a_larger_reliable_sample_over_a_smaller_lucky_one_at_the_same_rate():
    small = VettingResult(model_name="small_lucky", grounded=3, restraint_expected=0)
    small.runs = 4
    large = VettingResult(model_name="large_reliable", grounded=12, restraint_expected=0)
    large.runs = 16

    ranked = rank_vetting_results([small, large])
    assert ranked[0].model_name == "large_reliable"


def test_an_empty_result_reports_a_zero_width_interval_not_a_crash():
    empty = VettingResult(model_name="never-run")
    assert empty.grounding_wilson_interval == (0.0, 0.0)


# --------------------------------------------------------------------------
# Repeated trials: real models vary run to run
# --------------------------------------------------------------------------


def test_trials_per_case_multiplies_the_run_count():
    result = bench_vetting("correct", _correct_model, VETTING_CASES, vetting_graph(), table=vetting_table(), trials_per_case=3, measure_latency=False)
    assert result.runs == len(VETTING_CASES) * 3


def test_trials_per_case_folds_variation_into_one_result():
    calls = {"n": 0}

    def flaky_model(prompt):
        calls["n"] += 1
        case = _case_for_prompt(prompt)
        if case.is_restraint_case:
            return f"{case.factor} | {case.target} | no | none"
        # Correct on odd calls, wrong on even ones.
        if calls["n"] % 2 == 0:
            return f"{case.factor} | {case.target} | yes | cosmic ray exposure"
        return f"{case.factor} | {case.target} | yes | {case.expected_mechanism}"

    result = bench_vetting("flaky", flaky_model, VETTING_CASES, vetting_graph(), table=vetting_table(), trials_per_case=2, measure_latency=False)
    assert 0.0 < result.grounding_rate < 1.0, "the mix of right and wrong answers shows up as a rate, not a crash"


# --------------------------------------------------------------------------
# Latency tracking
# --------------------------------------------------------------------------


def test_latency_is_recorded_when_measurement_is_on():
    result = bench_vetting("correct", _correct_model, VETTING_CASES, vetting_graph(), table=vetting_table(), measure_latency=True)
    assert len(result.latencies_seconds) == len(VETTING_CASES)
    assert result.mean_latency_seconds >= 0.0


def test_latency_is_empty_when_measurement_is_off():
    result = bench_vetting("correct", _correct_model, VETTING_CASES, vetting_graph(), table=vetting_table(), measure_latency=False)
    assert result.latencies_seconds == []
    assert result.mean_latency_seconds == 0.0


def test_ranking_uses_latency_only_as_the_final_tie_break():
    fast = VettingResult(model_name="fast", grounded=10, restraint_correct=3, restraint_expected=3)
    fast.runs = 13
    fast.latencies_seconds = [1.0]
    slow = VettingResult(model_name="slow", grounded=10, restraint_correct=3, restraint_expected=3)
    slow.runs = 13
    slow.latencies_seconds = [5.0]

    ranked = rank_vetting_results([slow, fast])
    assert ranked[0].model_name == "fast"


# --------------------------------------------------------------------------
# The fixture itself must be well-formed and broad
# --------------------------------------------------------------------------


def test_the_case_set_is_broad_enough_to_matter():
    assert len(VETTING_CASES) >= 15, "four cases could not separate two candidates with confidence"


def test_every_conclusive_case_has_a_mechanism_the_graph_actually_connects():
    graph, table = vetting_graph(), vetting_table()
    for case in VETTING_CASES:
        if case.is_restraint_case:
            continue
        verification = verify_mechanism(graph, case.factor, case.target, case.expected_mechanism, table=table)
        assert verification.is_grounded, f"{case.case_id}: the fixture's own expected mechanism must be grounded"


def test_every_restraint_case_genuinely_has_no_path():
    """A restraint case with an actual path in the graph would silently stop
    testing what it claims to."""
    graph = vetting_graph()
    for case in VETTING_CASES:
        if not case.is_restraint_case:
            continue
        factor_node = resolve_concept(case.factor, graph)
        target_node = resolve_concept(case.target, graph)
        assert factor_node and target_node, f"{case.case_id}: factor/target must at least resolve"
        factor_reach = {edge.target for edge in graph.edges_from(factor_node)}
        target_reach = {edge.target for edge in graph.edges_from(target_node)}
        assert not (factor_reach & target_reach), f"{case.case_id}: factor and target share a direct neighbour"


def test_the_case_set_spans_multiple_organ_systems():
    """Breadth, not just count: four calcium/renal cases repeated sixteen
    times would not be a broader bench."""
    conclusive = [c for c in VETTING_CASES if not c.is_restraint_case]
    distinct_mechanisms = {c.expected_mechanism for c in conclusive}
    assert len(distinct_mechanisms) >= 10


def test_the_prompt_does_not_leak_the_expected_mechanism():
    for case in VETTING_CASES:
        if case.expected_mechanism:
            assert case.expected_mechanism not in case.prompt()


def test_every_shipped_case_parses_as_a_relevance_frame_when_answered_correctly():
    for case in VETTING_CASES:
        parsed = parse_frame_answer(FRAME_RELEVANCE, _correct_model(case.prompt()))
        assert parsed.value("factor")
        assert parsed.value("bears_on")


def test_an_empty_result_reports_zero_rather_than_dividing_by_zero():
    empty = VettingResult(model_name="never-run")
    assert empty.format_rate == 0.0
    assert empty.grounding_rate == 0.0
    assert empty.useful_rate == 0.0
    assert empty.restraint_rate == 0.0


def test_as_dict_carries_the_new_fields_a_reviewer_needs():
    result = bench_vetting("correct", _correct_model, VETTING_CASES, vetting_graph(), table=vetting_table(), measure_latency=False)
    payload = result.as_dict()
    for key in ("grounding_wilson_interval", "restraint_rate", "restraint_expected", "conclusive_runs", "mean_latency_seconds"):
        assert key in payload
