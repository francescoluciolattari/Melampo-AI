import time

from melampo.evaluation.format_adherence_bench import (
    BenchCase,
    BenchReport,
    ModelResult,
    bench_model,
    bench_models,
    rank_result_dicts,
)
from melampo.memory.context_environment import EnvironmentDocument
from melampo.models.model_client import ModelClientConfig, SafeModelClient
from melampo.models.rlm_model_adapter import (
    DEFAULT_CANDIDATES,
    LICENCE_APACHE_2,
    LICENCE_GEMMA_TERMS,
    RootModelAdapter,
    RootModelCandidate,
)
from melampo.reasoning.rlm_engine import Budget
from melampo.reasoning.rlm_engine import Budget as _RlmBudget


def _doc(text: str) -> EnvironmentDocument:
    return EnvironmentDocument("report_1", text, metadata={"data_class": "synthetic"})


CASES = (
    BenchCase("c1", (_doc("Chest radiograph shows bibasilar opacities. Prednisone 40 mg was started."),), "dose?"),
    BenchCase("c2", (_doc("Progressive dyspnoea over three weeks with no fever."),), "symptoms?"),
)


def _obedient(prompt: str) -> str:
    return "grep(prednisone)" if "grep(prednisone)" not in prompt else "final(40 mg)"


def _near_miss(prompt: str) -> str:
    return "grep prednisone"


def _prose(prompt: str) -> str:
    return "I should look for the prednisone dosage in the report."


# --------------------------------------------------------------------------
# The two numbers that decide
# --------------------------------------------------------------------------


def test_an_obedient_model_scores_full_adherence_and_completes():
    result = bench_model("obedient", _obedient, CASES)
    assert result.adherence == 1.0
    assert result.completion_rate == 1.0
    assert result.rejected_lines == 0


def test_a_model_writing_prose_scores_zero_adherence():
    result = bench_model("prose", _prose, CASES)
    assert result.adherence == 0.0
    assert result.completion_rate == 0.0
    assert result.prose_examples_present()


def test_stop_reasons_are_counted_so_failures_are_diagnosable():
    result = bench_model("prose", _prose, CASES)
    assert result.stop_reasons["model_emitted_no_action"] == 2


def test_a_model_can_emit_valid_actions_and_still_never_complete():
    """Well-formed forever is not success; the engine records budget exhaustion."""
    result = bench_model("never_finals", lambda prompt: "grep(fever)", CASES)
    assert result.adherence == 1.0
    assert result.completion_rate == 0.0
    assert "iteration_budget_exhausted" in result.stop_reasons


# --------------------------------------------------------------------------
# Near misses are the actionable half of a bad result
# --------------------------------------------------------------------------


def test_a_near_miss_is_separated_from_prose():
    """'grep prednisone' is an afternoon of parser work; prose is research."""
    near = bench_model("near", _near_miss, CASES)
    prose = bench_model("prose", _prose, CASES)

    assert near.near_miss_share == 1.0
    assert prose.near_miss_share == 0.0
    assert near.near_misses and not near.prose_lines
    assert prose.prose_lines and not prose.near_misses


def test_the_verdict_distinguishes_a_prompt_problem_from_a_model_problem():
    near_report = bench_models({"a": _near_miss, "b": _near_miss}, CASES)
    prose_report = bench_models({"a": _prose, "b": _prose}, CASES)

    assert "prompt and parser work" in near_report.verdict()
    assert "no model choice fixes that" in prose_report.verdict()


def test_a_model_meeting_the_target_settles_the_choice():
    report = bench_models({"obedient": _obedient, "prose": _prose}, CASES)
    assert "settled" in report.verdict()
    assert report.ranked()[0].model_name == "obedient"


def test_models_are_compared_on_the_same_cases():
    report = bench_models({"a": _obedient, "b": _prose}, CASES)
    assert {item.runs for item in report.results} == {len(CASES)}


def test_an_empty_bench_states_that_it_decided_nothing():
    from melampo.evaluation.format_adherence_bench import BenchReport

    assert BenchReport().verdict() == "no models benched"


# --------------------------------------------------------------------------
# Candidate registry: a licence question is visible next to the model
# --------------------------------------------------------------------------


def test_apache_candidates_are_cleared_for_eu_commercial_use():
    for candidate in DEFAULT_CANDIDATES:
        if candidate.licence == LICENCE_APACHE_2:
            assert candidate.eu_commercial_cleared is True


def test_gemma_is_benched_but_flagged_for_licence_review():
    """Benched, liked and adopted before anyone checks is the failure to prevent."""
    gemma = next(item for item in DEFAULT_CANDIDATES if item.provider == "google")
    assert gemma.licence == LICENCE_GEMMA_TERMS
    assert gemma.eu_commercial_cleared is None
    assert "review" in gemma.note.lower()


def test_an_unknown_licence_is_not_assumed_cleared():
    candidate = RootModelCandidate("x", "y", licence="Some Custom Licence")
    assert candidate.eu_commercial_cleared is None


def test_the_registry_covers_every_benched_family():
    providers = {item.provider for item in DEFAULT_CANDIDATES}
    assert providers == {
        "mistral", "qwen", "google", "meta", "anthropic", "openai", "z-ai",
        "moonshotai", "deepseek", "xai", "nvidia",
    }


def test_the_rejected_gateway_is_named_in_the_registry_source():
    """Data as well as prose: the rejection reasoning must be visible next to the
    candidates it governs, not only in the decision record."""
    import inspect

    import melampo.models.rlm_model_adapter as module

    source = inspect.getsource(module)
    assert "oneprovider.dev" in source
    assert "not Claude" in source


def test_all_three_claude_tiers_are_benched_rather_than_one_assumed_representative():
    """A cost premium is worth paying only if a measurement shows it earns adherence."""
    from melampo.models.rlm_model_adapter import DEFAULT_CANDIDATES

    tiers = {item.name for item in DEFAULT_CANDIDATES if item.provider == "anthropic"}
    assert len(tiers) == 3, "Sonnet, Opus and Fable must each be measured, not extrapolated from one"


def test_openai_is_present_after_being_omitted_from_the_first_registry():
    """The omission was an oversight, not a decision, and the record says so."""
    from melampo.models.rlm_model_adapter import LICENCE_OPENAI_COMMERCIAL

    openai_candidate = next(item for item in DEFAULT_CANDIDATES if item.provider == "openai")
    assert openai_candidate.licence == LICENCE_OPENAI_COMMERCIAL
    assert openai_candidate.eu_commercial_cleared is None
    assert "oversight" in openai_candidate.note.lower()


# --------------------------------------------------------------------------
# Adapter: a refused call is no action, not an exception
# --------------------------------------------------------------------------


def test_a_disabled_client_yields_empty_text_rather_than_raising():
    client = SafeModelClient(provider="p", model_name="m", role="root", config=ModelClientConfig())
    adapter = RootModelAdapter(client=client)
    assert adapter("any prompt") == ""
    assert adapter.report()["not_called"] == 1


def test_an_engine_driven_by_a_disabled_client_records_no_action():
    from melampo.reasoning.rlm_engine import STOP_NO_ACTION, RlmEngine

    client = SafeModelClient(provider="p", model_name="m", role="root", config=ModelClientConfig())
    trajectory = RlmEngine(root_model=RootModelAdapter(client=client)).run(
        "c1", [_doc("text")], "q"
    )
    assert trajectory.stop_reason == STOP_NO_ACTION
    assert trajectory.completed is False


def test_the_adapter_requests_deterministic_decoding():
    captured = {}

    class _Client:
        def execute(self, payload):
            captured.update(payload)
            return {"status": "completed", "text": "final(x)"}

    RootModelAdapter(client=_Client())("prompt")
    assert captured["temperature"] == 0.0


def test_text_is_extracted_from_a_nested_response_shape():
    class _Client:
        def execute(self, payload):
            return {"status": "completed", "response": {"completion": "grep(fever)"}}

    assert RootModelAdapter(client=_Client())("prompt") == "grep(fever)"


def test_a_completed_call_is_counted():
    class _Client:
        def execute(self, payload):
            return {"status": "completed", "text": "final(x)"}

    adapter = RootModelAdapter(client=_Client())
    adapter("prompt")
    assert adapter.report() == {"calls": 1, "not_called": 0}


def test_llama_is_benched_for_comparison_but_flagged():
    """Benching is not adopting: an unresolved licence must stay visible."""
    from melampo.models.rlm_model_adapter import (
        BENCH_ONLY_UNTIL_LICENCE_REVIEW,
    )

    llama = next(item for item in DEFAULT_CANDIDATES if item.provider == "meta")
    assert llama.name == "llama-3.3-70b", "the EU-restricted Llama 4 family is deliberately absent"
    assert llama.eu_commercial_cleared is None
    assert llama.licence in BENCH_ONLY_UNTIL_LICENCE_REVIEW


def test_every_candidate_with_an_unresolved_licence_is_marked_bench_only():
    from melampo.models.rlm_model_adapter import BENCH_ONLY_UNTIL_LICENCE_REVIEW

    for candidate in DEFAULT_CANDIDATES:
        if candidate.eu_commercial_cleared is None:
            assert candidate.licence in BENCH_ONLY_UNTIL_LICENCE_REVIEW
            assert "review" in candidate.note.lower()


def test_a_model_mixing_prose_with_actions_scores_between_the_extremes():
    """The realistic case: understands the format, wraps it in commentary."""
    result = bench_model("mixed", lambda p: "Let me search.\ngrep(prednisone)\nfinal(x)", CASES)
    assert 0.0 < result.adherence < 1.0
    assert result.completion_rate == 1.0, "commentary does not prevent completion"
    assert result.prose_lines == ["Let me search.", "Let me search."]


def test_a_model_producing_no_output_at_all_is_not_misdiagnosed_as_a_syntax_problem():
    """0/0 near-miss share must not read as 'mostly near misses': that would
    send a connectivity or auth failure down the prompt-fixing path instead."""
    silent_report = bench_models({"unreachable": lambda p: ""}, CASES)
    assert "check API keys" in silent_report.verdict()
    assert "near misses" not in silent_report.verdict()


def test_a_mix_of_silent_and_producing_models_uses_the_producing_ones_to_diagnose():
    report = bench_models({"unreachable": lambda p: "", "near": _near_miss}, CASES)
    assert "near misses" in report.verdict()


def test_glm_is_cleared_for_eu_commercial_use_under_its_mit_licence():
    """MIT carries no acceptable-use policy to review, unlike Llama or Gemma."""
    from melampo.models.rlm_model_adapter import LICENCE_MIT

    glm = next(item for item in DEFAULT_CANDIDATES if item.provider == "z-ai")
    assert glm.licence == LICENCE_MIT
    assert glm.eu_commercial_cleared is True


def test_bench_models_accepts_a_custom_budget_factory():
    """Every case must get its own Budget instance, not a shared one."""
    from melampo.reasoning.rlm_engine import Budget

    seen_budgets = []

    def _tracking_factory():
        budget = Budget(max_iterations=2)
        seen_budgets.append(budget)
        return budget

    bench_models({"a": lambda p: "grep(x)\ngrep(y)\ngrep(z)"}, CASES, budget_factory=_tracking_factory)
    assert len(seen_budgets) == len(CASES), "a fresh Budget per case, not one reused across all of them"
    assert all(budget.iterations <= 2 for budget in seen_budgets), "the custom limit must actually apply"


# --------------------------------------------------------------------------
# Iteration diagnostics: distinguishing budget-bound from genuinely stuck
# --------------------------------------------------------------------------


def test_a_model_that_always_hits_the_ceiling_is_reported_as_budget_bound():
    """This is the pattern Sonnet and Fable showed on the real run: every
    incomplete run used exactly the max_iterations allowed."""

    def _never_finals(prompt):
        return "grep(x)"

    result = bench_model(
        "never-finals", _never_finals, CASES, budget_factory=lambda: _RlmBudget(max_iterations=3)
    )
    assert result.completion_rate == 0.0
    assert result.budget_bound is True
    assert result.mean_iterations_on_incompletion == 3.0


def test_a_model_that_stops_short_is_not_reported_as_budget_bound():
    """A model emitting prose immediately stops at iteration 0, well under
    any budget -- a wider budget would not have helped it finish."""
    result = bench_model("prose-model", lambda p: "I will think about this.", CASES)
    assert result.completion_rate == 0.0
    assert result.budget_bound is False


def test_iterations_on_completion_are_tracked_separately():
    def _finals_immediately(prompt):
        return "final(answer)"

    result = bench_model("fast-finisher", _finals_immediately, CASES)
    assert result.completion_rate == 1.0
    assert result.mean_iterations_on_completion == 1.0
    assert result.iterations_on_incompletion == []


def test_no_incomplete_runs_reports_budget_bound_as_false_not_true():
    """An empty 'all incomplete runs hit the ceiling' must not vacuously read True."""

    def _finals_immediately(prompt):
        return "final(answer)"

    result = bench_model("always-finishes", _finals_immediately, CASES)
    assert result.budget_bound is False


def test_the_payload_carries_the_new_diagnostics():
    def _never_finals(prompt):
        return "grep(x)"

    payload = bench_model("m", _never_finals, CASES).as_dict()
    assert "mean_iterations_on_completion" in payload
    assert "mean_iterations_on_incompletion" in payload
    assert "budget_bound" in payload
    assert payload["mean_iterations_on_completion"] is None


# --------------------------------------------------------------------------
# Pure verdict/ranking functions: usable live or from results loaded off disk
# --------------------------------------------------------------------------


def test_compute_verdict_matches_bench_report_verdict_on_the_same_data():
    """The extraction must not change behaviour: same numbers, same conclusion,
    whether read from a live BenchReport or from plain dicts."""
    from melampo.evaluation.format_adherence_bench import BenchReport, compute_verdict

    live_report = BenchReport(
        results=[bench_model("obedient", lambda p: "final(x)", CASES)],
        adherence_target=0.95,
    )
    as_dicts = [item.as_dict() for item in live_report.results]
    assert compute_verdict(as_dicts, 0.95) == live_report.verdict()


def test_compute_verdict_on_dicts_reconstructed_from_json_round_trip():
    """The exact scenario a merge step faces: results loaded back from a file."""
    import json

    from melampo.evaluation.format_adherence_bench import compute_verdict

    result = bench_model("obedient", lambda p: "final(x)", CASES)
    round_tripped = json.loads(json.dumps(result.as_dict()))
    assert compute_verdict([round_tripped], 0.95) == (
        f"obedient meets the adherence target ({round_tripped['adherence']:.0%}); "
        "the choice is settled on these cases"
    )


def test_rank_result_dicts_orders_the_same_way_as_ranked():
    from melampo.evaluation.format_adherence_bench import rank_result_dicts

    a = bench_model("a", lambda p: "final(x)", CASES).as_dict()
    b = bench_model("b", lambda p: "I will think about it.", CASES).as_dict()
    ranked = rank_result_dicts([b, a])
    assert [item["model_name"] for item in ranked] == ["a", "b"]


def test_compute_verdict_on_an_empty_list():
    from melampo.evaluation.format_adherence_bench import compute_verdict

    assert compute_verdict([]) == "no models benched"


# --------------------------------------------------------------------------
# Regression: BenchReport.as_dict() was deleted during a refactor and never
# called by any existing test, so the gap surfaced only on a live run.
# --------------------------------------------------------------------------


def test_bench_report_as_dict_does_not_raise():
    """The exact call _run() makes: report.as_dict(), on a real BenchReport
    instance, not on a ModelResult. This is what every previous test in this
    file omitted -- each called .as_dict() on individual ModelResult objects,
    never on the BenchReport wrapping them, which is what a live run does."""
    report = bench_models({"a": lambda p: "final(x)"}, CASES)
    payload = report.as_dict()
    assert isinstance(payload, dict)


def test_bench_report_as_dict_has_the_shape_run_format_adherence_bench_writes():
    """scripts/run_format_adherence_bench.py's _run() reads models, verdict
    and results straight off this payload -- if any key is missing or
    renamed here, the live script breaks the same way it just did."""
    report = bench_models({"a": lambda p: "final(x)", "b": lambda p: "prose"}, CASES)
    payload = report.as_dict()

    assert payload["models"] == 2
    assert payload["adherence_target"] == 0.95
    assert isinstance(payload["verdict"], str) and payload["verdict"]
    assert len(payload["results"]) == 2
    assert all("model_name" in item for item in payload["results"])


def test_bench_report_as_dict_results_are_ranked():
    """as_dict()'s results must be in ranked order, matching .ranked(), since
    the summary step in both workflows reads row order directly as the
    comparison table's row order."""
    report = bench_models({"weak": lambda p: "prose", "strong": lambda p: "final(x)"}, CASES)
    payload = report.as_dict()
    assert [item["model_name"] for item in payload["results"]] == ["strong", "weak"]


def test_bench_report_as_dict_on_an_empty_report():
    payload = BenchReport().as_dict()
    assert payload["models"] == 0
    assert payload["results"] == []
    assert payload["verdict"] == "no models benched"


def test_grok_gemini_nemotron_are_in_the_default_candidates_registry():
    """The Python-level registry (with licence notes), not just the script's
    CANDIDATE_MODELS table, must also carry the new candidates -- they are
    two different data structures serving different purposes and both need
    updating together."""
    names = {item.name for item in DEFAULT_CANDIDATES}
    for expected in ("grok-4.6", "gemini-3-pro-preview", "nemotron-3-super"):
        assert expected in names


def test_nemotron_note_names_its_relevant_capability():
    """Not a generic capability claim: the note should reflect the specific
    reason this candidate was added over other options considered."""
    nemotron = next(item for item in DEFAULT_CANDIDATES if item.name == "nemotron-3-super")
    assert "cross-document" in nemotron.note.lower() or "multi-step" in nemotron.note.lower()


# --------------------------------------------------------------------------
# Latency circuit breaker: a candidate whose recent cases are consistently
# slow is abandoned early rather than run through every remaining case.
# --------------------------------------------------------------------------


def test_the_circuit_breaker_trips_after_the_window_of_consecutive_slow_cases():
    """The scenario a real run motivated this for: a candidate whose last few
    cases each used their entire per-case wall-clock allowance has already
    shown what running the rest would show again. A tiny real sleep paired
    with a smaller ceiling forces every case's elapsed/ceiling ratio past the
    threshold deterministically -- elapsed_seconds is rounded to milliseconds
    by Budget, so an effectively-instant scripted model would round to
    exactly 0.000 and never exceed any ceiling regardless of how small."""
    import time as time_module

    import melampo.evaluation.format_adherence_bench as bench_module

    cases = tuple(BenchCase(f"c{i}", (EnvironmentDocument("d", "text", metadata={"data_class": "synthetic"}),), "q")
                  for i in range(6))

    def slow_model(prompt):
        time_module.sleep(0.01)
        return "final(x)"

    result = bench_model("slow", slow_model, cases, budget_factory=lambda: Budget(max_wall_clock_seconds=0.005))

    assert result.runs == bench_module.LATENCY_CIRCUIT_BREAKER_WINDOW
    assert result.abandoned_for_latency is True
    assert result.cases_skipped_for_latency == len(cases) - bench_module.LATENCY_CIRCUIT_BREAKER_WINDOW


def test_the_circuit_breaker_does_not_trip_on_a_single_slow_case():
    """One slow case -- a network blip, a transient provider queue -- must
    not condemn an otherwise well-behaved candidate. The model genuinely
    sleeps only on the isolated slow case (call 2 of 5), paired with a tight
    ceiling for that one case only, so its ratio actually exceeds the
    threshold while the surrounding cases' fast, generously-budgeted ratios
    stay low -- never three consecutive over threshold, so no trip."""
    import time as time_module

    cases = tuple(BenchCase(f"c{i}", (EnvironmentDocument("d", "text", metadata={"data_class": "synthetic"}),), "q")
                  for i in range(5))

    call_count = {"n": 0}

    def one_tight_budget():
        call_count["n"] += 1
        return Budget(max_wall_clock_seconds=0.005) if call_count["n"] == 2 else Budget(max_wall_clock_seconds=5.0)

    def mostly_fast_model(prompt):
        if call_count["n"] == 2:  # the one case with the tight ceiling
            time_module.sleep(0.01)
        return "final(x)"

    result = bench_model("mostly-fast", mostly_fast_model, cases, budget_factory=one_tight_budget)

    assert result.runs == len(cases), "all cases must still run"
    assert result.abandoned_for_latency is False
    assert result.case_latency_ratios[1] >= 1.0, "the isolated slow case must genuinely have tripped its own ceiling"


def test_a_fast_model_never_trips_the_breaker():
    cases = tuple(BenchCase(f"c{i}", (EnvironmentDocument("d", "text", metadata={"data_class": "synthetic"}),), "q")
                  for i in range(6))
    result = bench_model("fast", lambda p: "final(x)", cases)
    assert result.abandoned_for_latency is False
    assert result.cases_skipped_for_latency == 0
    assert result.runs == len(cases)


def test_case_elapsed_seconds_are_tracked_per_case():
    cases = tuple(BenchCase(f"c{i}", (EnvironmentDocument("d", "text", metadata={"data_class": "synthetic"}),), "q")
                  for i in range(3))
    result = bench_model("m", lambda p: "final(x)", cases)
    assert len(result.case_elapsed_seconds) == 3
    assert all(isinstance(value, float) and value >= 0 for value in result.case_elapsed_seconds)


def test_mean_and_max_case_seconds_are_none_when_no_cases_ran():
    result = ModelResult(model_name="never-run")
    assert result.mean_case_seconds is None
    assert result.max_case_seconds is None


def test_as_dict_carries_the_latency_fields():
    cases = tuple(BenchCase(f"c{i}", (EnvironmentDocument("d", "text", metadata={"data_class": "synthetic"}),), "q")
                  for i in range(2))
    payload = bench_model("m", lambda p: "final(x)", cases).as_dict()
    for key in ("mean_case_seconds", "max_case_seconds", "abandoned_for_latency", "cases_skipped_for_latency"):
        assert key in payload


def test_the_breaker_condemning_reason_survives_into_stop_reasons_context():
    """A verdict-reader looking only at stop_reasons must not be misled: an
    abandoned candidate's stop_reasons reflect the cases actually run, and
    abandoned_for_latency is what explains the missing ones."""
    import time as time_module

    cases = tuple(BenchCase(f"c{i}", (EnvironmentDocument("d", "text", metadata={"data_class": "synthetic"}),), "q")
                  for i in range(5))

    def slow_model(prompt):
        time_module.sleep(0.01)
        return "final(x)"

    result = bench_model(
        "slow", slow_model, cases, budget_factory=lambda: Budget(max_wall_clock_seconds=0.005)
    )
    assert sum(result.stop_reasons.values()) == result.runs
    assert result.runs < len(cases)


def test_a_candidate_consistently_at_80_percent_of_ceiling_now_trips():
    """The exact real-world gap that motivated lowering the threshold from
    1.0 to 0.75: gemini-3-pro-preview reliably used most but not all of its
    90s ceiling, never hit 100% on any case, and was killed by the job
    timeout with zero diagnostic data instead of being recognised and
    abandoned early. A candidate using 80% of its ceiling every time must now
    trip -- it would not have under the old threshold."""
    import melampo.evaluation.format_adherence_bench as bench_module

    assert bench_module.LATENCY_CIRCUIT_BREAKER_THRESHOLD_FRACTION < 0.8, (
        "this test assumes the threshold was lowered below 0.8; if it changes again, "
        "this assumption needs revisiting alongside it"
    )

    cases = tuple(BenchCase(f"c{i}", (EnvironmentDocument("d", "text", metadata={"data_class": "synthetic"}),), "q")
                  for i in range(6))
    ceiling = 0.02

    def eighty_percent_model(prompt):
        time.sleep(ceiling * 0.8)
        return "final(x)"

    result = bench_model(
        "consistently-slow-not-exhausted", eighty_percent_model, cases,
        budget_factory=lambda: Budget(max_wall_clock_seconds=ceiling),
    )

    assert result.abandoned_for_latency is True
    assert all(ratio < 1.0 for ratio in result.case_latency_ratios), (
        "none of these cases exhausted their ceiling -- the old threshold would have missed this entirely"
    )


# --------------------------------------------------------------------------
# Efficiency tiebreak: three candidates tied at 100%/100% on a real run,
# distinguishable only by data the report already collected but never used
# for ranking.
# --------------------------------------------------------------------------


def _perfect_result(name: str, mean_seconds) -> dict:
    return {
        "model_name": name, "adherence": 1.0, "completion_rate": 1.0,
        "accepted_lines": 10, "rejected_lines": 0, "near_miss_share": 0.0,
        "mean_case_seconds": mean_seconds,
    }


def test_rank_result_dicts_breaks_a_tie_on_mean_case_seconds():
    results = [
        _perfect_result("slow", 5.3),
        _perfect_result("fast", 1.3),
        _perfect_result("medium", 4.3),
    ]
    ranked = rank_result_dicts(results)
    assert [item["model_name"] for item in ranked] == ["fast", "medium", "slow"]


def test_the_verdict_names_the_tiebreak_when_multiple_candidates_are_tied():
    results = [_perfect_result("slow", 5.3), _perfect_result("fast", 1.3)]
    from melampo.evaluation.format_adherence_bench import compute_verdict

    verdict = compute_verdict(results)
    assert "fast" in verdict
    assert "tied on adherence and completion" in verdict
    assert "1.3s" in verdict


def test_the_verdict_does_not_mention_a_tiebreak_when_only_one_candidate_wins_outright():
    results = [_perfect_result("clear-winner", 1.0), {**_perfect_result("also-ran", 9.0), "adherence": 0.5}]
    from melampo.evaluation.format_adherence_bench import compute_verdict

    verdict = compute_verdict(results)
    assert "tied" not in verdict


def test_a_missing_mean_case_seconds_sorts_last_among_ties_rather_than_crashing():
    """A result predating this field, or one with zero completed cases, must
    not break ranking or silently be treated as fastest."""
    results = [
        {**_perfect_result("has-timing", 3.0)},
        {**_perfect_result("no-timing", None)},
    ]
    ranked = rank_result_dicts(results)
    assert [item["model_name"] for item in ranked] == ["has-timing", "no-timing"]


def test_bench_report_ranked_applies_the_same_tiebreak_as_rank_result_dicts():
    """The two ranking implementations (ModelResult-based and dict-based)
    must agree, since a live run uses one and a merge step uses the other."""
    fast = bench_model("fast", lambda p: "final(x)", CASES)
    slow_report = BenchReport(results=[fast])
    assert slow_report.ranked()[0].model_name == "fast"


# --------------------------------------------------------------------------
# all_cases_completed: distinguishes budget_bound's ambiguous False
# --------------------------------------------------------------------------


def test_all_cases_completed_is_true_when_every_case_finished():
    result = bench_model("perfect", lambda p: "final(x)", CASES)
    assert result.all_cases_completed is True
    assert result.budget_bound is False, "budget_bound alone cannot tell this apart from a real failure"


def test_all_cases_completed_is_false_when_any_case_failed_for_any_reason():
    result = bench_model("prose", lambda p: "not an action", CASES)
    assert result.all_cases_completed is False


def test_all_cases_completed_is_false_on_a_never_run_result():
    result = ModelResult(model_name="never-run")
    assert result.all_cases_completed is False


def test_as_dict_carries_all_cases_completed():
    payload = bench_model("m", lambda p: "final(x)", CASES).as_dict()
    assert "all_cases_completed" in payload
