
from melampo.evaluation.format_adherence_bench import (
    BenchCase,
    bench_model,
    bench_models,
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
        "moonshotai", "deepseek", "xai",
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
