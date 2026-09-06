
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
    assert providers == {"mistral", "qwen", "google", "meta"}


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
