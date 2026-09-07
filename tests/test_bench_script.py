"""Tests for the bench runner script itself, not the library it calls.

The workflow's usefulness depends on these behaviours, and every one of them
was broken or absent in a version that looked correct: the results file was
not written on the failure path, so the artifact that should explain a failed
run was empty exactly when it mattered most.
"""

import importlib.util
import json
import sys
import urllib.error
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_format_adherence_bench.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("bench_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script():
    return _load_script()


# --------------------------------------------------------------------------
# The results file must exist on every path
# --------------------------------------------------------------------------


def test_results_are_written_even_when_no_candidate_survives(script, tmp_path, monkeypatch):
    """The artifact matters most when the run failed: it carries the reason."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out = tmp_path / "results.json"
    monkeypatch.setattr(sys, "argv", ["bench", "--out", str(out)])

    assert script.main() == 1, "no candidates is still a failure"
    assert out.exists(), "but the diagnostics must survive it"

    payload = json.loads(out.read_text())
    assert payload["status"] == "no_candidates"
    assert payload["results"] == []
    assert payload["preflight"], "every candidate's reason is recorded"


def test_the_failure_payload_names_what_each_status_code_means(script, tmp_path, monkeypatch):
    """A failed run should tell the operator which thing to fix."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out = tmp_path / "results.json"
    monkeypatch.setattr(sys, "argv", ["bench", "--out", str(out)])
    script.main()

    verdict = json.loads(out.read_text())["verdict"]
    assert "401/403" in verdict and "404" in verdict


def test_a_nested_output_directory_is_created(script, tmp_path, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out = tmp_path / "does" / "not" / "exist" / "results.json"
    monkeypatch.setattr(sys, "argv", ["bench", "--out", str(out)])

    script.main()
    assert out.exists()


# --------------------------------------------------------------------------
# Preflight returns an actionable reason, not just a boolean
# --------------------------------------------------------------------------


def test_preflight_reports_missing_keys_per_candidate(script, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    candidates, _skipped, detail = script.build_candidates()

    assert candidates == {}
    assert detail["mistral-small-3.1"] == "MISTRAL_API_KEY not set"
    assert detail["claude-sonnet-5"] == "OPENROUTER_API_KEY not set"
    assert len(detail) == 21, "all candidates accounted for, none silently dropped"


def test_every_candidate_appears_in_the_preflight_detail(script, monkeypatch):
    """A candidate that vanishes without a reason is the failure mode to avoid."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _, _, detail = script.build_candidates()

    for name in ("claude-sonnet-5", "claude-opus-5", "claude-fable-5.1", "gpt-6-astra",
                 "qwen-3.5", "glm-5", "llama-3.3-70b", "gemma-3-27b", "mistral-small-3.1"):
        assert name in detail, f"{name} disappeared without a recorded reason"


# --------------------------------------------------------------------------
# Budget and timeouts are bounded
# --------------------------------------------------------------------------


def test_the_bench_iteration_budget_is_tighter_than_the_engine_default(script):
    """Six one-fact questions about a two-sentence document need fewer turns
    than the engine's general-purpose default, even after the ceiling was
    raised once already following a first live run's 0% completion."""
    from melampo.reasoning.rlm_engine import Budget

    bench_budget = script._bench_budget()
    default = Budget()
    assert bench_budget.max_iterations < default.max_iterations


def test_the_bench_wall_clock_matches_the_engine_default(script):
    """Not tightened below the default: real provider latency, and the raised
    max_tokens for reasoning-capable candidates, both need the room."""
    from melampo.reasoning.rlm_engine import Budget

    assert script._bench_budget().max_wall_clock_seconds == Budget().max_wall_clock_seconds


def test_the_budget_factory_returns_a_fresh_instance_each_call(script):
    """Budget carries mutable per-run state; a shared instance corrupts the count."""
    first, second = script._bench_budget(), script._bench_budget()
    assert first is not second


def test_preflight_timeout_is_shorter_than_the_call_timeout(script):
    """Preflight exists to fail fast; a longer timeout would defeat it."""
    assert script.PREFLIGHT_TIMEOUT_SECONDS < script.CALL_TIMEOUT_SECONDS


def test_the_bench_uses_only_synthetic_documents(script):
    """Phase-one data-class discipline applies here as everywhere else."""
    for case in script.BENCH_CASES:
        for document in case.documents:
            assert document.metadata["data_class"] == "synthetic"


# --------------------------------------------------------------------------
# Malformed provider responses: the defect a real run actually hit
# --------------------------------------------------------------------------


class _FakeResponse:
    """Stands in for the object urllib.request.urlopen's context manager yields."""

    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8") if not isinstance(payload, (bytes, str)) else (
            payload.encode("utf-8") if isinstance(payload, str) else payload
        )

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _mocked_urlopen(monkeypatch, payload):
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _FakeResponse(payload))


def test_an_empty_choices_list_is_diagnosed_not_crashed(script, monkeypatch):
    """The actual defect: OpenRouter can return HTTP 200 with choices: []
    when no endpoint serves the requested model."""
    _mocked_urlopen(monkeypatch, {"choices": []})
    reachable, reason = script._preflight("t", "http://fake", "k", "some/model")
    assert reachable is False
    assert "no choices" in reason.lower()


def test_an_error_field_in_a_200_response_is_diagnosed(script, monkeypatch):
    _mocked_urlopen(monkeypatch, {"error": {"message": "no endpoints found", "code": 404}})
    reachable, reason = script._preflight("t", "http://fake", "k", "some/model")
    assert reachable is False
    assert "no endpoints found" in reason


def test_a_non_object_response_body_is_diagnosed(script, monkeypatch):
    _mocked_urlopen(monkeypatch, [1, 2, 3])
    reachable, reason = script._preflight("t", "http://fake", "k", "some/model")
    assert reachable is False
    assert "not a json object" in reason.lower()


def test_a_message_without_content_is_diagnosed(script, monkeypatch):
    _mocked_urlopen(monkeypatch, {"choices": [{"message": {}}]})
    reachable, reason = script._preflight("t", "http://fake", "k", "some/model")
    assert reachable is False
    assert "no text content" in reason.lower()


def test_bind_degrades_a_malformed_response_to_empty_text_not_a_crash(script, monkeypatch):
    """_bind is what the actual bench calls use; it must never propagate."""
    _mocked_urlopen(monkeypatch, {"choices": []})
    call = script._bind(script._http_chat_completion, "http://fake", "k", "some/model")
    assert call("any prompt") == ""


# --------------------------------------------------------------------------
# The top-level safety net: nothing crashes without leaving a diagnostic
# --------------------------------------------------------------------------


def test_a_completely_unforeseen_exception_still_produces_a_results_file(script, tmp_path, monkeypatch):
    """This is the property the original defect violated: an IndexError two
    calls below any except clause crashed the script with nothing written."""
    out = tmp_path / "results.json"
    monkeypatch.setattr(sys, "argv", ["bench", "--out", str(out)])
    monkeypatch.setattr(script, "build_candidates", lambda only=None: (_ for _ in ()).throw(RuntimeError("never seen before")))

    exit_code = script.main()

    assert exit_code == 1
    assert out.exists(), "a crash must still leave diagnostics behind"
    payload = json.loads(out.read_text())
    assert payload["status"] == "crashed"
    assert payload["error_type"] == "RuntimeError"
    assert "traceback" in payload


def test_the_safety_net_does_not_itself_crash_if_writing_fails(script, monkeypatch, capsys):
    """Belt and braces: even a failure to write the crash file must not raise."""
    monkeypatch.setattr(sys, "argv", ["bench", "--out", "/nonexistent-root-dir/x/results.json"])
    monkeypatch.setattr(script, "build_candidates", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    # _write_results creates missing parent directories, so a bare missing
    # path is not itself unwritable (particularly as root). Force the write
    # itself to fail instead, which is the actual failure this guards against
    # -- e.g. a read-only filesystem or a permissions error at write time.
    monkeypatch.setattr(script, "_write_results", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))

    exit_code = script.main()  # must return, not raise

    assert exit_code == 1
    assert "Could not write crash diagnostics" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Rate-limit retry: what actually failed the first live run (Mistral 429)
# --------------------------------------------------------------------------


def test_a_429_is_retried_once_honouring_retry_after(script, monkeypatch):
    import urllib.error

    calls = {"n": 0}

    def flaky(request, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError("http://fake", 429, "Too Many Requests", {"Retry-After": "0.01"}, None)
        return _FakeResponse({"choices": [{"message": {"content": "final(ok)"}}]})

    monkeypatch.setattr("urllib.request.urlopen", flaky)
    result = script._http_chat_completion("http://fake", "k", "m", "test", timeout=5)
    assert result == "final(ok)"
    assert calls["n"] == 2


def test_two_consecutive_429s_propagate_rather_than_retrying_forever(script, monkeypatch):
    import urllib.error

    calls = {"n": 0}

    def always_429(request, timeout):
        calls["n"] += 1
        raise urllib.error.HTTPError("http://fake", 429, "Too Many Requests", {"Retry-After": "0.01"}, None)

    monkeypatch.setattr("urllib.request.urlopen", always_429)
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        script._http_chat_completion("http://fake", "k", "m", "test", timeout=5)
    assert excinfo.value.code == 429
    assert calls["n"] == 2, "one original attempt plus exactly one retry, not an unbounded loop"


def test_retry_after_is_capped_rather_than_waited_out_in_full(script):
    assert script._parse_retry_after("999") == script.RATE_LIMIT_MAX_WAIT_SECONDS


def test_a_missing_retry_after_falls_back_to_the_default_wait(script):
    assert script._parse_retry_after(None) == script.RATE_LIMIT_DEFAULT_WAIT_SECONDS


def test_a_malformed_retry_after_falls_back_rather_than_raising(script):
    assert script._parse_retry_after("not-a-number") == script.RATE_LIMIT_DEFAULT_WAIT_SECONDS


# --------------------------------------------------------------------------
# The reasoning-disable hint: OpenRouter only, never the direct Mistral call
# --------------------------------------------------------------------------


def test_disable_reasoning_adds_the_openrouter_parameter(script, monkeypatch):
    captured = {}

    def capture(request, timeout):
        captured["body"] = __import__("json").loads(request.data.decode())
        return _FakeResponse({"choices": [{"message": {"content": "final(x)"}}]})

    monkeypatch.setattr("urllib.request.urlopen", capture)
    script._http_chat_completion("http://fake", "k", "m", "p", timeout=5, disable_reasoning=True)
    assert captured["body"]["reasoning"] == {"enabled": False}


def test_without_the_flag_no_reasoning_parameter_is_sent(script, monkeypatch):
    captured = {}

    def capture(request, timeout):
        captured["body"] = __import__("json").loads(request.data.decode())
        return _FakeResponse({"choices": [{"message": {"content": "final(x)"}}]})

    monkeypatch.setattr("urllib.request.urlopen", capture)
    script._http_chat_completion("http://fake", "k", "m", "p", timeout=5, disable_reasoning=False)
    assert "reasoning" not in captured["body"]


def test_the_mistral_direct_candidate_never_receives_the_reasoning_hint(script, monkeypatch):
    """Its tolerance for unrecognised top-level fields is not verified from here."""
    import inspect

    source = inspect.getsource(script.build_candidates)
    # The Mistral direct-API append must not carry disable_reasoning=True;
    # only the OpenRouter loop should set it, via reasoning_capable_via_openrouter.
    mistral_line = next(line for line in source.splitlines() if "mistral-small-3.1" in line and "api.mistral.ai" in line)
    assert mistral_line.rstrip().endswith("False)"), mistral_line


def test_max_tokens_was_raised_after_the_first_run_showed_zero_completion(script, monkeypatch):
    captured = {}

    def capture(request, timeout):
        captured["body"] = __import__("json").loads(request.data.decode())
        return _FakeResponse({"choices": [{"message": {"content": "final(x)"}}]})

    monkeypatch.setattr("urllib.request.urlopen", capture)
    script._http_chat_completion("http://fake", "k", "m", "p", timeout=5)
    assert captured["body"]["max_tokens"] == 1024


def test_the_system_prompt_includes_a_worked_example_and_explicit_prohibitions(script):
    """Targets the three failure patterns the first live run actually showed:
    never finalising, dialogue-style artifacts, narrating tool output."""
    prompt = script._SYSTEM_PROMPT.lower()
    assert "grep(dose)" in script._SYSTEM_PROMPT  # worked example present
    assert "one clean lookup" in prompt or "enough" in prompt  # anti-over-verification
    assert "dialogue" in prompt or "role labels" in prompt  # anti-Opus-artifact


# --------------------------------------------------------------------------
# New candidates: verified slugs, not guesses
# --------------------------------------------------------------------------


def test_all_twenty_one_candidates_are_present(script, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    _, _, detail = script.build_candidates()
    assert len(detail) == 21


def test_qwen_3_5_uses_the_corrected_slug_not_the_invented_one(script):
    """Slugs live in CANDIDATE_MODELS, the top-of-file table, not inline in
    build_candidates -- that is the point of the top-of-file reorganisation."""
    slugs = {model for _, model, _ in script.CANDIDATE_MODELS}
    assert "qwen/qwen3.5-plus-02-15" in slugs
    assert "qwen/qwen-3.5-72b-instruct" not in slugs



def test_mistral_3_6_was_not_added_because_it_does_not_exist(script):
    """Checked against Mistral's full release history before writing any code for it."""
    import inspect
    source = inspect.getsource(script.build_candidates)
    assert "3.6" not in source or "mistral-3.6" not in source.lower()


def test_new_model_families_are_present_in_the_openrouter_candidate_list(script):
    slugs = {model for _, model, _ in script.CANDIDATE_MODELS}
    for expected_slug in (
        "google/gemma-4-31b-it",
        "google/gemma-4-26b-a4b-it",
        "z-ai/glm-5.3",
        "moonshotai/kimi-k2.6",
        "deepseek/deepseek-v4-flash",
        "meta-llama/llama-4-maverick",
        "meta-llama/llama-4-scout",
    ):
        assert expected_slug in slugs, f"{expected_slug} missing from CANDIDATE_MODELS"
    assert any(model.startswith("x-ai/grok-4-fast") for model in slugs)


# --------------------------------------------------------------------------
# HTTP 400 fallback: some providers reject the reasoning-disable hint outright
# --------------------------------------------------------------------------


def test_a_400_with_the_reasoning_hint_retries_once_without_it(script, monkeypatch):
    """The exact mechanism hypothesised for glm-5.3, gpt-6-astra, qwen-3.8 and
    claude-fable-5.1 all returning HTTP 400 on the first live run."""
    calls = []

    def rejects_reasoning_hint(request, timeout):
        body = json.loads(request.data.decode())
        calls.append(body.get("reasoning"))
        if "reasoning" in body:
            raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, None)
        return _FakeResponse({"choices": [{"message": {"content": "final(ok)"}}]})

    monkeypatch.setattr("urllib.request.urlopen", rejects_reasoning_hint)
    result = script._http_chat_completion(
        "http://fake", "k", "z-ai/glm-5.3", "test", timeout=5, disable_reasoning=True
    )
    assert result == "final(ok)"
    assert calls == [{"enabled": False}, None], "hint sent once, then omitted on retry"


def test_a_400_without_the_reasoning_hint_is_not_retried(script, monkeypatch):
    """The fallback is specific to the reasoning hint; an unrelated 400 must
    still propagate rather than being silently swallowed."""

    def always_400(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", always_400)
    with pytest.raises(urllib.error.HTTPError):
        script._http_chat_completion("http://fake", "k", "m", "test", timeout=5, disable_reasoning=False)


def test_the_400_fallback_happens_at_most_once(script, monkeypatch):
    """If the retry without the hint still fails, that failure propagates --
    this is one fallback, not a loop."""
    calls = {"n": 0}

    def always_400(request, timeout):
        calls["n"] += 1
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", always_400)
    with pytest.raises(urllib.error.HTTPError):
        script._http_chat_completion("http://fake", "k", "m", "test", timeout=5, disable_reasoning=True)
    assert calls["n"] == 2, "one attempt with the hint, one retry without, then propagate"


def test_a_429_and_a_400_fallback_can_both_occur_in_one_call(script, monkeypatch):
    """The two retry mechanisms are independent and must compose."""
    sequence = iter(
        [
            lambda: (_ for _ in ()).throw(urllib.error.HTTPError("u", 429, "", {"Retry-After": "0.01"}, None)),
            lambda: (_ for _ in ()).throw(urllib.error.HTTPError("u", 400, "", {}, None)),
            lambda: _FakeResponse({"choices": [{"message": {"content": "final(ok)"}}]}),
        ]
    )

    def flaky(request, timeout):
        return next(sequence)()

    monkeypatch.setattr("urllib.request.urlopen", flaky)
    result = script._http_chat_completion("http://fake", "k", "m", "test", timeout=5, disable_reasoning=True)
    assert result == "final(ok)"


# --------------------------------------------------------------------------
# Grok slug: corrected after a real 404
# --------------------------------------------------------------------------


def test_grok_uses_the_free_suffix_verified_after_a_live_404(script):
    grok_entries = [(name, model) for name, model, _ in script.CANDIDATE_MODELS if name == "grok-4-fast"]
    assert grok_entries == [("grok-4-fast", "x-ai/grok-4-fast:free")]


# --------------------------------------------------------------------------
# Model/engine configuration lives at the top of the file
# --------------------------------------------------------------------------


def test_candidate_models_is_declared_before_any_function_definition(script):
    """The whole point of the reorganisation: config is data, found immediately,
    not logic requiring a read of build_candidates to discover."""
    import inspect

    source = inspect.getsource(script)
    config_position = source.index("CANDIDATE_MODELS")
    first_def_position = source.index("\ndef ")
    assert config_position < first_def_position


def test_every_candidate_models_entry_has_the_expected_shape(script):
    for entry in script.CANDIDATE_MODELS:
        assert len(entry) == 3
        name, slug, disable_reasoning = entry
        assert isinstance(name, str) and name
        assert isinstance(slug, str) and "/" in slug
        assert isinstance(disable_reasoning, bool)


def test_no_duplicate_candidate_names(script):
    names = [name for name, _, _ in script.CANDIDATE_MODELS]
    assert len(names) == len(set(names))


def test_mistral_direct_model_is_a_separate_named_constant(script):
    assert script.MISTRAL_DIRECT_MODEL == "mistral-small-latest"


# --------------------------------------------------------------------------
# Harder, multi-document bench cases
# --------------------------------------------------------------------------


def test_more_than_the_original_six_cases_are_present(script):
    """An 8-way tie at 100%/100% on the original six motivated this."""
    assert len(script.BENCH_CASES) > 6


def test_every_case_document_is_marked_synthetic(script):
    """Phase-one discipline applies to every document this bench sends, not
    only the baseline tier."""
    for case in script.BENCH_CASES:
        for document in case.documents:
            assert document.metadata["data_class"] == "synthetic"


def test_the_cross_document_case_actually_spans_two_documents(script):
    """A case testing multi-document navigation that only supplies one
    document would not test what its name claims."""
    case = next(c for c in script.BENCH_CASES if c.case_id == "cross_document_correlation")
    assert len(case.documents) == 2
    assert case.documents[0].document_id != case.documents[1].document_id


def test_case_ids_are_unique(script):
    ids = [case.case_id for case in script.BENCH_CASES]
    assert len(ids) == len(set(ids))


def test_the_cross_document_case_is_navigable_to_completion(script):
    """Concrete proof the harder case can be solved within budget by a model
    that actually looks at both documents, not just a structural check."""
    from melampo.reasoning.rlm_engine import STOP_FINAL, RlmEngine

    case = next(c for c in script.BENCH_CASES if c.case_id == "cross_document_correlation")
    steps = iter(["describe()", "grep(laboratory)", "final(elevated CRP, from report_3b)"])

    def scripted(prompt):
        return next(steps, "final(fallback)")

    engine = RlmEngine(root_model=scripted, depth=0)
    trajectory = engine.run("t", case.documents, case.question, budget=script._bench_budget())
    assert trajectory.stop_reason == STOP_FINAL


def test_the_negation_case_document_contains_both_an_affirmed_and_a_negated_finding(script):
    """The whole point of this case: it must contain the discrimination it
    claims to test, not just harder-sounding prose."""
    case = next(c for c in script.BENCH_CASES if c.case_id == "negation_discrimination")
    text = case.documents[0].text.lower()
    assert "no pleural effusion" in text
    assert "trace pericardial effusion" in text


def test_the_differential_case_states_a_confirmed_diagnosis_distinct_from_candidates(script):
    case = next(c for c in script.BENCH_CASES if c.case_id == "confirmed_vs_candidate_diagnosis")
    text = case.documents[0].text.lower()
    assert "confirmed" in text
    assert text.index("confirmed") > text.index("differential"), "the confirmation must come after the candidate list"


# --------------------------------------------------------------------------
# Parallel matrix support: one candidate per job
# --------------------------------------------------------------------------


def test_list_candidates_names_every_candidate_including_mistral_direct(script):
    names = script.all_candidate_names()
    assert "mistral-small-3.1" in names
    assert "claude-sonnet-5" in names
    assert len(names) == len(set(names)), "no duplicate names"
    assert len(names) == 1 + len(script.CANDIDATE_MODELS)


def test_build_candidates_with_only_restricts_preflight_to_that_one(script, monkeypatch):
    """What a matrix job needs: run its one assigned candidate without also
    preflighting the other twenty it will never use."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    checked = []

    def fake_preflight(name, endpoint, key, model, disable_reasoning=False):
        checked.append(name)
        return (name == "glm-5", "reachable" if name == "glm-5" else "fake failure")

    monkeypatch.setattr(script, "_preflight", fake_preflight)
    candidates, _skipped, detail = script.build_candidates(only="glm-5")

    assert checked == ["glm-5"], "no other candidate should have been preflighted at all"
    assert "glm-5" in candidates
    assert len(detail) == len(script.all_candidate_names())
    assert "not requested" in detail["claude-sonnet-5"]


def test_build_candidates_without_only_behaves_as_before(script, monkeypatch):
    """The restriction is opt-in; the default full-roster behaviour must be unchanged."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _, _, detail = script.build_candidates()
    assert all("not requested" not in reason for reason in detail.values())


def test_main_dashdash_list_candidates_prints_json_and_exits_zero(script, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["bench", "--list-candidates"])
    exit_code = script.main()
    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == script.all_candidate_names()


def test_main_dashdash_candidate_restricts_the_run(script, monkeypatch, tmp_path):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "bad-key")
    out = tmp_path / "single.json"
    monkeypatch.setattr(sys, "argv", ["bench", "--candidate", "glm-5", "--out", str(out)])

    script.main()

    payload = json.loads(out.read_text())
    assert "glm-5" in payload["preflight"]
    assert "not requested" in payload["preflight"]["claude-sonnet-5"]
