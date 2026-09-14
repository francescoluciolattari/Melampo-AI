"""Tests for scripts/run_vetting_bench.py -- the runnable counterpart to
melampo.evaluation.vetting_bench, closing the gap found by a direct question
("how do I run this") after the library shipped with no execution path."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_vetting_bench.py"


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("run_vetting_bench", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _correct_model(endpoint, key, model, prompt, *, timeout, disable_reasoning=False, system_prompt=None):
    """Answers any shipped vetting case correctly, matched by its own question
    text rather than a hand-picked keyword list -- the v1 version of this
    mock only recognised four cases and silently failed the other twelve
    once the case set grew, which is exactly the kind of drift a
    fixture-matched mock avoids."""
    from melampo.evaluation.vetting_bench import VETTING_CASES

    for case in VETTING_CASES:
        if case.question[:40] in prompt:
            if case.is_restraint_case:
                return f"{case.factor} | {case.target} | no | none"
            return f"{case.factor} | {case.target} | yes | {case.expected_mechanism}"
    return "unknown"


def _ok_preflight(name, endpoint, key, model, disable_reasoning=False):
    return True, "reachable"


# --------------------------------------------------------------------------
# --list-candidates
# --------------------------------------------------------------------------


def test_list_candidates_defaults_to_the_two_vetting_role_candidates(script, capsys):
    sys_argv_backup = sys.argv
    try:
        sys.argv = ["run_vetting_bench.py", "--list-candidates"]
        exit_code = script.main()
        assert exit_code == 0
        names = json.loads(capsys.readouterr().out)
        assert names == list(script.DEFAULT_ROSTER)
    finally:
        sys.argv = sys_argv_backup


def test_list_candidates_with_roster_restricts_the_list(script, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_vetting_bench.py", "--list-candidates", "--roster", "claude-opus-5"])
    exit_code = script.main()
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == ["claude-opus-5"]


def test_list_candidates_rejects_an_unknown_name(script, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_vetting_bench.py", "--list-candidates", "--roster", "not-a-real-model"])
    exit_code = script.main()
    assert exit_code == 1
    assert "not-a-real-model" in capsys.readouterr().err


def test_gpt_oss_120b_is_a_recognised_name_even_though_not_in_the_shared_registry_check(script, capsys, monkeypatch):
    """gpt-oss-120b IS in the shared CANDIDATE_MODELS registry now, but this
    guards the case where the roster names it explicitly alongside a name
    that predates the registry addition."""
    monkeypatch.setattr(sys, "argv", ["run_vetting_bench.py", "--list-candidates", "--roster", "gpt-oss-120b"])
    exit_code = script.main()
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == ["gpt-oss-120b"]


# --------------------------------------------------------------------------
# Running one candidate: the full path, preflight through scored result
# --------------------------------------------------------------------------


def test_a_reachable_correct_candidate_produces_a_completed_result(script, tmp_path, monkeypatch):
    out = tmp_path / "result.json"
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(script, "_preflight", _ok_preflight)
    monkeypatch.setattr(script, "_http_chat_completion", _correct_model)
    monkeypatch.setattr(sys, "argv", ["run_vetting_bench.py", "--candidate", "claude-opus-5", "--out", str(out)])

    exit_code = script.main()

    assert exit_code == 0
    payload = json.loads(out.read_text())
    assert payload["status"] == "completed"
    assert payload["results"][0]["grounding_rate"] == 1.0


def test_an_unreachable_candidate_still_writes_diagnostics(script, tmp_path, monkeypatch):
    out = tmp_path / "result.json"
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(script, "_preflight", lambda *a, **k: (False, "HTTP 403 Forbidden"))
    monkeypatch.setattr(sys, "argv", ["run_vetting_bench.py", "--candidate", "claude-opus-5", "--out", str(out)])

    exit_code = script.main()

    assert exit_code == 1
    payload = json.loads(out.read_text())
    assert payload["status"] == "no_candidates"
    assert payload["reason"] == "HTTP 403 Forbidden"


def test_a_missing_api_key_is_reported_not_silently_skipped(script, tmp_path, monkeypatch):
    out = tmp_path / "result.json"
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["run_vetting_bench.py", "--candidate", "claude-opus-5", "--out", str(out)])

    exit_code = script.main()

    assert exit_code == 1
    payload = json.loads(out.read_text())
    assert "API key" in payload["reason"]


def test_an_unknown_candidate_name_fails_clearly(script, tmp_path, monkeypatch):
    out = tmp_path / "result.json"
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(sys, "argv", ["run_vetting_bench.py", "--candidate", "not-a-real-model", "--out", str(out)])

    exit_code = script.main()

    assert exit_code == 1
    payload = json.loads(out.read_text())
    assert "unknown candidate" in payload["reason"]


def test_candidate_is_required_outside_list_candidates(script, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_vetting_bench.py"])
    exit_code = script.main()
    assert exit_code == 1
    assert "--candidate is required" in capsys.readouterr().err


# --------------------------------------------------------------------------
# The shared-registry reuse this script exists to guarantee
# --------------------------------------------------------------------------


def test_gpt_oss_120b_resolves_to_the_verified_slug_from_the_shared_registry(script):
    entry = script._find_candidate("gpt-oss-120b")
    assert entry == ("gpt-oss-120b", "openai/gpt-oss-120b", True)


def test_a_slug_correction_in_the_shared_registry_is_picked_up_without_editing_this_script(script):
    """The whole point of importing CANDIDATE_MODELS rather than copying it:
    a correction made in run_format_adherence_bench.py must be visible here
    automatically."""
    import run_format_adherence_bench as base

    for index, (name, model, disable_reasoning) in enumerate(base.CANDIDATE_MODELS):
        if name == "claude-opus-5":
            base.CANDIDATE_MODELS = (
                *base.CANDIDATE_MODELS[:index],
                (name, "anthropic/claude-opus-5-corrected", disable_reasoning),
                *base.CANDIDATE_MODELS[index + 1:],
            )
            break
    try:
        entry = script._find_candidate("claude-opus-5")
        assert entry[1] == "anthropic/claude-opus-5-corrected"
    finally:
        importlib.reload(base)


# --------------------------------------------------------------------------
# Crash safety, same discipline as run_format_adherence_bench.py
# --------------------------------------------------------------------------


def test_an_unforeseen_exception_still_writes_diagnostics(script, tmp_path, monkeypatch):
    out = tmp_path / "result.json"
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def boom(*args, **kwargs):
        raise RuntimeError("never seen before")

    monkeypatch.setattr(script, "_preflight", boom)
    monkeypatch.setattr(sys, "argv", ["run_vetting_bench.py", "--candidate", "claude-opus-5", "--out", str(out)])

    exit_code = script.main()

    assert exit_code == 1
    payload = json.loads(out.read_text())
    assert payload["status"] == "crashed"
    assert "never seen before" in payload["error"]


# --------------------------------------------------------------------------
# The system prompt: three live runs showed both candidates emitting
# document-navigation actions (describe(), grep(...), even wrapping their
# real answer in final(...)) because this script, by reusing
# _http_chat_completion unchanged, was sending run_format_adherence_bench's
# navigation-grammar system prompt to a task with no document to navigate.
# --------------------------------------------------------------------------


def test_the_vetting_bench_sends_its_own_system_prompt_not_the_navigation_one(script, monkeypatch):
    """Three live runs showed both candidates emitting document-navigation
    actions -- describe(), grep(...), even wrapping their real answer in
    final(...) -- because reusing _http_chat_completion unchanged sent
    run_format_adherence_bench's navigation-grammar prompt to a task with no
    document to navigate."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(script, "_preflight", _ok_preflight)

    captured = {}

    def capturing_call(endpoint, key, model, prompt, *, timeout, disable_reasoning=False, system_prompt=None):
        captured["system_prompt"] = system_prompt
        return "chronic kidney disease | renal osteodystrophy | yes | secondary hyperparathyroidism"

    monkeypatch.setattr(script, "_http_chat_completion", capturing_call)

    candidates, _ = script._build_one_candidate("claude-opus-5")
    candidates["claude-opus-5"]("any prompt")

    assert captured["system_prompt"] == script.VETTING_SYSTEM_PROMPT
    assert "you navigate a document environment" not in script.VETTING_SYSTEM_PROMPT.lower()


def test_the_vetting_system_prompt_explicitly_forbids_navigation_actions(script):
    """The exact failure mode a live run exposed: a model wrapping its real
    answer inside final(...) because it believed it was in a navigation
    task. The prompt now says plainly not to, for every verb the navigation
    grammar defines."""
    prompt = script.VETTING_SYSTEM_PROMPT.lower()
    for verb in ("search(", "grep(", "slice(", "describe(", "expand(", "query(", "final("):
        assert verb in prompt


def test_the_navigation_bench_system_prompt_is_unaffected(monkeypatch):
    """The fix must be additive: run_format_adherence_bench's own callers,
    which never pass system_prompt, must keep getting exactly the navigation
    prompt they always did."""
    import json
    import urllib.request

    import run_format_adherence_bench as rfab

    captured = {}

    class _FakeResponse:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "final(ok)"}}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=30):
        captured["system_prompt"] = json.loads(request.data.decode())["messages"][0]["content"]
        return _FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    rfab._http_chat_completion("https://example", "key", "model", "a question", timeout=30)

    assert "you navigate a document environment" in captured["system_prompt"].lower()
