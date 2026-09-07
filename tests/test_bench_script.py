"""Tests for the bench runner script itself, not the library it calls.

The workflow's usefulness depends on these behaviours, and every one of them
was broken or absent in a version that looked correct: the results file was
not written on the failure path, so the artifact that should explain a failed
run was empty exactly when it mattered most.
"""

import importlib.util
import json
import sys
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
    assert len(detail) == 9, "all nine candidates accounted for, none silently dropped"


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


def test_the_bench_budget_is_tighter_than_the_engine_default(script):
    """Three one-fact questions about a two-sentence document need no more."""
    from melampo.reasoning.rlm_engine import Budget

    bench_budget = script._bench_budget()
    default = Budget()
    assert bench_budget.max_iterations < default.max_iterations
    assert bench_budget.max_wall_clock_seconds < default.max_wall_clock_seconds


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
