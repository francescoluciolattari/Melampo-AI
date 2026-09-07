"""Tests for scripts/merge_bench_results.py, which combines per-candidate
results from a parallel matrix run into the same shape a sequential run
would have produced.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_bench_results.py"


@pytest.fixture
def merge_script():
    spec = importlib.util.spec_from_file_location("merge_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(directory: Path, name: str, payload: dict) -> None:
    (directory / f"{name}.json").write_text(json.dumps(payload))


def _result(name: str, adherence: float, completion: float, accepted=5, rejected=0, near_miss=0.0) -> dict:
    return {
        "model_name": name,
        "runs": 6,
        "adherence": adherence,
        "completion_rate": completion,
        "accepted_lines": accepted,
        "rejected_lines": rejected,
        "near_miss_share": near_miss,
        "near_miss_examples": [],
        "prose_examples": [],
        "stop_reasons": {},
        "mean_iterations_on_completion": 3.0,
        "mean_iterations_on_incompletion": None,
        "budget_bound": False,
    }


# --------------------------------------------------------------------------
# Merging genuine per-candidate results
# --------------------------------------------------------------------------


def test_merging_two_successful_candidates_combines_their_results(merge_script, tmp_path):
    _write(
        tmp_path,
        "a",
        {"results": [_result("a", 1.0, 1.0)], "skipped": [], "preflight": {"a": "reachable"}},
    )
    _write(
        tmp_path,
        "b",
        {"results": [_result("b", 0.5, 0.0)], "skipped": [], "preflight": {"b": "reachable"}},
    )

    payload = merge_script.merge(tmp_path)

    assert {item["model_name"] for item in payload["results"]} == {"a", "b"}
    assert payload["status"] == "completed"
    assert "a meets the adherence target" in payload["verdict"]


def test_the_merged_verdict_matches_what_a_sequential_run_would_have_computed(merge_script, tmp_path):
    """The point of extracting compute_verdict: a parallel run and a
    sequential run must reach the same conclusion from the same numbers."""
    from melampo.evaluation.format_adherence_bench import compute_verdict

    results = [_result("a", 1.0, 1.0), _result("b", 0.5, 0.0)]
    for item in results:
        _write(tmp_path, item["model_name"], {"results": [item], "skipped": [], "preflight": {}})

    payload = merge_script.merge(tmp_path)
    assert payload["verdict"] == compute_verdict(results, merge_script.ADHERENCE_TARGET)


def test_results_are_ranked_in_the_merged_output(merge_script, tmp_path):
    _write(tmp_path, "weak", {"results": [_result("weak", 0.3, 0.0)], "skipped": [], "preflight": {}})
    _write(tmp_path, "strong", {"results": [_result("strong", 1.0, 1.0)], "skipped": [], "preflight": {}})

    payload = merge_script.merge(tmp_path)
    assert [item["model_name"] for item in payload["results"]] == ["strong", "weak"]


# --------------------------------------------------------------------------
# Preflight and skipped merging
# --------------------------------------------------------------------------


def test_skipped_lists_are_combined_and_deduplicated(merge_script, tmp_path):
    _write(tmp_path, "a", {"results": [], "skipped": ["x (bad key)"], "preflight": {}})
    _write(tmp_path, "b", {"results": [], "skipped": ["x (bad key)", "y (404)"], "preflight": {}})

    payload = merge_script.merge(tmp_path)
    assert payload["skipped"] == ["x (bad key)", "y (404)"]


def test_a_genuine_preflight_reason_overrides_a_not_requested_placeholder(merge_script, tmp_path):
    """Each matrix job reports every OTHER candidate as 'not requested'; the
    job actually responsible for a candidate has the real reason, and that
    real reason must win in the merge."""
    _write(
        tmp_path,
        "job_for_a",
        {"results": [], "skipped": [], "preflight": {"a": "reachable", "b": "not requested (running only 'a')"}},
    )
    _write(
        tmp_path,
        "job_for_b",
        {"results": [], "skipped": [], "preflight": {"a": "not requested (running only 'b')", "b": "HTTP 404 Not Found"}},
    )

    payload = merge_script.merge(tmp_path)
    assert payload["preflight"]["a"] == "reachable"
    assert payload["preflight"]["b"] == "HTTP 404 Not Found"


# --------------------------------------------------------------------------
# Robustness: a malformed file must not erase the rest
# --------------------------------------------------------------------------


def test_a_malformed_json_file_is_recorded_as_a_failure_not_a_crash(merge_script, tmp_path):
    (tmp_path / "broken.json").write_text("not valid json{{{")
    _write(tmp_path, "good", {"results": [_result("good", 1.0, 1.0)], "skipped": [], "preflight": {}})

    payload = merge_script.merge(tmp_path)

    assert len(payload["results"]) == 1
    assert payload["results"][0]["model_name"] == "good"
    assert any("broken.json" in item for item in payload["merge_failures"])


def test_a_non_object_top_level_json_is_recorded_as_a_failure(merge_script, tmp_path):
    (tmp_path / "list.json").write_text("[1, 2, 3]")
    payload = merge_script.merge(tmp_path)
    assert any("list.json" in item for item in payload["merge_failures"])


def test_an_empty_directory_produces_no_results_and_a_named_failure(merge_script, tmp_path):
    payload = merge_script.merge(tmp_path)
    assert payload["results"] == []
    assert payload["status"] == "no_candidates"
    assert any("no JSON files found" in item for item in payload["merge_failures"])


def test_main_writes_a_diagnostic_when_the_input_directory_does_not_exist(merge_script, tmp_path, monkeypatch):
    out = tmp_path / "out.json"
    missing = tmp_path / "does_not_exist"
    monkeypatch.setattr(sys, "argv", ["merge", str(missing), "--out", str(out)])

    exit_code = merge_script.main()

    assert exit_code == 1
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["status"] == "crashed"


def test_main_writes_the_merged_file_and_returns_zero_on_success(merge_script, tmp_path, monkeypatch):
    _write(tmp_path, "a", {"results": [_result("a", 1.0, 1.0)], "skipped": [], "preflight": {}})
    out = tmp_path / "merged.json"
    monkeypatch.setattr(sys, "argv", ["merge", str(tmp_path), "--out", str(out)])

    exit_code = merge_script.main()

    assert exit_code == 0
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["models"] == 1
