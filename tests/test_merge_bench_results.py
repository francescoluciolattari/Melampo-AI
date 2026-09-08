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


# --------------------------------------------------------------------------
# Regression: the exact download-artifact@v4 layout, filename collision
# --------------------------------------------------------------------------


def test_merge_finds_results_nested_in_per_artifact_subdirectories(merge_script, tmp_path):
    """This is the real shape actions/download-artifact@v4 produces with
    merge-multiple: false (its default): each artifact in its own
    subdirectory, every one containing an identically-named file. A prior
    version of the workflow flattened these with a shell cp step first,
    which silently collided on the repeated filename and kept only the last
    one copied -- this test is against merge() being pointed directly at
    the nested layout instead, with no flattening step at all."""
    names = ["llama-4-maverick", "gemma-4-31b", "gemma-3-27b", "mistral-large-openrouter"]
    for index, name in enumerate(names):
        subdir = tmp_path / f"candidate-result-{index}"
        subdir.mkdir()
        (subdir / "result.json").write_text(
            json.dumps({"results": [_result(name, 1.0, 1.0)], "skipped": [], "preflight": {name: "reachable"}})
        )

    payload = merge_script.merge(tmp_path)

    assert {item["model_name"] for item in payload["results"]} == set(names)
    assert len(payload["results"]) == 4, "all four, not just the last one a flattening cp would have kept"


def test_a_flattening_copy_step_would_have_lost_three_of_four_results(merge_script, tmp_path):
    """Documents the bug this fixes by reproducing what the old shell step
    did, so the contrast with the test above is explicit rather than
    implicit."""
    nested = tmp_path / "nested"
    nested.mkdir()
    names = ["a", "b", "c", "d"]
    for index, name in enumerate(names):
        subdir = nested / f"candidate-result-{index}"
        subdir.mkdir()
        (subdir / "result.json").write_text(
            json.dumps({"results": [_result(name, 1.0, 1.0)], "skipped": [], "preflight": {}})
        )

    flattened = tmp_path / "flattened"
    flattened.mkdir()
    for path in sorted(nested.rglob("result.json")):
        # Reproduces "cp {} merged_input/" for every match: same destination
        # filename every time, so each copy overwrites the last.
        (flattened / "result.json").write_bytes(path.read_bytes())

    old_way_payload = merge_script.merge(flattened)
    assert len(old_way_payload["results"]) == 1, "the collision this fix removes"

    new_way_payload = merge_script.merge(nested)
    assert len(new_way_payload["results"]) == 4, "merge() pointed at the nested layout directly loses nothing"


def test_source_files_use_relative_paths_not_bare_names_when_nested(merge_script, tmp_path):
    """Two files both literally named result.json in different
    subdirectories must stay distinguishable in the report."""
    for index in range(2):
        subdir = tmp_path / f"candidate-result-{index}"
        subdir.mkdir()
        (subdir / "result.json").write_text(json.dumps({"results": [], "skipped": [], "preflight": {}}))

    payload = merge_script.merge(tmp_path)

    assert len(set(payload["source_files"])) == 2, "bare names would collide; relative paths must not"
    assert all("/" in item for item in payload["source_files"])
