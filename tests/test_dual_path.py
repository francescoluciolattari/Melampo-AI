"""The dual path, executed (14 November 2026): the two root models read the case through the recursive engine
and what they read is reconciled with the one-shot retrieval."""

from melampo.reasoning import dual_path as dp


def _reader(answer):
    def model(prompt):
        return "grep(prednisone)" if "(none yet)" in prompt else f"final({answer})"
    return model


DOCS = dp.case_documents("c1", "Prednisone 40 mg daily was started for the flare.", data_class="deidentified")


def test_not_needed_and_not_configured_leave_the_one_shot_retrieval(monkeypatch):
    assert dp.run_dual_path("c1", DOCS, {}, "one_shot")["status"] == "not_needed"
    monkeypatch.delenv("MELAMPO_ROOT_MODELS", raising=False)
    got = dp.run_dual_path("c1", DOCS, {}, "dual_path")
    assert got["status"] == "not_configured" and "MELAMPO_ROOT_MODELS" in got["reason"]
    # a patient's documents never leave this machine
    monkeypatch.setenv("MELAMPO_LOCAL_CHAT_ENDPOINT", "https://api.example.com/v1")
    monkeypatch.setenv("MELAMPO_ROOT_MODELS", "nemotron-3-ultra,gemma-4-31b")
    assert dp.run_dual_path("c1", DOCS, {}, "dual_path")["status"] == "not_configured"
    # a patient's text, not de-identified: the engine's rule refuses it, and the dual path says so
    patient = dp.case_documents("c1", "Prednisone 40 mg daily.")
    got = dp.run_dual_path("c1", patient, {}, "dual_path", models={"a": _reader("x"), "b": _reader("x")})
    assert got["status"] == "refused" and "de-identified" in got["reason"]


def test_the_two_models_read_and_what_they_read_is_reconciled():
    models = {"nemotron-3-ultra": _reader("40 mg daily, c1:report"), "gemma-4-31b": _reader("40 mg daily, c1:report")}
    got = dp.run_dual_path("c1", DOCS, {"query": "q", "evidence": []}, "dual_path", models=models)
    assert got["status"] == "executed" and got["cross_check"]["agree"] and not got["cross_check"]["needs_review"]
    evidence = got["retrieval"]["evidence"]
    assert evidence and all(item["reconciliation"] == "recursive_only_verified" for item in evidence)
    assert got["retrieval"]["retrieval_mode"] and got["reconciliation"]["recall_gain"] == len(evidence)
    # two different answers go to a person with both
    models["gemma-4-31b"] = _reader("no dose is stated")
    got = dp.run_dual_path("c1", DOCS, {}, "dual_path_extended_budget", models=models)
    assert got["cross_check"]["needs_review"] and got["cross_check"]["answers"][1] == "no dose is stated"
