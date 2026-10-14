"""Review decisions become training rows only when pseudonymised and confirmed by a second person."""

import json

import pytest

from melampo.evaluation.review_feedback import FeedbackStore, ReviewRecord


def record(**over):
    s = "Lisi delle aderenze e liberazione del cieco e dell'appendice."
    base = dict(report_id="p-001", sentence=s, mention="appendice", start=s.index("appendice"), language="it",
                decided_by="md-7", roles=["procedure_site", "structure"], site=True, structure="appendix",
                pseudonymised=True, system={"reader": "structure", "model": "procedure_site"})
    base.update(over)
    return ReviewRecord(**base)


def test_only_confirmed_records_train(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    store.add(record())
    store.add(record(report_id="p-002", confirmed_by="md-3"))
    store.add(record(report_id="p-003", confirmed_by="md-7"))       # confirmed by the same person: not enough
    rows = store.training_rows()
    assert [r["report_id"] for r in rows] == ["p-002"]
    answer = json.loads(rows[0]["completion"][0]["content"])
    assert answer["roles"] == ["procedure_site", "structure"] and 0 <= rows[0]["fold"] < 5


def test_text_that_is_not_pseudonymised_is_refused(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    with pytest.raises(ValueError, match="pseudonymised"):
        store.add(record(pseudonymised=False))
    with pytest.raises(ValueError, match="site and roles"):
        store.add(record(site=False))
