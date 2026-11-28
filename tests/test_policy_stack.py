"""The policy of the case by explicit rules (28 November 2026): no weights, no thresholds, nothing withheld."""

from melampo.reasoning.case_signals import case_signals, declared_severity
from melampo.reasoning.policy_stack import PolicyStack

MEMORY = {"evidence_count": 3, "retrieval_quality": {"memory_backed": True, "fallback_used": False}}
SOURCE = {"provenance": {"source": "test"}}


def _discordances(**kinds):
    return {"count": sum(kinds.values()), "kinds": kinds}


def test_a_quiet_case_is_clear_and_has_no_flags():
    signals = case_signals(SOURCE, _discordances(), MEMORY)
    assert signals["escalate"] is False and signals["reasons"] == [] and signals["flags"] == []
    policy = PolicyStack().evaluate(signals)
    assert policy == {"abstain": False, "allow": True, "escalate": False, "reasons": [], "flags": [],
                      "sentences": [], "decision_band": "clear"}


def test_discordances_and_unread_documents_escalate_each_with_its_sentence():
    signals = case_signals(SOURCE, _discordances(card=2, unread=1), MEMORY)
    assert [r["rule"] for r in signals["reasons"]] == ["discordances_in_the_reading", "document_not_read"]
    assert signals["reasons"][0]["sentence"] == "the reading of the case sent 2 discordances to a person"
    assert signals["reasons"][1]["sentence"] == "1 documents of the case were read by no interpreter"
    policy = PolicyStack().evaluate(signals)
    assert policy["escalate"] is True and policy["abstain"] is False and policy["allow"] is True
    assert policy["decision_band"] == "guarded"


def test_severity_is_a_declared_word_not_a_number():
    assert declared_severity({"clinical_severity": "high"}) == "high"
    assert declared_severity({"clinical_severity": "Alta"}) == "high"
    assert declared_severity({"clinical_severity": 0.9}) == "not_declared"
    assert declared_severity({}) == "not_declared"
    signals = case_signals({**SOURCE, "clinical_severity": "grave"}, _discordances(), MEMORY)
    assert [r["rule"] for r in signals["reasons"]] == ["declared_severity_high"]


def test_no_evidence_and_no_provenance_are_flags_not_escalations():
    fallback = {"evidence_count": 3, "retrieval_quality": {"memory_backed": False, "fallback_used": True}}
    signals = case_signals({}, _discordances(), fallback)
    assert signals["escalate"] is False
    assert [f["rule"] for f in signals["flags"]] == ["no_evidence", "no_provenance"]
    assert PolicyStack().evaluate(signals)["flags"] == ["no_evidence", "no_provenance"]
    empty = {"evidence_count": 0, "retrieval_quality": {"memory_backed": True}}
    assert [f["rule"] for f in case_signals(SOURCE, _discordances(), empty)["flags"]] == ["no_evidence"]
