"""Tests for the bridge between what the RLM reads and what the graph knows."""

from melampo.evaluation.enumeration_bench import differential_graph
from melampo.reasoning.rlm_graph_bridge import (
    ORIGIN_GRAPH,
    ORIGIN_RLM_CONJECTURE,
    bridge,
    findings_from_trajectory,
    predicted_findings_for,
    vet_rlm_claims,
)
from melampo.training.mechanism_enumeration import MechanismEnumerator


class _Trajectory:
    """Stands in for what the RLM produces after reading a case."""

    def __init__(self, fragments, final_answer=None):
        self._fragments = fragments
        self.final_answer = final_answer

    def evidence(self):
        return [{"record_id": f"r{i}", "text": text} for i, text in enumerate(self._fragments)]


_SARCOID_CASE = _Trajectory(
    [
        "CT chest: bilateral hilar lymphadenopathy, no effusion.",
        "Serum calcium elevated; hypercalcaemia confirmed on repeat.",
        "Skin lesions on shins consistent with erythema nodosum.",
    ],
    final_answer="The chest imaging shows bilateral hilar lymphadenopathy.",
)


# --------------------------------------------------------------------------
# Documents to graph: the direction that makes the enumerator usable
# --------------------------------------------------------------------------


def test_findings_are_read_from_the_evidence_not_only_the_final_answer():
    """The answer is one sentence; the fragments are everything the run
    touched. A differential should come from what the case contains."""
    resolved, _ = findings_from_trajectory(_SARCOID_CASE, differential_graph())
    assert "hypercalcaemia" in resolved
    assert "erythema nodosum" in resolved, "present only in a fragment, not in the final answer"


def test_the_full_bridge_produces_a_ranked_differential_from_raw_documents():
    """What neither half could do alone: from document text to a ranked
    differential, with nobody supplying findings or candidates by hand."""
    result = bridge(_SARCOID_CASE, differential_graph())
    assert result.outcome is not None
    assert result.outcome.hypotheses[0].condition == "sarcoidosis"


def test_candidates_come_from_the_graph_not_from_the_caller():
    result = bridge(_SARCOID_CASE, differential_graph())
    assert set(result.candidate_conditions) == {"sarcoidosis", "lymphoma", "tuberculosis"}


def test_a_trajectory_with_nothing_the_graph_recognises_produces_no_false_differential():
    """Silence is the right output when the graph recognises nothing --
    inventing a differential from unrecognised text would be worse than
    saying nothing."""
    result = bridge(_Trajectory(["The patient reports feeling generally unwell."]), differential_graph())
    assert result.candidate_conditions == []
    assert result.outcome is None


# --------------------------------------------------------------------------
# Graph back to documents: a hypothesis must be checkable
# --------------------------------------------------------------------------


def test_the_graph_returns_findings_to_look_for():
    """What a clinician does on forming a hypothesis: look for what it
    predicts. A ranking nobody can act on is not yet useful."""
    result = bridge(_SARCOID_CASE, differential_graph())
    assert result.predicted_findings
    assert "night sweats" in result.predicted_findings


def test_predictions_exclude_findings_the_case_already_shows():
    """Already-observed findings confirm nothing new and would crowd out the
    ones worth asking about."""
    result = bridge(_SARCOID_CASE, differential_graph())
    for observed in result.findings_from_documents:
        assert observed not in result.predicted_findings


def test_no_predictions_are_offered_when_the_enumerator_declined_to_rank():
    """If the graph could not support a conclusion, it has no basis for
    predicting anything either."""
    graph = differential_graph()
    outcome = MechanismEnumerator(graph=graph).run(["periorbital purpura"], ["amyloidosis"])
    assert predicted_findings_for(outcome, graph, ["periorbital purpura"]) == []


# --------------------------------------------------------------------------
# The RLM's own conjectures: neither trusted nor discarded
# --------------------------------------------------------------------------


def test_an_rlm_claim_the_graph_supports_is_marked_grounded():
    """Two findings of the same disease are connected *through* that disease,
    which is a real mediating concept in this graph -- unlike a biochemical
    intermediate, which this fixture deliberately does not model."""
    claims = [("hypercalcaemia", "erythema nodosum", "sarcoidosis")]
    vetted = vet_rlm_claims(claims, differential_graph())
    assert vetted[0].is_grounded is True


def test_an_rlm_claim_the_graph_cannot_support_is_held_as_a_conjecture():
    """The interesting middle: the graph knows both concepts but has no
    connection. Not noise to discard, not a finding to trust -- material for
    ConjectureLedger to hold until confirmations accumulate."""
    claims = [("erythema nodosum", "hypercalcaemia", "shared granulomatous process")]
    vetted = vet_rlm_claims(claims, differential_graph())
    assert vetted[0].is_grounded is False
    assert vetted[0].is_candidate_conjecture is True


def test_a_claim_whose_concepts_the_graph_cannot_resolve_is_not_a_conjecture():
    """Nothing to hold: a claim about concepts the graph has never heard of
    is not new material, it is unverifiable input."""
    claims = [("some unheard-of thing", "another unheard-of thing", "whatever")]
    vetted = vet_rlm_claims(claims, differential_graph())
    assert vetted[0].is_candidate_conjecture is False


def test_rlm_claims_get_the_same_scrutiny_as_any_model_answer():
    """No gentler path for "our own engine's ideas" -- that is how a system
    starts trusting its own output."""
    claims = [("hypercalcaemia", "erythema nodosum", "cosmic ray exposure")]
    vetted = vet_rlm_claims(claims, differential_graph())
    assert vetted[0].is_grounded is False


def test_the_bridge_separates_grounded_claims_from_conjectures():
    result = bridge(
        _SARCOID_CASE,
        differential_graph(),
        rlm_claims=[
            ("hypercalcaemia", "erythema nodosum", "sarcoidosis"),
            ("erythema nodosum", "night sweats", "shared granulomatous process"),
        ],
    )
    assert len(result.grounded_claims) == 1
    assert len(result.conjectures_for_the_ledger) == 1


# --------------------------------------------------------------------------
# Provenance stays straight throughout
# --------------------------------------------------------------------------


def test_every_part_of_the_output_declares_where_it_came_from():
    """A reader downstream must always be able to tell a finding read from a
    document, a concept the graph supplied, and one the RLM merely
    proposed."""
    result = bridge(
        _SARCOID_CASE, differential_graph(), rlm_claims=[("hypercalcaemia", "erythema nodosum", "sarcoidosis")]
    )
    payload = result.as_dict()

    assert payload["findings_from_documents"]
    assert payload["hypotheses"][0]["origin"] == ORIGIN_GRAPH
    assert payload["vetted_claims"][0]["origin"] == ORIGIN_RLM_CONJECTURE


def test_an_injected_enumerator_is_used_rather_than_a_fresh_one():
    """A caller supplying an enumerator bound to a persistent graph must not
    have it silently replaced by one using only the imported layer."""
    graph = differential_graph()
    calls = {"n": 0}

    class _CountingEnumerator(MechanismEnumerator):
        def run(self, findings, candidates, **kwargs):
            calls["n"] += 1
            return super().run(findings, candidates, **kwargs)

    bridge(_SARCOID_CASE, graph, enumerator=_CountingEnumerator(graph=graph))
    assert calls["n"] == 1


def test_as_dict_carries_what_a_reviewer_needs():
    payload = bridge(_SARCOID_CASE, differential_graph()).as_dict()
    for key in ("findings_from_documents", "candidate_conditions", "hypotheses", "predicted_findings"):
        assert key in payload
