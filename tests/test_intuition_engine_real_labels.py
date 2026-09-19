"""Tests for IntuitionEngine producing real diagnosis labels via rank_differential,
instead of positional "candidate_N" placeholders.

Traced to its root before this fix: SemanticMemoryStore ships empty in
app.py's build_default_runtime(), so MemoryRetriever.retrieve() always
falls through to three hard-coded fake evidence items
(grounding_score 0.58/0.46/0.5, values like "analogy_for:<query>"), which
IntuitionEngine then labels candidate_1/candidate_2/candidate_3. This fix
does not touch that retrieval gap -- it gives IntuitionEngine a second,
independent source of real content (rank_differential, already built and
tested this session, never previously wired into production) that replaces
the placeholder labels whenever real findings are available, regardless of
what MemoryRetriever itself returns.
"""

from melampo.models.quantum_belief_layer import QuantumBeliefLayer
from melampo.reasoning.intuition_engine import IntuitionEngine


def _engine():
    return IntuitionEngine(belief_layer=QuantumBeliefLayer())


def _ranked_evidence(n=3):
    return [{"weight": 1.0, "item": {"source": "test"}} for _ in range(n)]


def _real_candidate(condition="marfan syndrome", specificity_score=0.72):
    return {"condition": condition, "specificity_score": specificity_score, "matched_findings": [], "unmatched_findings": [], "coverage": 1.0, "profile_size": 3}


# --------------------------------------------------------------------------
# Without graph_candidates: exactly the previous behaviour, no regression
# --------------------------------------------------------------------------


def test_no_graph_candidates_produces_the_original_placeholder_labels():
    result = _engine().infer(case_id="c1", ranked_evidence=_ranked_evidence(), nexus={}, quantum_allowed=False)
    labels = [c["label"] for c in result["inductive_candidates"]]
    assert labels == ["candidate_1", "candidate_2", "candidate_3"]


def test_an_empty_graph_candidates_list_behaves_identically_to_none():
    result = _engine().infer(
        case_id="c1", ranked_evidence=_ranked_evidence(), nexus={}, quantum_allowed=False, graph_candidates=[]
    )
    labels = [c["label"] for c in result["inductive_candidates"]]
    assert labels == ["candidate_1", "candidate_2", "candidate_3"]


# --------------------------------------------------------------------------
# With graph_candidates: real labels replace placeholders, one-for-one
# --------------------------------------------------------------------------


def test_a_real_candidate_replaces_the_first_placeholder():
    result = _engine().infer(
        case_id="c1", ranked_evidence=_ranked_evidence(), nexus={}, quantum_allowed=False,
        graph_candidates=[_real_candidate("marfan syndrome")],
    )
    labels = [c["label"] for c in result["inductive_candidates"]]
    assert labels[0] == "marfan syndrome"
    assert labels[1:] == ["candidate_2", "candidate_3"]


def test_fewer_real_candidates_than_evidence_slots_fills_only_what_it_has():
    result = _engine().infer(
        case_id="c1", ranked_evidence=_ranked_evidence(), nexus={}, quantum_allowed=False,
        graph_candidates=[_real_candidate("a"), _real_candidate("b")],
    )
    labels = [c["label"] for c in result["inductive_candidates"]]
    assert labels == ["a", "b", "candidate_3"]


def test_the_real_candidates_own_specificity_score_becomes_the_support_weight():
    result = _engine().infer(
        case_id="c1", ranked_evidence=_ranked_evidence(), nexus={}, quantum_allowed=False,
        graph_candidates=[_real_candidate("marfan syndrome", specificity_score=0.91)],
    )
    assert result["inductive_candidates"][0]["support_weight"] == 0.91


def test_a_real_label_flows_through_to_rapid_intuition():
    """rapid_intuition reads inductive_candidates[0]["label"] directly --
    unchanged downstream logic, now fed a real name."""
    result = _engine().infer(
        case_id="c1", ranked_evidence=_ranked_evidence(), nexus={}, quantum_allowed=False,
        graph_candidates=[_real_candidate("marfan syndrome")],
    )
    assert result["rapid_intuition"] == "marfan syndrome"


def test_the_source_field_still_comes_from_ranked_evidence_not_graph_candidates():
    """Which item in the retrieval structure this candidate came from is
    orthogonal to whether its name is now known -- unaffected by this fix."""
    evidence = [{"weight": 1.0, "item": {"source": "semantic_memory"}}]
    result = _engine().infer(
        case_id="c1", ranked_evidence=evidence, nexus={}, quantum_allowed=False,
        graph_candidates=[_real_candidate("marfan syndrome")],
    )
    assert result["inductive_candidates"][0]["source"] == "semantic_memory"


# --------------------------------------------------------------------------
# DifferentialEngine's promotion logic (from the previous change) correctly
# stands down once intuition itself has a real label -- no double-promotion
# --------------------------------------------------------------------------


def test_differential_engine_does_not_promote_over_an_already_real_intuition_label():
    from melampo.reasoning.differential_engine import DifferentialEngine

    intuition = {"candidate_scores": [{"label": "marfan syndrome", "score": 7.3}]}
    nexus = {
        "alternative_hypotheses": [
            {"label": "loeys-dietz syndrome", "kind": "enumerated_mechanism", "plausibility": 0.6, "paths": []}
        ]
    }
    result = DifferentialEngine().rank(evidence=["a"], intuition=intuition, nexus=nexus)
    assert result["hypotheses"][0]["label"] == "marfan syndrome"
    assert result["hypotheses"][0]["source"] == "intuition_engine"


def test_differential_engine_still_promotes_when_intuition_has_no_graph_candidates_available():
    """The safety net from the previous change remains intact for exactly
    the case this fix does not cover: findings present, but rank_differential
    found nothing (e.g. no candidates share a finding with the graph)."""
    from melampo.reasoning.differential_engine import DifferentialEngine

    intuition = {"candidate_scores": [{"label": "candidate_1", "score": 7.3}]}
    nexus = {
        "alternative_hypotheses": [
            {"label": "loeys-dietz syndrome", "kind": "enumerated_mechanism", "plausibility": 0.6, "paths": []}
        ]
    }
    result = DifferentialEngine().rank(evidence=["a"], intuition=intuition, nexus=nexus)
    assert result["hypotheses"][0]["label"] == "loeys-dietz syndrome"
    assert result["hypotheses"][0]["source"] == "graph_enumeration"
