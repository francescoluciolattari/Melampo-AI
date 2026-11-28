"""The intuition engine, rapid part (28 November 2026): ``rank_differential``'s candidates, in its order.

Without graph candidates the positions of the ranked evidence stay ``candidate_N`` placeholders, which the
differential engine knows and puts a graph-enumerated hypothesis before.
"""

from melampo.reasoning.intuition_engine import RAPID, IntuitionEngine


def _ranked_evidence(n=3):
    return [{"weight": 1.0, "item": {"source": "test"}} for _ in range(n)]


def _real_candidate(condition="marfan syndrome", specificity_score=0.72):
    return {
        "condition": condition,
        "specificity_score": specificity_score,
        "matched_findings": ["arachnodactyly"],
        "unmatched_findings": [],
        "coverage": 1.0,
        "profile_size": 3,
    }


def test_no_graph_candidates_gives_the_placeholders():
    for graph in (None, []):
        result = IntuitionEngine().infer("c1", _ranked_evidence(), {}, graph_candidates=graph)
        assert [c["label"] for c in result["candidate_scores"]] == ["candidate_1", "candidate_2", "candidate_3"]
        assert result["source"] == "placeholder"
        assert result["intuition"] == "candidate_1"


def test_the_graph_candidates_come_in_rank_differentials_order_with_their_own_score():
    graph = [_real_candidate("marfan syndrome", 0.9), _real_candidate("loeys-dietz syndrome", 0.4)]
    result = IntuitionEngine().infer("c1", _ranked_evidence(), {}, graph_candidates=graph)
    assert [c["label"] for c in result["candidate_scores"]] == ["marfan syndrome", "loeys-dietz syndrome"]
    assert result["candidate_scores"][0]["score"] == 0.9
    assert result["candidate_scores"][0]["matched_findings"] == ["arachnodactyly"]
    assert result["intuition"] == "marfan syndrome"
    assert result["source"] == RAPID


def test_at_most_max_candidates_and_no_hand_set_modes():
    graph = [_real_candidate(f"d{i}", 1.0 - i / 10) for i in range(8)]
    result = IntuitionEngine(max_candidates=5).infer("c1", [], {}, graph_candidates=graph)
    assert len(result["candidate_scores"]) == 5
    assert not {"deductive_filter", "belief_update", "reasoning_mode", "area_dynamics"} & set(result)


def test_the_nexus_alternatives_are_passed_through_and_the_trace_is_short():
    nexus = {"alternative_hypotheses": [{"label": "x", "kind": "enumerated_mechanism"}]}
    engine = IntuitionEngine()
    result = engine.infer("c1", _ranked_evidence(1), nexus)
    assert result["nexus_alternatives"] == nexus["alternative_hypotheses"]
    assert engine.summarize_for_trace(result) == {"intuition": "candidate_1", "source": "placeholder",
                                                   "candidate_count": 1}


# --------------------------------------------------------------------------
# DifferentialEngine's promotion logic (from the previous change) correctly
# stands down once intuition itself has a real label -- no double-promotion
# --------------------------------------------------------------------------


def test_differential_engine_does_not_promote_over_an_already_real_intuition_label():
    from melampo.reasoning.differential_engine import DifferentialEngine

    intuition = {"candidate_scores": [{"label": "marfan syndrome", "score": 7.3}]}
    nexus = {
        "alternative_hypotheses": [
            {
                "label": "loeys-dietz syndrome",
                "kind": "enumerated_mechanism",
                "plausibility": 0.6,
                "paths": [],
            }
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
            {
                "label": "loeys-dietz syndrome",
                "kind": "enumerated_mechanism",
                "plausibility": 0.6,
                "paths": [],
            }
        ]
    }
    result = DifferentialEngine().rank(evidence=["a"], intuition=intuition, nexus=nexus)
    assert result["hypotheses"][0]["label"] == "loeys-dietz syndrome"
    assert result["hypotheses"][0]["source"] == "graph_enumeration"
