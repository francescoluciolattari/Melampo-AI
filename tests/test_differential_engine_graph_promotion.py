"""Tests for promoting a real, graph-grounded hypothesis over IntuitionEngine's
placeholder labels in DifferentialEngine.rank().

IntuitionEngine's candidate labels are literally "candidate_1", "candidate_2"
-- indices into ranked evidence, not diagnosis names -- until the functional
areas feeding it produce real clinical signal. Wired here so the primary
hypothesis slot is never worse than a placeholder while that gap remains,
without discarding intuition's own contribution (demoted, not dropped) or
duplicating a promoted hypothesis inside its own alternatives list.
"""

from melampo.reasoning.differential_engine import (
    DifferentialEngine,
    _best_graph_hypothesis,
)


def _real_hypothesis(label="marfan syndrome", plausibility=0.82, paths=None):
    return {
        "label": label,
        "kind": "enumerated_mechanism",
        "focus": "cardiovascular",
        "novelty": 0.4,
        "plausibility": plausibility,
        "guaranteed": False,
        "corroboration": 2,
        "paths": paths if paths is not None else [{"hops": 2, "via": ["aortic root aneurysm"]}],
        "density": 0.6,
    }


def _rehearsal_hypothesis(label="alt_1"):
    return {"label": label, "kind": "rare_case", "focus": "epidemiology"}


def _placeholder_intuition(label="candidate_1", score=0.7):
    return {"candidate_scores": [{"label": label, "score": score}], "deductive_filter": {"reasoning_mode": "rapid_intuition"}}


def _real_intuition(label="marfan syndrome", score=0.9):
    """The rarer case where IntuitionEngine's own contradiction_revision
    branch already carried a real condition name through -- must not be
    treated as a placeholder needing promotion."""
    return {"candidate_scores": [{"label": label, "score": score}], "deductive_filter": {"reasoning_mode": "rapid_intuition"}}


# --------------------------------------------------------------------------
# _best_graph_hypothesis: picks the real one, by plausibility, ignores rehearsal labels
# --------------------------------------------------------------------------


def test_finds_the_only_real_hypothesis_among_rehearsal_labels():
    alternatives = [_rehearsal_hypothesis("a"), _real_hypothesis("marfan syndrome"), _rehearsal_hypothesis("b")]
    result = _best_graph_hypothesis(alternatives)
    assert result["label"] == "marfan syndrome"


def test_returns_none_when_every_alternative_is_a_rehearsal_label():
    alternatives = [_rehearsal_hypothesis("a"), _rehearsal_hypothesis("b")]
    assert _best_graph_hypothesis(alternatives) is None


def test_returns_none_for_an_empty_list():
    assert _best_graph_hypothesis([]) is None


def test_picks_the_highest_plausibility_among_several_real_hypotheses():
    alternatives = [
        _real_hypothesis("low", plausibility=0.2),
        _real_hypothesis("high", plausibility=0.9),
        _real_hypothesis("mid", plausibility=0.5),
    ]
    assert _best_graph_hypothesis(alternatives)["label"] == "high"


# --------------------------------------------------------------------------
# DifferentialEngine.rank(): the actual promotion behaviour
# --------------------------------------------------------------------------


def test_a_real_hypothesis_is_promoted_over_a_placeholder_primary():
    engine = DifferentialEngine()
    result = engine.rank(
        evidence=["finding a", "finding b"],
        intuition=_placeholder_intuition(),
        dream={"alternative_hypotheses": [_real_hypothesis("marfan syndrome", plausibility=0.82)]},
    )
    assert result["hypotheses"][0]["label"] == "marfan syndrome"
    assert result["hypotheses"][0]["source"] == "graph_enumeration"


def test_the_demoted_intuition_hypothesis_is_kept_not_dropped():
    engine = DifferentialEngine()
    result = engine.rank(
        evidence=["a"], intuition=_placeholder_intuition("candidate_1"),
        dream={"alternative_hypotheses": [_real_hypothesis()]},
    )
    labels = [h["label"] for h in result["hypotheses"]]
    assert "candidate_1" in labels
    demoted = next(h for h in result["hypotheses"] if h["label"] == "candidate_1")
    assert demoted["source"] == "intuition_engine"
    assert demoted["hypothesis_type"] == "revision_alternative"


def test_a_real_intuition_label_is_not_treated_as_a_placeholder():
    """The rarer case where IntuitionEngine's own contradiction_revision
    branch already carried a real condition name -- must be left as the
    primary hypothesis, not overridden."""
    engine = DifferentialEngine()
    result = engine.rank(
        evidence=["a"], intuition=_real_intuition("marfan syndrome"),
        dream={"alternative_hypotheses": [_real_hypothesis("loeys-dietz syndrome", plausibility=0.95)]},
    )
    assert result["hypotheses"][0]["label"] == "marfan syndrome"
    assert result["hypotheses"][0]["source"] == "intuition_engine"


def test_no_real_hypotheses_available_leaves_the_placeholder_as_before():
    engine = DifferentialEngine()
    result = engine.rank(
        evidence=["a"], intuition=_placeholder_intuition("candidate_1"),
        dream={"alternative_hypotheses": [_rehearsal_hypothesis("alt_1")]},
    )
    assert result["hypotheses"][0]["label"] == "candidate_1"
    assert result["hypotheses"][0]["source"] == "intuition_engine"


def test_no_dream_data_at_all_leaves_the_placeholder_as_before():
    engine = DifferentialEngine()
    result = engine.rank(evidence=["a"], intuition=_placeholder_intuition("candidate_1"), dream={})
    assert result["hypotheses"][0]["label"] == "candidate_1"


def test_the_promoted_hypothesis_is_never_duplicated_in_its_own_alternatives_list():
    """The real defect this guards against: a first version appended every
    dream alternative to the list below the primary, including the one just
    promoted to primary -- listing it twice."""
    real = _real_hypothesis("marfan syndrome", plausibility=0.82)
    engine = DifferentialEngine()
    result = engine.rank(
        evidence=["a"], intuition=_placeholder_intuition(),
        dream={"alternative_hypotheses": [real, _rehearsal_hypothesis("alt_1")]},
    )
    marfan_entries = [h for h in result["hypotheses"] if h["label"] == "marfan syndrome"]
    assert len(marfan_entries) == 1


def test_a_promoted_hypothesis_carries_its_real_graph_paths_as_provenance():
    paths = [{"hops": 2, "via": ["aortic root aneurysm", "ectopia lentis"]}]
    engine = DifferentialEngine()
    result = engine.rank(
        evidence=["a"], intuition=_placeholder_intuition(),
        dream={"alternative_hypotheses": [_real_hypothesis(paths=paths)]},
    )
    assert result["hypotheses"][0]["paths"] == paths


def test_other_real_hypotheses_still_appear_as_ordinary_alternatives():
    """Only the single best one is promoted -- the rest of the real
    candidates remain visible further down, not discarded."""
    engine = DifferentialEngine()
    result = engine.rank(
        evidence=["a"], intuition=_placeholder_intuition(),
        dream={
            "alternative_hypotheses": [
                _real_hypothesis("best", plausibility=0.9),
                _real_hypothesis("second", plausibility=0.5),
            ]
        },
    )
    labels = [h["label"] for h in result["hypotheses"]]
    assert "best" in labels and "second" in labels
    assert labels[0] == "best"
