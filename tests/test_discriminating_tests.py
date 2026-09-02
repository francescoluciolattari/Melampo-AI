import math

import pytest

from melampo.evaluation.falsification_program import (
    CLAIM_CORROBORATED,
    CLAIM_OPEN,
    CLAIM_WITHDRAWN,
    ROUTE_EXTERNAL_API,
    ROUTE_ON_PREMISE_ROOT,
    FalsificationProgram,
)
from melampo.memory.concept_paths import ConceptEdge, InMemoryConceptGraph
from melampo.reasoning.discriminating_tests import (
    DiscriminatingTestSelector,
    Investigation,
    WeightedHypothesis,
    entropy,
    expected_information_gain,
)

# --------------------------------------------------------------------------
# Claim registry
# --------------------------------------------------------------------------


def test_withdrawn_claim_is_retained_with_its_reason_not_deleted():
    program = FalsificationProgram()
    claim = program.get("rlm.recursive_helps_only_on_complex_cases")
    assert claim.status == CLAIM_WITHDRAWN
    assert claim.evidence and "cost and latency" in claim.evidence[0]
    assert claim not in program.open_claims()


def test_conditional_claim_gates_only_its_own_route():
    program = FalsificationProgram()
    program.active_routes = {ROUTE_ON_PREMISE_ROOT}
    blocking = {claim.claim_id for claim in program.blocking_claims()}
    assert "rlm.open_weight_root_is_sufficient" in blocking
    assert "privacy.predicate_budget_prevents_reconstruction" not in blocking

    program.active_routes = {ROUTE_EXTERNAL_API}
    blocking = {claim.claim_id for claim in program.blocking_claims()}
    assert "privacy.predicate_budget_prevents_reconstruction" in blocking
    assert "rlm.open_weight_root_is_sufficient" not in blocking


def test_taking_a_route_substitutes_open_questions_rather_than_removing_them():
    program = FalsificationProgram()
    program.active_routes = {ROUTE_ON_PREMISE_ROOT}
    on_premise = len(program.blocking_claims())
    program.active_routes = {ROUTE_EXTERNAL_API}
    external = len(program.blocking_claims())
    assert on_premise == external == 3


def test_dormant_claims_are_reported_not_forgotten():
    program = FalsificationProgram()
    program.active_routes = {ROUTE_EXTERNAL_API}
    dormant = {claim.claim_id for claim in program.dormant_claims()}
    assert "rlm.open_weight_root_is_sufficient" in dormant
    assert program.get("rlm.open_weight_root_is_sufficient").status == CLAIM_OPEN


def test_activating_both_routes_makes_every_conditional_claim_live():
    program = FalsificationProgram()
    program.activate_route(ROUTE_ON_PREMISE_ROOT)
    program.activate_route(ROUTE_EXTERNAL_API)
    assert len(program.blocking_claims()) == 4
    assert program.dormant_claims() == []


def test_unconditional_blocking_claims_are_live_under_any_route():
    program = FalsificationProgram()
    program.active_routes = set()
    blocking = {claim.claim_id for claim in program.blocking_claims()}
    assert blocking == {"rlm.dual_path_beats_single_path", "rlm.disagreement_is_informative"}


def test_resolving_a_claim_removes_it_from_blocking():
    program = FalsificationProgram()
    program.resolve("rlm.dual_path_beats_single_path", CLAIM_CORROBORATED, evidence="run_7")
    assert "rlm.dual_path_beats_single_path" not in {claim.claim_id for claim in program.blocking_claims()}


def test_reformulated_claims_name_what_would_refute_them():
    program = FalsificationProgram()
    disagreement = program.get("rlm.disagreement_is_informative")
    assert "further investigation" in disagreement.statement
    assert "additional investigation" in disagreement.refutation_criterion

    dream = program.get("rlm.dream_hypotheses_add_value")
    assert "indeterminacy" in dream.statement
    assert "no more often than chance" in dream.refutation_criterion


# --------------------------------------------------------------------------
# Information theory
# --------------------------------------------------------------------------


def test_entropy_of_a_certainty_is_zero_and_of_a_fair_coin_is_one_bit():
    assert entropy([1.0]) == 0.0
    assert entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert entropy([0.25, 0.25, 0.25, 0.25]) == pytest.approx(2.0)


def test_zero_probability_outcomes_do_not_break_entropy():
    assert entropy([1.0, 0.0]) == pytest.approx(0.0)
    assert entropy([0.5, 0.5, 0.0]) == pytest.approx(1.0)


def test_a_perfectly_discriminating_test_resolves_a_full_bit():
    prior = {"a": 0.5, "b": 0.5}
    gain = expected_information_gain(prior, {"a": 1.0, "b": 0.0})
    assert gain == pytest.approx(1.0)


def test_a_test_equally_likely_under_every_hypothesis_gains_nothing():
    prior = {"a": 0.5, "b": 0.5}
    assert expected_information_gain(prior, {"a": 0.6, "b": 0.6}) == pytest.approx(0.0, abs=1e-9)


def test_information_gain_is_never_negative():
    prior = {"a": 0.2, "b": 0.3, "c": 0.5}
    for likelihood in (0.0, 0.1, 0.5, 0.9, 1.0):
        gain = expected_information_gain(prior, {"a": likelihood, "b": 0.4, "c": 0.7})
        assert gain >= 0.0


def test_when_one_hypothesis_already_dominates_there_is_little_left_to_learn():
    confident = expected_information_gain({"a": 0.99, "b": 0.01}, {"a": 1.0, "b": 0.0})
    uncertain = expected_information_gain({"a": 0.5, "b": 0.5}, {"a": 1.0, "b": 0.0})
    assert confident < uncertain
    assert confident < 0.1


# --------------------------------------------------------------------------
# Selector over the concept graph
# --------------------------------------------------------------------------


def _graph() -> InMemoryConceptGraph:
    return InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("congestive cardiac failure", "indicates", "bnp assay", 0.9, "cardiology_guideline"),
            ConceptEdge("congestive cardiac failure", "indicates", "chest radiograph", 0.8, "cardiology_guideline"),
            ConceptEdge("pneumonia", "indicates", "chest radiograph", 0.8, "radiology_ontology"),
            ConceptEdge("pneumonia", "indicates", "sputum culture", 0.75, "microbiology_guideline"),
            ConceptEdge("congestive cardiac failure", "causes", "pulmonary oedema", 0.9, "cardiology_ontology"),
        ]
    )


def _selector(**kwargs) -> DiscriminatingTestSelector:
    return DiscriminatingTestSelector(graph=_graph(), **kwargs)


def test_candidate_tests_come_only_from_the_graph():
    selector = _selector()
    hypotheses = [WeightedHypothesis("congestive cardiac failure", 0.5), WeightedHypothesis("pneumonia", 0.5)]
    candidates = selector.candidate_tests(hypotheses)
    assert set(candidates) == {"bnp assay", "chest radiograph", "sputum culture"}
    assert "pulmonary oedema" not in candidates, "a causal relation is not an investigation"


def test_a_test_shared_equally_by_both_hypotheses_is_not_suggested():
    """It may still be clinically necessary; it simply resolves nothing here."""
    selector = _selector()
    ranked = selector.rank(
        [WeightedHypothesis("congestive cardiac failure", 0.5), WeightedHypothesis("pneumonia", 0.5)]
    )
    names = [item.name for item in ranked]
    assert names[0] in {"bnp assay", "sputum culture"}
    assert "chest radiograph" not in names


def test_the_shared_test_can_be_reported_with_a_gain_of_zero_on_request():
    selector = _selector()
    ranked = {
        item.name: item
        for item in selector.rank(
            [WeightedHypothesis("congestive cardiac failure", 0.5), WeightedHypothesis("pneumonia", 0.5)],
            include_non_discriminating=True,
        )
    }
    assert ranked["chest radiograph"].information_gain == pytest.approx(0.0, abs=1e-9)
    assert ranked["bnp assay"].information_gain > ranked["chest radiograph"].information_gain
    assert ranked["chest radiograph"].discriminates_between == (
        "congestive cardiac failure",
        "pneumonia",
    ), "attached to both, so it separates neither"


def test_a_single_hypothesis_yields_no_suggestions():
    selector = _selector()
    assert selector.rank([WeightedHypothesis("pneumonia", 1.0)]) == []


def test_hypotheses_absent_from_the_graph_produce_nothing_rather_than_guesses():
    selector = _selector()
    ranked = selector.rank(
        [WeightedHypothesis("fractured radius", 0.5), WeightedHypothesis("sprained ankle", 0.5)]
    )
    assert ranked == []


def test_burden_is_reported_separately_and_never_folded_into_the_gain():
    selector = _selector(burdens={"bnp assay": 4.0})
    ranked = {item.name: item for item in selector.rank(
        [WeightedHypothesis("congestive cardiac failure", 0.5), WeightedHypothesis("pneumonia", 0.5)]
    )}
    bnp = ranked["bnp assay"]
    assert bnp.burden == 4.0
    assert bnp.gain_per_burden == pytest.approx(bnp.information_gain / 4.0, abs=1e-4)
    assert bnp.information_gain > bnp.gain_per_burden, "raw gain must remain visible"


def test_missing_edges_are_treated_as_silence_not_as_impossibility():
    selector = _selector(absent_likelihood=0.05)
    ranked = {item.name: item for item in selector.rank(
        [WeightedHypothesis("congestive cardiac failure", 0.5), WeightedHypothesis("pneumonia", 0.5)]
    )}
    assert ranked["bnp assay"].likelihoods["pneumonia"] == 0.05


def test_suggestions_carry_provenance_and_decision_support_markers():
    selector = _selector()
    ranked = selector.rank(
        [WeightedHypothesis("congestive cardiac failure", 0.5), WeightedHypothesis("pneumonia", 0.5)]
    )
    payload = ranked[0].as_dict()
    assert payload["decision_support_only"] is True
    assert payload["requires_clinician_judgement"] is True
    assert payload["category"] == "discriminating_test"
    assert payload["provenance"] and payload["provenance"][0]["provenance"]


def test_ranking_is_deterministic():
    hypotheses = [WeightedHypothesis("congestive cardiac failure", 0.5), WeightedHypothesis("pneumonia", 0.5)]
    first = [item.name for item in _selector().rank(hypotheses)]
    second = [item.name for item in _selector().rank(hypotheses)]
    assert first == second


# --------------------------------------------------------------------------
# Integration with the differential payload
# --------------------------------------------------------------------------


def test_hypotheses_are_read_from_the_differential_payload():
    differential = {
        "hypotheses": [
            {"label": "congestive cardiac failure", "score": 0.55},
            {"label": "pneumonia", "score": 0.45},
            {"label": "", "score": 0.9},
            "not a dict",
        ]
    }
    extracted = WeightedHypothesis.from_differential(differential)
    assert [item.label for item in extracted] == ["congestive cardiac failure", "pneumonia"]


def test_suggest_reports_prior_entropy_alongside_the_tests():
    selector = _selector()
    result = selector.suggest(
        {
            "hypotheses": [
                {"label": "congestive cardiac failure", "score": 0.5},
                {"label": "pneumonia", "score": 0.5},
            ]
        }
    )
    assert result["hypothesis_count"] == 2
    assert result["prior_entropy_bits"] == pytest.approx(1.0)
    assert result["discriminating_tests"]
    assert result["decision_support_only"] is True


def test_a_malformed_differential_yields_no_suggestions_rather_than_an_error():
    selector = _selector()
    for payload in ({}, {"hypotheses": None}, {"hypotheses": []}, {"hypotheses": [{}]}):
        result = selector.suggest(payload)
        assert result["discriminating_tests"] == []


def test_unnormalised_scores_are_converted_to_a_distribution():
    selector = _selector()
    result = selector.suggest(
        {
            "hypotheses": [
                {"label": "congestive cardiac failure", "score": 55},
                {"label": "pneumonia", "score": 45},
            ]
        }
    )
    assert 0.0 < result["prior_entropy_bits"] <= 1.0


def test_zero_scores_fall_back_to_a_uniform_prior():
    selector = _selector()
    result = selector.suggest(
        {
            "hypotheses": [
                {"label": "congestive cardiac failure", "score": 0},
                {"label": "pneumonia", "score": 0},
            ]
        }
    )
    assert result["prior_entropy_bits"] == pytest.approx(1.0)


def test_investigation_with_zero_burden_does_not_divide_by_zero():
    item = Investigation(name="x", information_gain=0.5, burden=0.0, discriminates_between=())
    assert item.gain_per_burden == 0.0


def test_gain_matches_a_hand_computed_value():
    """Guard the arithmetic itself, not only its properties."""
    prior = {"a": 0.5, "b": 0.5}
    likelihoods = {"a": 0.8, "b": 0.2}
    positive = 0.5 * 0.8 + 0.5 * 0.2
    posterior_positive = [0.5 * 0.8 / positive, 0.5 * 0.2 / positive]
    posterior_negative = [0.5 * 0.2 / (1 - positive), 0.5 * 0.8 / (1 - positive)]
    expected = 1.0 - (
        positive * entropy(posterior_positive) + (1 - positive) * entropy(posterior_negative)
    )
    assert expected_information_gain(prior, likelihoods) == pytest.approx(expected)
    assert expected == pytest.approx(1.0 - -(0.8 * math.log2(0.8) + 0.2 * math.log2(0.2)), abs=1e-9)
