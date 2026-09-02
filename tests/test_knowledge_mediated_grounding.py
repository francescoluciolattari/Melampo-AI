from melampo.evaluation.grounding_judge import (
    VERDICT_KNOWLEDGE_MEDIATED,
    GroundingJudge,
)
from melampo.memory.concept_paths import (
    ConceptEdge,
    InMemoryConceptGraph,
    find_paths,
    shared_mechanisms,
)
from melampo.training.hypothesis_channel import HYPOTHESIS_ROLE, HypothesisChannel
from melampo.training.mechanism_enumeration import MechanismEnumerator


def _clinical_graph() -> InMemoryConceptGraph:
    """A fragment of clinical knowledge, weighted by how well attested each relation is."""
    return InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("pulmonary oedema", "causes", "bibasilar opacities", 0.9, "radiology_ontology"),
            ConceptEdge("congestive cardiac failure", "causes", "pulmonary oedema", 0.9, "cardiology_ontology"),
            ConceptEdge("pneumonia", "causes", "bibasilar opacities", 0.7, "radiology_ontology"),
            ConceptEdge("renal failure", "causes", "pulmonary oedema", 0.6, "nephrology_ontology"),
            ConceptEdge("congestive cardiac failure", "causes", "pleural effusion", 0.8, "cardiology_ontology"),
            ConceptEdge("amyloidosis", "causes", "congestive cardiac failure", 0.2, "rare_disease_registry"),
            ConceptEdge("amyloidosis", "causes", "renal failure", 0.25, "rare_disease_registry"),
        ]
    )


def test_path_search_finds_the_mechanism_between_finding_and_condition():
    graph = _clinical_graph()
    paths = find_paths(graph, "bibasilar opacities", "congestive cardiac failure", max_hops=3)

    assert paths
    best = paths[0]
    assert best.hops == 2
    assert "pulmonary oedema" in [concept.lower() for concept in best.intermediates]
    assert 0.7 < best.strength < 0.9


def test_shared_mechanism_names_the_intermediate_rather_than_asserting_causation():
    graph = _clinical_graph()
    mechanisms = shared_mechanisms(graph, "bibasilar opacities", "congestive cardiac failure")
    assert [item.lower() for item in mechanisms] == ["pulmonary oedema"]


def test_unrelated_concepts_have_no_path():
    graph = _clinical_graph()
    assert find_paths(graph, "bibasilar opacities", "fractured radius", max_hops=3) == []


def test_search_is_bounded_so_the_check_can_still_fail():
    graph = _clinical_graph()
    assert find_paths(graph, "bibasilar opacities", "amyloidosis", max_hops=1) == []
    assert find_paths(graph, "bibasilar opacities", "amyloidosis", max_hops=3)


def test_long_weak_paths_score_lower_than_short_strong_ones():
    graph = _clinical_graph()
    strong = find_paths(graph, "bibasilar opacities", "congestive cardiac failure", max_hops=3)[0]
    weak = find_paths(graph, "bibasilar opacities", "amyloidosis", max_hops=3)[0]
    assert weak.hops > strong.hops
    assert weak.strength < strong.strength


def test_relation_absent_from_the_case_but_present_in_the_graph_is_inference_not_fabrication():
    """The correction: an unsupported relation is not automatically an error."""
    judge = GroundingJudge(concept_graph=_clinical_graph())
    fragments = [
        {"text": "The radiograph shows bibasilar opacities."},
        {"text": "The patient has congestive cardiac failure."},
    ]
    claim = "The bibasilar opacities are caused by congestive cardiac failure."

    assessment = judge.assess(claim, fragments)

    assert assessment.verdict == VERDICT_KNOWLEDGE_MEDIATED
    assert assessment.is_admissible
    assert not assessment.is_grounded
    assert assessment.unsupported_relations == []
    assert assessment.mediated_relations
    assert "pulmonary oedema" in assessment.mediated_relations[0]["description"].lower()


def test_the_mediated_relation_carries_its_path_as_provenance():
    judge = GroundingJudge(concept_graph=_clinical_graph())
    fragments = [
        {"text": "The radiograph shows bibasilar opacities."},
        {"text": "The patient has congestive cardiac failure."},
    ]
    assessment = judge.assess("The bibasilar opacities are caused by congestive cardiac failure.", fragments)

    path = assessment.mediated_relations[0]["path"]
    assert path["kind"] == "concept_graph_path"
    assert path["hops"] == 2
    assert path["edges"][0]["provenance"]


def test_relation_with_no_graph_path_remains_fabrication():
    judge = GroundingJudge(concept_graph=_clinical_graph())
    fragments = [
        {"text": "The radiograph shows bibasilar opacities."},
        {"text": "The patient sustained a fractured radius last year."},
    ]
    assessment = judge.assess("The bibasilar opacities are caused by the fractured radius.", fragments)

    assert "caused by" in assessment.unsupported_relations
    assert assessment.mediated_relations == []
    assert not assessment.is_admissible


def test_without_a_graph_the_judge_keeps_its_previous_conservative_behaviour():
    judge = GroundingJudge()
    fragments = [
        {"text": "The radiograph shows bibasilar opacities."},
        {"text": "The patient has congestive cardiac failure."},
    ]
    assessment = judge.assess("The bibasilar opacities are caused by congestive cardiac failure.", fragments)
    assert "caused by" in assessment.unsupported_relations


def test_enumeration_finds_a_condition_the_case_never_raised():
    enumerator = MechanismEnumerator(graph=_clinical_graph(), max_hops=3)
    hypotheses = enumerator.enumerate(
        findings=["bibasilar opacities", "pleural effusion"],
        candidate_conditions=["congestive cardiac failure", "pneumonia", "amyloidosis", "fractured radius"],
        already_considered=["pneumonia"],
    )

    labels = [item.condition for item in hypotheses]
    assert "congestive cardiac failure" in labels
    assert "pneumonia" not in labels, "an already considered condition is not a hypothesis"
    assert "fractured radius" not in labels, "no path means no hypothesis"


def test_corroboration_by_two_findings_outranks_a_single_link():
    enumerator = MechanismEnumerator(graph=_clinical_graph(), max_hops=3)
    hypotheses = enumerator.enumerate(
        findings=["bibasilar opacities", "pleural effusion"],
        candidate_conditions=["congestive cardiac failure", "amyloidosis"],
    )
    assert hypotheses[0].condition == "congestive cardiac failure"
    assert hypotheses[0].corroboration == 2


def test_novelty_and_support_are_reported_separately():
    enumerator = MechanismEnumerator(graph=_clinical_graph(), max_hops=4)
    hypotheses = {
        item.condition: item
        for item in enumerator.enumerate(
            findings=["bibasilar opacities"],
            candidate_conditions=["congestive cardiac failure", "amyloidosis"],
        )
    }
    obvious = hypotheses["congestive cardiac failure"]
    rare = hypotheses["amyloidosis"]

    assert rare.novelty > obvious.novelty, "the rare route is the novel one"
    assert rare.support < obvious.support, "and it is the weakly supported one"


def test_enumerated_hypotheses_reach_the_differential_only_as_exclusion_hypotheses():
    enumerator = MechanismEnumerator(graph=_clinical_graph(), max_hops=3)
    hypotheses = enumerator.enumerate(
        findings=["bibasilar opacities"],
        candidate_conditions=["congestive cardiac failure", "amyloidosis"],
    )
    channel = HypothesisChannel(max_hypotheses=2)
    result = channel.open_for(
        enumerator.envelopes(hypotheses),
        dynamics={"convergence_index": 0.3, "conflict_load": 0.7},
        risk=0.6,
    )

    assert result["channel_open"] is True
    for hypothesis in result["hypotheses"]:
        assert hypothesis["role"] == HYPOTHESIS_ROLE
        assert hypothesis["usable_as_evidence"] is False
        assert "via" in hypothesis["rationale"]


def test_hypotheses_stay_closed_when_the_differential_has_converged():
    enumerator = MechanismEnumerator(graph=_clinical_graph(), max_hops=3)
    hypotheses = enumerator.enumerate(
        findings=["bibasilar opacities"],
        candidate_conditions=["amyloidosis"],
    )
    channel = HypothesisChannel()
    result = channel.open_for(
        enumerator.envelopes(hypotheses),
        dynamics={"convergence_index": 0.95, "conflict_load": 0.8},
        risk=0.9,
    )
    assert result["channel_open"] is False
