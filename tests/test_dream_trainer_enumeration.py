from melampo.memory.concept_paths import ConceptEdge, InMemoryConceptGraph
from melampo.models.quantum_belief_layer import QuantumBeliefLayer
from melampo.training.counterfactual_sampler import CounterfactualSampler
from melampo.training.dream_trainer import DreamTrainer
from melampo.training.mechanism_enumeration import MechanismEnumerator
from melampo.training.replay_filter import ReplayFilter

PROFILE = {"rare_case_hint": False, "boundary_case_hint": False, "contradiction_rehearsal": False}


def _graph() -> InMemoryConceptGraph:
    return InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("bibasilar opacities", "caused_by", "pulmonary oedema", 0.9, "radiology"),
            ConceptEdge("pulmonary oedema", "caused_by", "cardiac failure", 0.9, "cardiology"),
            ConceptEdge("pleural effusion", "caused_by", "cardiac failure", 0.8, "cardiology"),
        ]
    )


def _sparse_graph() -> InMemoryConceptGraph:
    return InMemoryConceptGraph.from_edges(
        [ConceptEdge.unknown("bibasilar opacities", "caused_by", "unmapped")]
    )


def _trainer(graph=None, **kwargs) -> DreamTrainer:
    enumerator = MechanismEnumerator(graph=graph, max_hops=3, **kwargs) if graph else None
    return DreamTrainer(ReplayFilter(), CounterfactualSampler(), QuantumBeliefLayer(), enumerator=enumerator)


def _context(trainer: DreamTrainer, **case):
    payload = {
        "case_id": "c1",
        "findings": ["bibasilar opacities", "pleural effusion"],
        "candidate_conditions": ["cardiac failure", "pneumonia"],
        **case,
    }
    return trainer._runtime_context(payload, coherence=0.8, risk=0.6)


def _hypotheses(trainer: DreamTrainer, **case):
    return trainer._alternative_hypotheses(_context(trainer, **case), PROFILE)


# --------------------------------------------------------------------------
# Hypotheses are found, not written
# --------------------------------------------------------------------------


def test_hypotheses_are_conditions_reached_through_the_graph():
    """Previously these were f-string labels: base_label + '_alt_1'."""
    hypotheses = _hypotheses(_trainer(_graph()))
    assert [item["label"] for item in hypotheses] == ["cardiac failure"]
    assert hypotheses[0]["kind"] == "enumerated_mechanism"


def test_each_hypothesis_carries_the_path_that_produced_it():
    hypothesis = _hypotheses(_trainer(_graph()))[0]
    assert hypothesis["paths"]
    assert hypothesis["paths"][0]["kind"] == "concept_graph_path"
    assert hypothesis["paths"][0]["edges"][0]["provenance"]


def test_novelty_and_plausibility_are_reported_separately():
    hypothesis = _hypotheses(_trainer(_graph()))[0]
    assert "novelty" in hypothesis
    assert "plausibility" in hypothesis
    assert "guaranteed" in hypothesis


def test_corroboration_by_two_findings_is_recorded():
    assert _hypotheses(_trainer(_graph()))[0]["corroboration"] == 2


def test_a_condition_already_considered_is_not_proposed():
    hypotheses = _hypotheses(_trainer(_graph()), already_considered=["cardiac failure"])
    assert [item["label"] for item in hypotheses] == []


# --------------------------------------------------------------------------
# Register switch survives the wiring
# --------------------------------------------------------------------------


def test_a_sparse_neighbourhood_yields_questions_not_hypotheses():
    """The branch changes register rather than falling silent."""
    hypotheses = _hypotheses(
        _trainer(_sparse_graph()),
        findings=["bibasilar opacities"],
        candidate_conditions=["amyloidosis"],
    )
    assert hypotheses
    assert all(item["kind"] == "knowledge_gap_question" for item in hypotheses)
    assert all(item["clinical_use"] is False for item in hypotheses)
    assert "question" in hypotheses[0]


def test_the_local_density_travels_with_the_output():
    assert "density" in _hypotheses(_trainer(_graph()))[0]


# --------------------------------------------------------------------------
# Backward compatibility
# --------------------------------------------------------------------------


def test_without_an_enumerator_the_rehearsal_labels_are_unchanged():
    labels = [item["label"] for item in _hypotheses(_trainer())]
    assert labels == ["c1_alt_1", "c1_alt_2"]


def test_a_case_without_findings_falls_back_to_rehearsal_labels():
    """No findings to enumerate from is not a graph problem."""
    labels = [item["label"] for item in _hypotheses(_trainer(_graph()), findings=[])]
    assert labels == ["c1_alt_1", "c1_alt_2"]


def test_a_case_without_candidate_conditions_falls_back():
    labels = [item["label"] for item in _hypotheses(_trainer(_graph()), candidate_conditions=[])]
    assert labels == ["c1_alt_1", "c1_alt_2"]
