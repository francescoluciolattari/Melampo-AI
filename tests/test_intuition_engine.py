from melampo.models.quantum_belief_layer import QuantumBeliefLayer
from melampo.reasoning.intuition_engine import IntuitionEngine


def test_intuition_engine_returns_candidate_and_belief_mode():
    engine = IntuitionEngine(belief_layer=QuantumBeliefLayer())
    ranked_evidence = [
        {"rank": 1, "weight": 3, "item": {"source": "semantic_memory"}},
        {"rank": 2, "weight": 2, "item": {"source": "episodic_memory"}},
    ]
    payload = engine.infer(
        case_id="case-1",
        ranked_evidence=ranked_evidence,
        dream={"belief": {"mode": "quantum_like_belief_update"}},
        quantum_allowed=True,
    )
    assert payload["intuition"] == "candidate_1"
    assert payload["belief_update"]["mode"] == "quantum_like_belief_update"
