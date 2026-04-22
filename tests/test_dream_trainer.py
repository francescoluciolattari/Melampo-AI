from melampo.models.quantum_belief_layer import QuantumBeliefLayer
from melampo.training.counterfactual_sampler import CounterfactualSampler
from melampo.training.dream_trainer import DreamTrainer
from melampo.training.replay_filter import ReplayFilter


def test_dream_trainer_runs():
    trainer = DreamTrainer(
        replay_filter=ReplayFilter(min_coherence=0.7, max_risk=0.3),
        sampler=CounterfactualSampler(),
        belief_layer=QuantumBeliefLayer(),
    )
    result = trainer.run(case_context={"case_id": "x", "bundle_keys": ["a", "b"]}, coherence=0.9, risk=0.1)
    assert result["accepted"] is True
    assert result["belief"]["mode"] == "quantum_like_belief_update"
    assert "rehearsal_profile" in result
    assert result["rehearsal_profile"]["revision_bias"] in ["exploratory", "conservative"]
