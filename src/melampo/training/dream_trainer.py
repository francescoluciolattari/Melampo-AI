from dataclasses import dataclass

from .counterfactual_sampler import CounterfactualSampler
from .replay_filter import ReplayFilter
from ..models.quantum_belief_layer import QuantumBeliefLayer


@dataclass
class DreamTrainer:
    """Offline dream-replay trainer with optional quantum-like belief updates."""

    replay_filter: ReplayFilter
    sampler: CounterfactualSampler
    belief_layer: QuantumBeliefLayer

    def run(self, case_context: dict, coherence: float, risk: float) -> dict:
        accepted = self.replay_filter.accept(coherence=coherence, risk=risk)
        sampled = self.sampler.sample(case_context)
        belief = self.belief_layer.update(prior={"accepted": accepted}, context=sampled)
        return {
            "accepted": accepted,
            "sampled": sampled,
            "belief": belief,
        }
