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
        bundle_keys = case_context.get("bundle_keys", []) if isinstance(case_context, dict) else []
        rehearsal_profile = {
            "rare_case_hint": bool(accepted and len(bundle_keys) <= 2),
            "boundary_case_hint": bool(coherence < 0.95 and risk <= 0.15),
            "contradiction_rehearsal": bool(not accepted or risk > 0.2),
            "revision_bias": "conservative" if risk > 0.15 else "exploratory",
        }
        belief = self.belief_layer.update(
            prior={"accepted": accepted},
            context={"sampled": sampled, "rehearsal_profile": rehearsal_profile},
        )
        return {
            "accepted": accepted,
            "sampled": sampled,
            "rehearsal_profile": rehearsal_profile,
            "belief": belief,
        }
