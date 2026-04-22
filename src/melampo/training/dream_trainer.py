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
        base_label = case_context.get("case_id", "case") if isinstance(case_context, dict) else "case"
        rehearsal_profile = {
            "rare_case_hint": bool(accepted and len(bundle_keys) <= 2),
            "boundary_case_hint": bool(coherence < 0.95 and risk <= 0.15),
            "contradiction_rehearsal": bool(not accepted or risk > 0.2),
            "revision_bias": "conservative" if risk > 0.15 else "exploratory",
            "post_error_adjustment": "re-rank_alternatives" if (not accepted or risk > 0.2) else "stabilize_primary",
        }
        alternative_hypotheses = [
            {"label": f"{base_label}_alt_1", "kind": "rare_case" if rehearsal_profile["rare_case_hint"] else "adjacent_case"},
            {"label": f"{base_label}_alt_2", "kind": "boundary_case" if rehearsal_profile["boundary_case_hint"] else "counterfactual_case"},
        ]
        belief = self.belief_layer.update(
            prior={"accepted": accepted},
            context={
                "sampled": sampled,
                "rehearsal_profile": rehearsal_profile,
                "alternative_hypotheses": alternative_hypotheses,
            },
        )
        return {
            "accepted": accepted,
            "sampled": sampled,
            "rehearsal_profile": rehearsal_profile,
            "alternative_hypotheses": alternative_hypotheses,
            "belief": belief,
        }
