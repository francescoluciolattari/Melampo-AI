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
        filter_assessment = self.replay_filter.assess(coherence=coherence, risk=risk)
        accepted = filter_assessment["accepted"]
        sampled = self.sampler.sample(case_context)
        case_context = case_context or {}
        bundle_keys = case_context.get("bundle_keys", [])
        base_label = case_context.get("case_id", "case")
        exposures = case_context.get("exposures", {})
        report_text = case_context.get("report_text", "")
        patient_complaints = case_context.get("patient_complaints", "")
        area_dynamics = case_context.get("area_dynamics", {})
        mismatch_score = float(area_dynamics.get("mismatch_score", 0.0))
        coherence_pairs = area_dynamics.get("coherence_pairs", [])
        variant_focus = sampled.get("variant_focus", "context")

        rehearsal_profile = {
            "rare_case_hint": bool(accepted and len(bundle_keys) <= 2),
            "boundary_case_hint": bool(coherence < 0.95 and risk <= 0.15),
            "contradiction_rehearsal": bool((not accepted) or risk > 0.2 or mismatch_score > 0.6),
            "revision_bias": "conservative" if (risk > 0.15 or mismatch_score > 0.4) else "exploratory",
            "post_error_adjustment": "re-rank_alternatives" if ((not accepted) or risk > 0.2 or mismatch_score > 0.6) else "stabilize_primary",
            "coherence_guidance": "multimodal_support" if coherence_pairs else "single_stream",
            "replay_mode": filter_assessment["replay_mode"],
            "acceptance_score": filter_assessment["acceptance_score"],
            "variant_focus": variant_focus,
        }

        alternative_hypotheses = [
            {
                "label": f"{base_label}_alt_1",
                "kind": "rare_case" if rehearsal_profile["rare_case_hint"] else "adjacent_case",
                "focus": "epidemiology" if exposures else variant_focus,
            },
            {
                "label": f"{base_label}_alt_2",
                "kind": "boundary_case" if rehearsal_profile["boundary_case_hint"] else "counterfactual_case",
                "focus": "language_listening" if (report_text or patient_complaints) else variant_focus,
            },
        ]
        if rehearsal_profile["contradiction_rehearsal"]:
            alternative_hypotheses.append(
                {
                    "label": f"{base_label}_alt_3",
                    "kind": "contradiction_revision",
                    "focus": "multi_area_recheck",
                }
            )
        if mismatch_score > 0.6:
            alternative_hypotheses.append(
                {
                    "label": f"{base_label}_alt_4",
                    "kind": "mismatch_resolution",
                    "focus": "cross_area_alignment",
                }
            )

        belief = self.belief_layer.update(
            prior={"accepted": accepted},
            context={
                "sampled": sampled,
                "filter_assessment": filter_assessment,
                "rehearsal_profile": rehearsal_profile,
                "alternative_hypotheses": alternative_hypotheses,
                "area_dynamics": area_dynamics,
            },
        )
        return {
            "accepted": accepted,
            "filter_assessment": filter_assessment,
            "sampled": sampled,
            "rehearsal_profile": rehearsal_profile,
            "alternative_hypotheses": alternative_hypotheses,
            "belief": belief,
        }
