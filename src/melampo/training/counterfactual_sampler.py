from dataclasses import dataclass


@dataclass
class CounterfactualSampler:
    """Generate lightweight counterfactual case variants for replay and review."""

    def _infer_focus(self, case_context: dict) -> str:
        exposures = case_context.get("exposures", {}) if isinstance(case_context, dict) else {}
        report_text = case_context.get("report_text", "") if isinstance(case_context, dict) else ""
        patient_complaints = case_context.get("patient_complaints", "") if isinstance(case_context, dict) else ""
        area_dynamics = case_context.get("area_dynamics", {}) if isinstance(case_context, dict) else {}
        mismatch_score = float(area_dynamics.get("mismatch_score", 0.0))
        if mismatch_score > 0.6:
            return "cross_area_alignment"
        if exposures:
            return "epidemiology"
        if report_text or patient_complaints:
            return "language_listening"
        return "context"

    def sample(self, case_context: dict) -> dict:
        case_context = case_context or {}
        focus = self._infer_focus(case_context)
        perturbation_plan = ["context_recombination"]
        if focus == "epidemiology":
            perturbation_plan.append("exposure_shift")
        elif focus == "language_listening":
            perturbation_plan.append("narrative_reframing")
        elif focus == "cross_area_alignment":
            perturbation_plan.append("multimodal_reconciliation")
        return {
            "source": case_context,
            "mode": "counterfactual_variant",
            "variant_focus": focus,
            "perturbation_plan": perturbation_plan,
            "novelty_score": round(0.2 * len(perturbation_plan), 3),
        }
