from dataclasses import dataclass


@dataclass
class CritiqueLoop:
    config: object
    logger: object

    def review(self, draft: dict) -> dict:
        intuition = draft.get("intuition", {}) if isinstance(draft, dict) else {}
        area_dynamics = draft.get("area_dynamics", {}) if isinstance(draft, dict) else {}
        dream = draft.get("dream", {}) if isinstance(draft, dict) else {}
        coordinated = draft.get("coordinated", {}) if isinstance(draft, dict) else {}

        warnings = []
        if area_dynamics.get("mismatch_score", 0.0) > 0.5:
            warnings.append("high_cross_area_mismatch")
        if intuition.get("deductive_filter", {}).get("reasoning_mode") == "contradiction_revision":
            warnings.append("contradiction_revision_active")
        if dream.get("rehearsal_profile", {}).get("boundary_case_hint", False):
            warnings.append("boundary_case_review")

        suggestions = []
        if "high_cross_area_mismatch" in warnings:
            suggestions.append("recheck multimodal consistency")
        if coordinated.get("policy", {}).get("escalate", False):
            suggestions.append("escalate to expert review")
        if not suggestions:
            suggestions.append("proceed with monitored interpretation")

        return {
            "status": "reviewed",
            "warnings": warnings,
            "suggestions": suggestions,
            "draft": draft,
        }
