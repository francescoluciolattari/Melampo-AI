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
        contradiction_profiles = coordinated.get("differential", {}).get("contradiction_profiles", [])
        contradiction_classes = {item.get("class", "unknown") for item in contradiction_profiles if isinstance(item, dict)}

        warnings = []
        if area_dynamics.get("mismatch_score", 0.0) > 0.5:
            warnings.append("high_cross_area_mismatch")
        if intuition.get("deductive_filter", {}).get("reasoning_mode") == "contradiction_revision":
            warnings.append("contradiction_revision_active")
        if dream.get("rehearsal_profile", {}).get("boundary_case_hint", False):
            warnings.append("boundary_case_review")
        if "useful_contradiction" in contradiction_classes:
            warnings.append("useful_contradiction_detected")
        if "spurious_conflict" in contradiction_classes:
            warnings.append("possible_spurious_conflict")

        suggestions = []
        if "high_cross_area_mismatch" in warnings or "useful_contradiction_detected" in warnings:
            suggestions.append("recheck multimodal consistency")
        if coordinated.get("policy", {}).get("escalate", False):
            suggestions.append("escalate to expert review")
        if "possible_spurious_conflict" in warnings:
            suggestions.append("down-weight low-value conflicts")
        if not suggestions:
            suggestions.append("proceed with monitored interpretation")

        return {
            "status": "reviewed",
            "warnings": warnings,
            "suggestions": suggestions,
            "draft": draft,
        }
