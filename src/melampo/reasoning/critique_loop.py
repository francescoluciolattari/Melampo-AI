from dataclasses import dataclass


@dataclass
class CritiqueLoop:
    config: object
    logger: object

    def _prioritize_actions(self, actions: list, mismatch_score: float, escalate: bool, reasoning_mode: str) -> list:
        prioritized = []
        for action in actions:
            category = action.get("category", "confirmation_test") if isinstance(action, dict) else "confirmation_test"
            label = action.get("label", str(action)) if isinstance(action, dict) else str(action)
            priority = "medium"
            if category == "multimodal_reconciliation" and mismatch_score > 0.5:
                priority = "high"
            elif category == "disambiguation_test" and reasoning_mode == "contradiction_revision":
                priority = "high"
            elif category == "confirmation_test":
                priority = "medium"
            if escalate:
                priority = "high"
            prioritized.append({"category": category, "label": label, "priority": priority})
        return prioritized

    def review(self, draft: dict) -> dict:
        intuition = draft.get("intuition", {}) if isinstance(draft, dict) else {}
        area_dynamics = draft.get("area_dynamics", {}) if isinstance(draft, dict) else {}
        dream = draft.get("dream", {}) if isinstance(draft, dict) else {}
        coordinated = draft.get("coordinated", {}) if isinstance(draft, dict) else {}
        differential = coordinated.get("differential", {}) if isinstance(coordinated, dict) else {}
        contradiction_profiles = differential.get("contradiction_profiles", [])
        contradiction_classes = {item.get("class", "unknown") for item in contradiction_profiles if isinstance(item, dict)}
        recommended_actions = differential.get("recommended_actions", [])
        mismatch_score = float(area_dynamics.get("mismatch_score", 0.0))
        reasoning_mode = intuition.get("deductive_filter", {}).get("reasoning_mode", "rapid_intuition")
        escalate = bool(coordinated.get("policy", {}).get("escalate", False)) if isinstance(coordinated, dict) else False

        warnings = []
        if mismatch_score > 0.5:
            warnings.append("high_cross_area_mismatch")
        if reasoning_mode == "contradiction_revision":
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
        if escalate:
            suggestions.append("escalate to expert review")
        if "possible_spurious_conflict" in warnings:
            suggestions.append("down-weight low-value conflicts")
        if not suggestions:
            suggestions.append("proceed with monitored interpretation")

        prioritized_actions = self._prioritize_actions(
            actions=recommended_actions,
            mismatch_score=mismatch_score,
            escalate=escalate,
            reasoning_mode=reasoning_mode,
        )

        return {
            "status": "reviewed",
            "warnings": warnings,
            "suggestions": suggestions,
            "prioritized_actions": prioritized_actions,
            "draft": draft,
        }
