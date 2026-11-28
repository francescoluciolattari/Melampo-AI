from dataclasses import dataclass


@dataclass
class CritiqueLoop:
    config: object
    logger: object

    def _prioritize_actions(
        self, actions: list, discordances: int, escalate: bool
    ) -> list:
        prioritized = []
        for action in actions:
            category = (
                action.get("category", "confirmation_test")
                if isinstance(action, dict)
                else "confirmation_test"
            )
            label = (
                action.get("label", str(action))
                if isinstance(action, dict)
                else str(action)
            )
            priority = "medium"
            if (
                category in {"multimodal_reconciliation", "disambiguation_test"}
                and discordances > 0
            ):
                priority = "high"
            elif category == "confirmation_test":
                priority = "medium"
            if escalate:
                priority = "high"
            prioritized.append(
                {"category": category, "label": label, "priority": priority}
            )
        return prioritized

    def review(self, draft: dict) -> dict:
        discordances = (
            draft.get("discordances", {}) if isinstance(draft, dict) else {}
        )
        nexus = draft.get("nexus", {}) if isinstance(draft, dict) else {}
        coordinated = draft.get("coordinated", {}) if isinstance(draft, dict) else {}
        differential = (
            coordinated.get("differential", {}) if isinstance(coordinated, dict) else {}
        )
        contradiction_profiles = differential.get("contradiction_profiles", [])
        contradiction_classes = {
            item.get("class", "unknown")
            for item in contradiction_profiles
            if isinstance(item, dict)
        }
        recommended_actions = differential.get("recommended_actions", [])
        discordance_count = int((discordances or {}).get("count", 0) or 0)
        escalate = (
            bool(coordinated.get("policy", {}).get("escalate", False))
            if isinstance(coordinated, dict)
            else False
        )

        warnings = []
        if discordance_count:
            warnings.append("discordances_in_the_reading")
        if nexus.get("rehearsal_profile", {}).get("boundary_case_hint", False):
            warnings.append("boundary_case_review")
        if "useful_contradiction" in contradiction_classes:
            warnings.append("useful_contradiction_detected")
        if "spurious_conflict" in contradiction_classes:
            warnings.append("possible_spurious_conflict")

        suggestions = []
        if (
            "discordances_in_the_reading" in warnings
            or "useful_contradiction_detected" in warnings
        ):
            suggestions.append("recheck multimodal consistency")
        if escalate:
            suggestions.append("escalate to expert review")
        if "possible_spurious_conflict" in warnings:
            suggestions.append("down-weight low-value conflicts")
        if not suggestions:
            suggestions.append("proceed with monitored interpretation")

        prioritized_actions = self._prioritize_actions(
            actions=recommended_actions,
            discordances=discordance_count,
            escalate=escalate,
        )

        return {
            "status": "reviewed",
            "warnings": warnings,
            "suggestions": suggestions,
            "prioritized_actions": prioritized_actions,
            "draft": draft,
        }
