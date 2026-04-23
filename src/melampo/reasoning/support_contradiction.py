from dataclasses import dataclass


@dataclass
class SupportContradictionAnalyzer:
    """Build lightweight support and contradiction signals from evidence and area dynamics."""

    def analyze(self, evidence: list, intuition: dict | None = None, dream: dict | None = None, area_dynamics: dict | None = None) -> dict:
        intuition = intuition or {}
        dream = dream or {}
        area_dynamics = area_dynamics or {}

        reasoning_mode = intuition.get("deductive_filter", {}).get("reasoning_mode", "rapid_intuition")
        mismatch_score = float(area_dynamics.get("mismatch_score", 0.0))
        coherence_score = float(area_dynamics.get("coherence_score", 0.0))
        mismatch_pairs = area_dynamics.get("mismatch_pairs", [])
        coherence_pairs = area_dynamics.get("coherence_pairs", [])
        rehearsal_profile = dream.get("rehearsal_profile", {})

        support_profiles = []
        contradiction_profiles = []

        for item in evidence:
            if not isinstance(item, dict):
                continue
            source = item.get("source", "unknown")
            kind = item.get("kind", "signal")
            if source in ["bundle", "retrieval", "fusion", "service"]:
                support_profiles.append({
                    "label": f"{source}:{kind}",
                    "class": "grounded_support",
                    "strength": 0.2,
                })
            if source == "intuition" and reasoning_mode == "contradiction_revision":
                contradiction_profiles.append({
                    "label": "intuition:contradiction_revision",
                    "class": "useful_contradiction",
                    "strength": 0.4,
                })

        if coherence_score > 0.4:
            for pair in coherence_pairs[:2]:
                support_profiles.append({
                    "label": f"pair:{pair[0]}-{pair[1]}",
                    "class": "multimodal_support",
                    "strength": 0.3,
                })

        if rehearsal_profile.get("coherence_guidance") == "multimodal_support":
            support_profiles.append({
                "label": "dream:multimodal_support",
                "class": "replay_supported_alignment",
                "strength": 0.25,
            })

        if mismatch_score > 0.6:
            contradiction_profiles.append({
                "label": "areas:high_mismatch",
                "class": "useful_contradiction",
                "strength": 0.5,
            })
            for pair in mismatch_pairs[:2]:
                contradiction_profiles.append({
                    "label": f"pair:{pair[0]}-{pair[1]}",
                    "class": "useful_contradiction",
                    "strength": 0.35,
                })
        elif mismatch_score > 0.3:
            contradiction_profiles.append({
                "label": "areas:moderate_mismatch",
                "class": "weak_contradiction",
                "strength": 0.2,
            })
        elif mismatch_pairs:
            contradiction_profiles.append({
                "label": "areas:low_conflict",
                "class": "spurious_conflict",
                "strength": 0.1,
            })

        if rehearsal_profile.get("boundary_case_hint", False):
            contradiction_profiles.append({
                "label": "dream:boundary_case",
                "class": "weak_contradiction",
                "strength": 0.15,
            })

        support_signals = [item["label"] for item in support_profiles]
        contradiction_signals = [item["label"] for item in contradiction_profiles]
        support_strength = round(sum(item["strength"] for item in support_profiles), 3)
        contradiction_strength = round(sum(item["strength"] for item in contradiction_profiles), 3)

        return {
            "support_profiles": support_profiles,
            "contradiction_profiles": contradiction_profiles,
            "support_signals": support_signals,
            "contradiction_signals": contradiction_signals,
            "support_strength": support_strength,
            "contradiction_strength": contradiction_strength,
        }
