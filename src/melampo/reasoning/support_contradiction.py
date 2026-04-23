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

        support_signals = []
        contradiction_signals = []

        for item in evidence:
            if not isinstance(item, dict):
                continue
            source = item.get("source", "unknown")
            kind = item.get("kind", "signal")
            if source in ["bundle", "retrieval", "fusion", "service"]:
                support_signals.append(f"{source}:{kind}")
            if source == "intuition" and reasoning_mode == "contradiction_revision":
                contradiction_signals.append("intuition:contradiction_revision")

        if coherence_score > 0.4:
            for pair in coherence_pairs[:2]:
                support_signals.append(f"pair:{pair[0]}-{pair[1]}")
        if mismatch_score > 0.5:
            contradiction_signals.append("areas:high_mismatch")
            for pair in mismatch_pairs[:2]:
                contradiction_signals.append(f"pair:{pair[0]}-{pair[1]}")
        if rehearsal_profile.get("boundary_case_hint", False):
            contradiction_signals.append("dream:boundary_case")
        if rehearsal_profile.get("coherence_guidance") == "multimodal_support":
            support_signals.append("dream:multimodal_support")

        return {
            "support_signals": support_signals,
            "contradiction_signals": contradiction_signals,
            "support_strength": len(support_signals),
            "contradiction_strength": len(contradiction_signals),
        }
