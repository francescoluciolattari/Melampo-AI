from dataclasses import dataclass

from .support_contradiction import SupportContradictionAnalyzer


@dataclass
class DifferentialEngine:
    """Baseline differential engine using evidence, intuition, dream alternatives, and area dynamics."""

    support_analyzer: SupportContradictionAnalyzer = SupportContradictionAnalyzer()

    def rank(self, evidence: list, intuition: dict | None = None, dream: dict | None = None, area_dynamics: dict | None = None) -> dict:
        intuition = intuition or {}
        dream = dream or {}
        area_dynamics = area_dynamics or {}

        candidate_scores = intuition.get("candidate_scores", [])
        top_candidate = candidate_scores[0] if candidate_scores else {"label": "working_hypothesis", "score": 0.7}
        alternatives = dream.get("alternative_hypotheses", [])
        mismatch_score = float(area_dynamics.get("mismatch_score", 0.0))
        coherence_score = float(area_dynamics.get("coherence_score", 0.0))
        reasoning_mode = intuition.get("deductive_filter", {}).get("reasoning_mode", "rapid_intuition")

        signals = self.support_analyzer.analyze(evidence=evidence, intuition=intuition, dream=dream, area_dynamics=area_dynamics)
        support_signals = signals["support_signals"]
        contradiction_signals = signals["contradiction_signals"]
        support_profiles = signals["support_profiles"]
        contradiction_profiles = signals["contradiction_profiles"]

        hypotheses = [
            {
                "label": top_candidate.get("label", "working_hypothesis"),
                "score": round(float(top_candidate.get("score", 0.7)) + coherence_score * 0.1 + signals["support_strength"] * 0.02, 3),
                "support": len(evidence),
                "source": "intuition_engine",
                "support_signals": support_signals[:4],
                "contradiction_signals": contradiction_signals[:3],
                "support_profile_classes": [item["class"] for item in support_profiles[:3]],
                "contradiction_profile_classes": [item["class"] for item in contradiction_profiles[:3]],
            }
        ]

        for index, alt in enumerate(alternatives[:3]):
            hypotheses.append(
                {
                    "label": alt.get("label", f"alternative_{index + 1}"),
                    "score": round(0.4 + mismatch_score * 0.1 - index * 0.05 + signals["contradiction_strength"] * 0.01, 3),
                    "support": max(len(evidence) - index - 1, 0),
                    "source": alt.get("kind", "dream_alternative"),
                    "support_signals": [f"dream:{alt.get('focus', 'unknown')}"] + support_signals[:1],
                    "contradiction_signals": contradiction_signals[:3],
                    "support_profile_classes": [item["class"] for item in support_profiles[:2]],
                    "contradiction_profile_classes": [item["class"] for item in contradiction_profiles[:3]],
                }
            )

        if len(hypotheses) == 1:
            hypotheses.append(
                {
                    "label": "alternative_hypothesis",
                    "score": round(0.3 + mismatch_score * 0.05, 3),
                    "support": max(len(evidence) - 1, 0),
                    "source": "fallback_alternative",
                    "support_signals": support_signals[:2],
                    "contradiction_signals": contradiction_signals[:2],
                    "support_profile_classes": [item["class"] for item in support_profiles[:2]],
                    "contradiction_profile_classes": [item["class"] for item in contradiction_profiles[:2]],
                }
            )

        recommended_tests = []
        contradiction_classes = {item["class"] for item in contradiction_profiles}
        if "useful_contradiction" in contradiction_classes:
            recommended_tests.append("recheck multimodal alignment")
        if coherence_score < 0.5:
            recommended_tests.append("expand corroborating evidence")
        if reasoning_mode == "contradiction_revision":
            recommended_tests.append("review alternative hypotheses")
        if "weak_contradiction" in contradiction_classes:
            recommended_tests.append("monitor boundary conditions")
        if not recommended_tests:
            recommended_tests.append("continue standard differential refinement")

        return {
            "status": "grounded_differential_ready",
            "evidence_count": len(evidence),
            "mismatch_score": mismatch_score,
            "coherence_score": coherence_score,
            "reasoning_mode": reasoning_mode,
            "support_strength": signals["support_strength"],
            "contradiction_strength": signals["contradiction_strength"],
            "support_profiles": support_profiles,
            "contradiction_profiles": contradiction_profiles,
            "hypotheses": hypotheses,
            "recommended_tests": recommended_tests,
        }
