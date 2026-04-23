from dataclasses import dataclass


@dataclass
class DifferentialEngine:
    """Baseline differential engine using evidence, intuition, dream alternatives, and area dynamics."""

    def rank(self, evidence: list, intuition: dict | None = None, dream: dict | None = None, area_dynamics: dict | None = None) -> dict:
        intuition = intuition or {}
        dream = dream or {}
        area_dynamics = area_dynamics or {}

        candidate_scores = intuition.get("candidate_scores", [])
        top_candidate = candidate_scores[0] if candidate_scores else {"label": "working_hypothesis", "score": 0.7}
        alternatives = dream.get("alternative_hypotheses", [])
        mismatch_score = float(area_dynamics.get("mismatch_score", 0.0))
        coherence_score = float(area_dynamics.get("coherence_score", 0.0))
        mismatch_pairs = area_dynamics.get("mismatch_pairs", [])
        coherence_pairs = area_dynamics.get("coherence_pairs", [])
        reasoning_mode = intuition.get("deductive_filter", {}).get("reasoning_mode", "rapid_intuition")

        support_signals = []
        contradiction_signals = []
        for item in evidence:
            if isinstance(item, dict):
                source = item.get("source", "unknown")
                kind = item.get("kind", "signal")
                if source in ["retrieval", "bundle", "fusion", "service"]:
                    support_signals.append(f"{source}:{kind}")
                if source == "intuition" and reasoning_mode == "contradiction_revision":
                    contradiction_signals.append("intuition:contradiction_revision")
        if mismatch_score > 0.5:
            contradiction_signals.append("areas:high_mismatch")
        contradiction_signals.extend([f"pair:{a}-{b}" for a, b in mismatch_pairs[:2]])

        hypotheses = [
            {
                "label": top_candidate.get("label", "working_hypothesis"),
                "score": round(float(top_candidate.get("score", 0.7)) + coherence_score * 0.1, 3),
                "support": len(evidence),
                "source": "intuition_engine",
                "support_signals": support_signals[:3],
                "contradiction_signals": contradiction_signals[:2],
            }
        ]

        for index, alt in enumerate(alternatives[:3]):
            hypotheses.append(
                {
                    "label": alt.get("label", f"alternative_{index + 1}"),
                    "score": round(0.4 + mismatch_score * 0.1 - index * 0.05, 3),
                    "support": max(len(evidence) - index - 1, 0),
                    "source": alt.get("kind", "dream_alternative"),
                    "support_signals": [f"dream:{alt.get('focus', 'unknown')}"] + [f"pair:{a}-{b}" for a, b in coherence_pairs[:1]],
                    "contradiction_signals": contradiction_signals[:2],
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
                    "contradiction_signals": contradiction_signals[:1],
                }
            )

        recommended_tests = []
        if mismatch_score > 0.5:
            recommended_tests.append("recheck multimodal alignment")
        if coherence_score < 0.5:
            recommended_tests.append("expand corroborating evidence")
        if reasoning_mode == "contradiction_revision":
            recommended_tests.append("review alternative hypotheses")
        if not recommended_tests:
            recommended_tests.append("continue standard differential refinement")

        return {
            "status": "grounded_differential_ready",
            "evidence_count": len(evidence),
            "mismatch_score": mismatch_score,
            "coherence_score": coherence_score,
            "reasoning_mode": reasoning_mode,
            "hypotheses": hypotheses,
            "recommended_tests": recommended_tests,
        }
