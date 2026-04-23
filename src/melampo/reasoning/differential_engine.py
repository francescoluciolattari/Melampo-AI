from dataclasses import dataclass

from .support_contradiction import SupportContradictionAnalyzer


@dataclass
class DifferentialEngine:
    """Baseline differential engine using evidence, intuition, dream alternatives, and area dynamics."""

    support_analyzer: SupportContradictionAnalyzer = SupportContradictionAnalyzer()

    def _infer_domain(self, support_signals: list, source: str, hypothesis_type: str) -> str:
        joined = " ".join(support_signals)
        if "fusion:" in joined or "visual_diagnostic" in joined or source in ["mismatch_resolution", "contradiction_revision"]:
            return "imaging_led"
        if "dream:language_listening" in joined:
            return "language_led"
        if "dream:epidemiology" in joined:
            return "epidemiology_led"
        if hypothesis_type == "mismatch_resolution":
            return "mismatch_resolution_led"
        return "multimodal_led"

    def _recommended_tests_for_domain(self, domain: str, contradiction_classes: set[str], coherence_score: float, reasoning_mode: str) -> list[str]:
        tests = []
        if domain == "imaging_led":
            tests.extend(["repeat targeted imaging review", "correlate imaging with structured report findings"])
        elif domain == "language_led":
            tests.extend(["reassess symptom chronology", "clarify narrative inconsistencies"])
        elif domain == "epidemiology_led":
            tests.extend(["review exposure history", "check prevalence-guided differential filters"])
        elif domain == "mismatch_resolution_led":
            tests.extend(["cross-check modality agreement", "request multimodal reconciliation review"])
        else:
            tests.append("expand corroborating evidence")

        if "useful_contradiction" in contradiction_classes:
            tests.append("recheck multimodal alignment")
        if coherence_score < 0.5:
            tests.append("expand corroborating evidence")
        if reasoning_mode == "contradiction_revision":
            tests.append("review alternative hypotheses")
        if "weak_contradiction" in contradiction_classes:
            tests.append("monitor boundary conditions")

        deduped = []
        for item in tests:
            if item not in deduped:
                deduped.append(item)
        return deduped or ["continue standard differential refinement"]

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
        contradiction_classes = {item["class"] for item in contradiction_profiles}

        primary_type = "primary_hypothesis"
        if reasoning_mode == "rational_revision":
            primary_type = "revision_hypothesis"
        elif reasoning_mode == "contradiction_revision":
            primary_type = "contradiction_revision_hypothesis"

        primary_domain = self._infer_domain(support_signals[:4], "intuition_engine", primary_type)
        hypotheses = [
            {
                "label": top_candidate.get("label", "working_hypothesis"),
                "hypothesis_type": primary_type,
                "hypothesis_domain": primary_domain,
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
            hypothesis_type = "revision_alternative"
            if alt.get("kind") == "mismatch_resolution":
                hypothesis_type = "mismatch_resolution"
            elif alt.get("kind") == "contradiction_revision":
                hypothesis_type = "contradiction_revision_alternative"
            alt_support = [f"dream:{alt.get('focus', 'unknown')}"] + support_signals[:1]
            hypotheses.append(
                {
                    "label": alt.get("label", f"alternative_{index + 1}"),
                    "hypothesis_type": hypothesis_type,
                    "hypothesis_domain": self._infer_domain(alt_support, alt.get("kind", "dream_alternative"), hypothesis_type),
                    "score": round(0.4 + mismatch_score * 0.1 - index * 0.05 + signals["contradiction_strength"] * 0.01, 3),
                    "support": max(len(evidence) - index - 1, 0),
                    "source": alt.get("kind", "dream_alternative"),
                    "support_signals": alt_support,
                    "contradiction_signals": contradiction_signals[:3],
                    "support_profile_classes": [item["class"] for item in support_profiles[:2]],
                    "contradiction_profile_classes": [item["class"] for item in contradiction_profiles[:3]],
                }
            )

        if len(hypotheses) == 1:
            hypotheses.append(
                {
                    "label": "alternative_hypothesis",
                    "hypothesis_type": "fallback_alternative",
                    "hypothesis_domain": "multimodal_led",
                    "score": round(0.3 + mismatch_score * 0.05, 3),
                    "support": max(len(evidence) - 1, 0),
                    "source": "fallback_alternative",
                    "support_signals": support_signals[:2],
                    "contradiction_signals": contradiction_signals[:2],
                    "support_profile_classes": [item["class"] for item in support_profiles[:2]],
                    "contradiction_profile_classes": [item["class"] for item in contradiction_profiles[:2]],
                }
            )

        recommended_tests = self._recommended_tests_for_domain(primary_domain, contradiction_classes, coherence_score, reasoning_mode)

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
