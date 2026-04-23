from dataclasses import dataclass


@dataclass
class DifferentialEngine:
    """Baseline differential engine using evidence, intuition, and dream alternatives."""

    def rank(self, evidence: list, intuition: dict | None = None, dream: dict | None = None, area_dynamics: dict | None = None) -> dict:
        intuition = intuition or {}
        dream = dream or {}
        area_dynamics = area_dynamics or {}

        candidate_scores = intuition.get("candidate_scores", [])
        top_candidate = candidate_scores[0] if candidate_scores else {"label": "working_hypothesis", "score": 0.7}
        alternatives = dream.get("alternative_hypotheses", [])
        mismatch_score = float(area_dynamics.get("mismatch_score", 0.0))
        coherence_score = float(area_dynamics.get("coherence_score", 0.0))

        hypotheses = [
            {
                "label": top_candidate.get("label", "working_hypothesis"),
                "score": round(float(top_candidate.get("score", 0.7)) + coherence_score * 0.1, 3),
                "support": len(evidence),
                "source": "intuition_engine",
            }
        ]

        for index, alt in enumerate(alternatives[:2]):
            hypotheses.append(
                {
                    "label": alt.get("label", f"alternative_{index + 1}"),
                    "score": round(0.4 + mismatch_score * 0.1 - index * 0.05, 3),
                    "support": max(len(evidence) - index - 1, 0),
                    "source": alt.get("kind", "dream_alternative"),
                }
            )

        if len(hypotheses) == 1:
            hypotheses.append(
                {
                    "label": "alternative_hypothesis",
                    "score": round(0.3 + mismatch_score * 0.05, 3),
                    "support": max(len(evidence) - 1, 0),
                    "source": "fallback_alternative",
                }
            )

        return {
            "status": "grounded_differential_ready",
            "evidence_count": len(evidence),
            "mismatch_score": mismatch_score,
            "coherence_score": coherence_score,
            "hypotheses": hypotheses,
        }
