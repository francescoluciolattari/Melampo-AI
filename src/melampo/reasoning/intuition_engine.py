from dataclasses import dataclass

from ..models.quantum_belief_layer import QuantumBeliefLayer


@dataclass
class IntuitionEngine:
    """Research scaffold for biologically inspired inductive-deductive intuition."""

    belief_layer: QuantumBeliefLayer

    def infer(self, case_id: str, ranked_evidence: list, dream: dict, quantum_allowed: bool, area_signals: dict | None = None, area_dynamics: dict | None = None) -> dict:
        area_signals = area_signals or {}
        area_dynamics = area_dynamics or {}
        rehearsal_profile = dream.get("rehearsal_profile", {}) if isinstance(dream, dict) else {}
        alternative_hypotheses = dream.get("alternative_hypotheses", []) if isinstance(dream, dict) else []
        contradiction_rehearsal = bool(rehearsal_profile.get("contradiction_rehearsal", False))
        revision_bias = rehearsal_profile.get("revision_bias", "exploratory")
        post_error_adjustment = rehearsal_profile.get("post_error_adjustment", "stabilize_primary")
        coherence_score_ext = float(area_dynamics.get("coherence_score", 0.0))
        mismatch_score_ext = float(area_dynamics.get("mismatch_score", 0.0))

        inductive_candidates = [
            {
                "label": f"candidate_{index + 1}",
                "support_weight": item.get("weight", 1),
                "source": item.get("item", {}).get("source", "unknown"),
            }
            for index, item in enumerate(ranked_evidence[:3])
        ]
        area_ranking = []
        for name, payload in area_signals.items():
            signal_size = len(payload) if isinstance(payload, dict) else 1
            area_ranking.append({"area": name, "weight": signal_size})
        area_ranking.sort(key=lambda item: item["weight"], reverse=True)
        top_areas = [item["area"] for item in area_ranking[:2]]

        if area_ranking:
            top_weight = area_ranking[0]["weight"]
            second_weight = area_ranking[1]["weight"] if len(area_ranking) > 1 else 0
            convergence_score = round(second_weight / max(top_weight, 1), 3)
            conflict_score = round((top_weight - second_weight) / max(top_weight, 1), 3)
        else:
            convergence_score = 0.0
            conflict_score = 0.0

        area_pair_bonus = 0.0
        top_area_pair = tuple(sorted(top_areas))
        if top_area_pair == ("language_listening", "visual_diagnostic"):
            area_pair_bonus = 0.2
        elif top_area_pair == ("epidemiology", "visual_diagnostic"):
            area_pair_bonus = 0.15
        elif top_area_pair == ("case_context", "language_listening"):
            area_pair_bonus = 0.1
        area_pair_bonus = round(area_pair_bonus + (0.1 * coherence_score_ext), 3)

        disagreement_penalty = round(max(conflict_score - 0.4, 0.0) + mismatch_score_ext * 0.1, 3)

        rapid_intuition = inductive_candidates[0]["label"] if inductive_candidates else "no_candidate"
        rational_revision = inductive_candidates[1]["label"] if len(inductive_candidates) > 1 else rapid_intuition
        contradiction_revision = alternative_hypotheses[0]["label"] if alternative_hypotheses else rational_revision

        rapid_score = round((inductive_candidates[0]["support_weight"] if inductive_candidates else 0.0) + area_pair_bonus + convergence_score - disagreement_penalty, 3)
        rational_score = round((inductive_candidates[1]["support_weight"] if len(inductive_candidates) > 1 else 0.0) + conflict_score + (0.2 if revision_bias == "conservative" else 0.0) + mismatch_score_ext * 0.1, 3)
        contradiction_score = round((1.0 if contradiction_rehearsal else 0.0) + (0.3 if post_error_adjustment == "re-rank_alternatives" else 0.0) + (0.1 * len(alternative_hypotheses)) + mismatch_score_ext * 0.2, 3)

        candidate_scores = [
            {"mode": "rapid_intuition", "label": rapid_intuition, "score": rapid_score},
            {"mode": "rational_revision", "label": rational_revision, "score": rational_score},
            {"mode": "contradiction_revision", "label": contradiction_revision, "score": contradiction_score},
        ]
        candidate_scores.sort(key=lambda item: item["score"], reverse=True)
        selected = candidate_scores[0] if candidate_scores else {"mode": "rapid_intuition", "label": "no_candidate", "score": 0.0}
        intuition = selected["label"]
        reasoning_mode = selected["mode"]

        deductive_filter = {
            "kept": len(inductive_candidates),
            "rejected": max(len(ranked_evidence) - len(inductive_candidates), 0),
            "criterion": "top_ranked_grounded_evidence",
            "active_areas": sorted(area_signals.keys()),
            "top_areas": top_areas,
            "convergence_score": convergence_score,
            "conflict_score": conflict_score,
            "coherence_score": coherence_score_ext,
            "mismatch_score": mismatch_score_ext,
            "area_pair_bonus": area_pair_bonus,
            "disagreement_penalty": disagreement_penalty,
            "contradiction_rehearsal": contradiction_rehearsal,
            "revision_bias": revision_bias,
            "post_error_adjustment": post_error_adjustment,
            "reasoning_mode": reasoning_mode,
        }
        if quantum_allowed:
            dream_mode = "none"
            if isinstance(dream, dict):
                belief = dream.get("belief", {})
                if isinstance(belief, dict):
                    dream_mode = belief.get("mode", "none")
            belief_update = self.belief_layer.update(
                prior={"case_id": case_id, "candidate_count": len(inductive_candidates)},
                context={
                    "dream_mode": dream_mode,
                    "quantum_allowed": quantum_allowed,
                    "area_count": len(area_signals),
                    "top_areas": top_areas,
                    "convergence_score": convergence_score,
                    "conflict_score": conflict_score,
                    "coherence_score": coherence_score_ext,
                    "mismatch_score": mismatch_score_ext,
                    "area_pair_bonus": area_pair_bonus,
                    "disagreement_penalty": disagreement_penalty,
                    "contradiction_rehearsal": contradiction_rehearsal,
                    "revision_bias": revision_bias,
                    "reasoning_mode": reasoning_mode,
                },
            )
        else:
            belief_update = {
                "mode": "classical_only",
                "area_count": len(area_signals),
                "top_areas": top_areas,
                "convergence_score": convergence_score,
                "conflict_score": conflict_score,
                "coherence_score": coherence_score_ext,
                "mismatch_score": mismatch_score_ext,
                "area_pair_bonus": area_pair_bonus,
                "disagreement_penalty": disagreement_penalty,
                "contradiction_rehearsal": contradiction_rehearsal,
                "revision_bias": revision_bias,
                "reasoning_mode": reasoning_mode,
            }
        return {
            "intuition": intuition,
            "rapid_intuition": rapid_intuition,
            "rational_revision": rational_revision,
            "contradiction_revision": contradiction_revision,
            "candidate_scores": candidate_scores,
            "inductive_candidates": inductive_candidates,
            "dream_alternatives": alternative_hypotheses,
            "deductive_filter": deductive_filter,
            "belief_update": belief_update,
            "area_signals": area_signals,
            "area_ranking": area_ranking,
            "area_dynamics": area_dynamics,
        }

    def summarize_for_trace(self, intuition_payload: dict) -> dict:
        area_signals = intuition_payload.get("area_signals", {})
        deductive = intuition_payload.get("deductive_filter", {})
        return {
            "intuition": intuition_payload.get("intuition", "none"),
            "candidate_count": len(intuition_payload.get("inductive_candidates", [])),
            "belief_mode": intuition_payload.get("belief_update", {}).get("mode", "none"),
            "active_areas": sorted(area_signals.keys()),
            "top_areas": deductive.get("top_areas", []),
            "convergence_score": deductive.get("convergence_score", 0.0),
            "conflict_score": deductive.get("conflict_score", 0.0),
            "coherence_score": deductive.get("coherence_score", 0.0),
            "mismatch_score": deductive.get("mismatch_score", 0.0),
            "area_pair_bonus": deductive.get("area_pair_bonus", 0.0),
            "disagreement_penalty": deductive.get("disagreement_penalty", 0.0),
            "contradiction_rehearsal": deductive.get("contradiction_rehearsal", False),
            "revision_bias": deductive.get("revision_bias", "exploratory"),
            "post_error_adjustment": deductive.get("post_error_adjustment", "stabilize_primary"),
            "reasoning_mode": deductive.get("reasoning_mode", "rapid_intuition"),
        }
