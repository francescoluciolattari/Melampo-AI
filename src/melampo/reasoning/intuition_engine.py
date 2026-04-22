from dataclasses import dataclass

from ..models.quantum_belief_layer import QuantumBeliefLayer


@dataclass
class IntuitionEngine:
    """Research scaffold for biologically inspired inductive-deductive intuition."""

    belief_layer: QuantumBeliefLayer

    def infer(self, case_id: str, ranked_evidence: list, dream: dict, quantum_allowed: bool, area_signals: dict | None = None) -> dict:
        area_signals = area_signals or {}
        rehearsal_profile = dream.get("rehearsal_profile", {}) if isinstance(dream, dict) else {}
        contradiction_rehearsal = bool(rehearsal_profile.get("contradiction_rehearsal", False))
        revision_bias = rehearsal_profile.get("revision_bias", "exploratory")

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

        if inductive_candidates:
            if contradiction_rehearsal and len(inductive_candidates) > 1:
                intuition = inductive_candidates[1]["label"]
            elif revision_bias == "conservative" and len(inductive_candidates) > 1:
                intuition = inductive_candidates[1]["label"]
            elif convergence_score >= 0.5:
                intuition = inductive_candidates[0]["label"]
            elif len(inductive_candidates) > 1:
                intuition = inductive_candidates[1]["label"]
            else:
                intuition = inductive_candidates[0]["label"]
        else:
            intuition = "no_candidate"

        deductive_filter = {
            "kept": len(inductive_candidates),
            "rejected": max(len(ranked_evidence) - len(inductive_candidates), 0),
            "criterion": "top_ranked_grounded_evidence",
            "active_areas": sorted(area_signals.keys()),
            "top_areas": top_areas,
            "convergence_score": convergence_score,
            "conflict_score": conflict_score,
            "contradiction_rehearsal": contradiction_rehearsal,
            "revision_bias": revision_bias,
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
                    "contradiction_rehearsal": contradiction_rehearsal,
                    "revision_bias": revision_bias,
                },
            )
        else:
            belief_update = {
                "mode": "classical_only",
                "area_count": len(area_signals),
                "top_areas": top_areas,
                "convergence_score": convergence_score,
                "conflict_score": conflict_score,
                "contradiction_rehearsal": contradiction_rehearsal,
                "revision_bias": revision_bias,
            }
        return {
            "intuition": intuition,
            "inductive_candidates": inductive_candidates,
            "deductive_filter": deductive_filter,
            "belief_update": belief_update,
            "area_signals": area_signals,
            "area_ranking": area_ranking,
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
            "contradiction_rehearsal": deductive.get("contradiction_rehearsal", False),
            "revision_bias": deductive.get("revision_bias", "exploratory"),
        }
