from dataclasses import dataclass

from ..models.quantum_belief_layer import QuantumBeliefLayer


@dataclass
class IntuitionEngine:
    """Research scaffold for biologically inspired inductive-deductive intuition."""

    belief_layer: QuantumBeliefLayer

    def infer(self, case_id: str, ranked_evidence: list, dream: dict, quantum_allowed: bool, area_signals: dict | None = None) -> dict:
        area_signals = area_signals or {}
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
        deductive_filter = {
            "kept": len(inductive_candidates),
            "rejected": max(len(ranked_evidence) - len(inductive_candidates), 0),
            "criterion": "top_ranked_grounded_evidence",
            "active_areas": sorted(area_signals.keys()),
            "top_areas": top_areas,
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
                },
            )
        else:
            belief_update = {"mode": "classical_only", "area_count": len(area_signals), "top_areas": top_areas}
        intuition = inductive_candidates[0]["label"] if inductive_candidates else "no_candidate"
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
        return {
            "intuition": intuition_payload.get("intuition", "none"),
            "candidate_count": len(intuition_payload.get("inductive_candidates", [])),
            "belief_mode": intuition_payload.get("belief_update", {}).get("mode", "none"),
            "active_areas": sorted(area_signals.keys()),
            "top_areas": intuition_payload.get("deductive_filter", {}).get("top_areas", []),
        }
