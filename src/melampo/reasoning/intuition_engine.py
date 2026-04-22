from dataclasses import dataclass

from ..models.quantum_belief_layer import QuantumBeliefLayer


@dataclass
class IntuitionEngine:
    """Research scaffold for inductive-deductive intuition over ranked evidence."""

    belief_layer: QuantumBeliefLayer

    def infer(self, case_id: str, ranked_evidence: list, dream: dict, quantum_allowed: bool) -> dict:
        inductive_candidates = [
            {
                "label": f"candidate_{index + 1}",
                "support_weight": item.get("weight", 1),
                "source": item.get("item", {}).get("source", "unknown"),
            }
            for index, item in enumerate(ranked_evidence[:3])
        ]
        deductive_filter = {
            "kept": len(inductive_candidates),
            "rejected": max(len(ranked_evidence) - len(inductive_candidates), 0),
            "criterion": "top_ranked_grounded_evidence",
        }
        if quantum_allowed:
            dream_mode = "none"
            if isinstance(dream, dict):
                belief = dream.get("belief", {})
                if isinstance(belief, dict):
                    dream_mode = belief.get("mode", "none")
            belief_update = self.belief_layer.update(
                prior={"case_id": case_id, "candidate_count": len(inductive_candidates)},
                context={"dream_mode": dream_mode, "quantum_allowed": quantum_allowed},
            )
        else:
            belief_update = {"mode": "classical_only"}
        intuition = inductive_candidates[0]["label"] if inductive_candidates else "no_candidate"
        return {
            "intuition": intuition,
            "inductive_candidates": inductive_candidates,
            "deductive_filter": deductive_filter,
            "belief_update": belief_update,
        }

    def summarize_for_trace(self, intuition_payload: dict) -> dict:
        return {
            "intuition": intuition_payload.get("intuition", "none"),
            "candidate_count": len(intuition_payload.get("inductive_candidates", [])),
            "belief_mode": intuition_payload.get("belief_update", {}).get("mode", "none"),
        }
