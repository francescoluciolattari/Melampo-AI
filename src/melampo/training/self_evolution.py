from __future__ import annotations

from dataclasses import dataclass, field

from ..memory.vector_memory import InMemoryVectorStore


@dataclass(slots=True)
class DreamSelfEvolutionLoop:
    """Controlled offline self-evolution loop for dream/intuitive rehearsal.

    The loop promotes only candidates with favorable neuro-dynamic metrics and
    keeps all generated memories marked as research artifacts. It is designed for
    idle-time rehearsal, not unsupervised clinical deployment.
    """

    vector_store: InMemoryVectorStore = field(default_factory=InMemoryVectorStore)
    min_pi_score: float = 0.55
    max_prediction_error: float = 0.45
    min_bias_suppression: float = 0.45

    def generate_candidate(self, case_context: dict, area_dynamics: dict, dream: dict | None = None) -> dict:
        dream = dream or {}
        neuro = area_dynamics.get("neuro_dynamic_metrics", {}) if isinstance(area_dynamics, dict) else {}
        top_pairs = area_dynamics.get("coherence_pairs", [])[:2] if isinstance(area_dynamics, dict) else []
        mismatch_pairs = area_dynamics.get("mismatch_pairs", [])[:2] if isinstance(area_dynamics, dict) else []
        case_id = case_context.get("case_id", "unknown_case")
        text = (
            f"Dream rehearsal for {case_id}. "
            f"Coherent pairs: {top_pairs}. Mismatch pairs: {mismatch_pairs}. "
            f"Report: {case_context.get('report_text', '')}. "
            f"Complaints: {case_context.get('patient_complaints', '')}."
        )
        return {
            "record_id": f"dream-{case_id}-{len(self.vector_store.records) + 1}",
            "text": text,
            "metadata": {
                "case_id": case_id,
                "pi_score": neuro.get("pi_score", area_dynamics.get("pi_score", 0.0)),
                "prediction_error": neuro.get("prediction_error", area_dynamics.get("prediction_error", 0.0)),
                "bias_suppression_score": neuro.get("bias_suppression_score", 0.0),
                "reasoning_mode": dream.get("rehearsal_profile", {}).get("replay_mode", "dream_rehearsal"),
                "coherence_pairs": top_pairs,
                "mismatch_pairs": mismatch_pairs,
            },
            "source": "dream_self_evolution_loop",
        }

    def evaluate_candidate(self, candidate: dict) -> dict:
        metadata = candidate.get("metadata", {})
        pi_score = float(metadata.get("pi_score", 0.0))
        prediction_error = float(metadata.get("prediction_error", 1.0))
        bias_suppression = float(metadata.get("bias_suppression_score", 0.0))
        accepted = (
            pi_score >= self.min_pi_score
            and prediction_error <= self.max_prediction_error
            and bias_suppression >= self.min_bias_suppression
        )
        return {
            "accepted": accepted,
            "pi_score": pi_score,
            "prediction_error": prediction_error,
            "bias_suppression_score": bias_suppression,
            "criteria": {
                "min_pi_score": self.min_pi_score,
                "max_prediction_error": self.max_prediction_error,
                "min_bias_suppression": self.min_bias_suppression,
            },
            "decision": "promote_to_memory" if accepted else "retain_as_candidate_only",
        }

    def rehearse(self, case_context: dict, area_dynamics: dict, dream: dict | None = None) -> dict:
        candidate = self.generate_candidate(case_context=case_context, area_dynamics=area_dynamics, dream=dream)
        evaluation = self.evaluate_candidate(candidate)
        status = "candidate"
        if evaluation["accepted"]:
            status = "promoted"
        record = self.vector_store.upsert_text(
            record_id=candidate["record_id"],
            text=candidate["text"],
            metadata={**candidate["metadata"], "evaluation": evaluation},
            source=candidate["source"],
            learning_status=status,
        )
        return {
            "candidate": candidate,
            "evaluation": evaluation,
            "memory_record": record,
            "vector_memory": self.vector_store.describe(),
        }
