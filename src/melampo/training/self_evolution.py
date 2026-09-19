from __future__ import annotations

from dataclasses import dataclass, field

from ..memory.vector_memory import InMemoryVectorStore


@dataclass(slots=True)
class NexusSelfEvolutionLoop:
    """Controlled offline self-evolution loop for nexus/intuitive rehearsal.

    The loop promotes only candidates with favorable neuro-dynamic metrics and
    keeps all generated memories marked as research artifacts. It is designed for
    idle-time rehearsal, not unsupervised clinical deployment.
    """

    vector_store: InMemoryVectorStore = field(default_factory=InMemoryVectorStore)
    min_pi_score: float = 0.55
    max_prediction_error: float = 0.45
    min_bias_suppression: float = 0.45

    def generate_candidate(
        self, case_context: dict, area_dynamics: dict, nexus: dict | None = None, governance_scores: dict | None = None
    ) -> dict:
        nexus = nexus or {}
        governance_scores = governance_scores or {}
        neuro = area_dynamics.get("neuro_dynamic_metrics", {}) if isinstance(area_dynamics, dict) else {}
        top_pairs = area_dynamics.get("coherence_pairs", [])[:2] if isinstance(area_dynamics, dict) else []
        mismatch_pairs = area_dynamics.get("mismatch_pairs", [])[:2] if isinstance(area_dynamics, dict) else []
        case_id = case_context.get("case_id", "unknown_case")
        text = (
            f"Nexus rehearsal for {case_id}. "
            f"Coherent pairs: {top_pairs}. Mismatch pairs: {mismatch_pairs}. "
            f"Report: {case_context.get('report_text', '')}. "
            f"Complaints: {case_context.get('patient_complaints', '')}."
        )
        # candidate_score consolidated here from NexusTrainer's former
        # _auto_evolution_plan() -- the only field of that method's output
        # promotion_policy.decide() and rational_control_validator.py
        # actually read (as the primary promotion threshold and a
        # validation fallback respectively). Computed fresh from the same
        # inputs _auto_evolution_plan() used: neuro-dynamic metrics already
        # available via area_dynamics, plus visual_morphing and risk, now
        # read directly from `nexus` (NexusTrainer's own output) and the
        # newly added `governance_scores` parameter -- nothing here needs
        # NexusTrainer to keep computing it.
        visual_morphing = nexus.get("visual_morphing", {}) if isinstance(nexus.get("visual_morphing", {}), dict) else {}
        candidate_score = round(
            float(neuro.get("pi_score", 0.0)) * 0.32
            + float(neuro.get("convergence_index", 0.0)) * 0.27
            + float(neuro.get("nexus_plasticity", 0.0)) * 0.18
            + float(visual_morphing.get("visual_morph_intuition_gain", 0.0)) * 0.08
            - float(governance_scores.get("risk", 0.0)) * 0.15,
            3,
        )
        return {
            "record_id": f"nexus-{case_id}-{len(self.vector_store.records) + 1}",
            "text": text,
            "metadata": {
                "case_id": case_id,
                "pi_score": neuro.get("pi_score", area_dynamics.get("pi_score", 0.0)),
                "prediction_error": neuro.get("prediction_error", area_dynamics.get("prediction_error", 0.0)),
                "bias_suppression_score": neuro.get("bias_suppression_score", 0.0),
                "reasoning_mode": nexus.get("rehearsal_profile", {}).get("replay_mode", "nexus_rehearsal"),
                "coherence_pairs": top_pairs,
                "mismatch_pairs": mismatch_pairs,
                "candidate_score": candidate_score,
            },
            "source": "nexus_self_evolution_loop",
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

    def rehearse(
        self, case_context: dict, area_dynamics: dict, nexus: dict | None = None, governance_scores: dict | None = None
    ) -> dict:
        candidate = self.generate_candidate(
            case_context=case_context, area_dynamics=area_dynamics, nexus=nexus, governance_scores=governance_scores
        )
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
