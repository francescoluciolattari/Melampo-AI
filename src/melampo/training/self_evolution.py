from __future__ import annotations

from dataclasses import dataclass, field

from ..memory.vector_memory import InMemoryVectorStore


@dataclass(slots=True)
class NexusSelfEvolutionLoop:
    """Packages a case's rehearsal candidate for the real governance chain.

    Reduced after removing rehearse() and evaluate_candidate(): both wrote
    directly to vector memory using their own threshold check
    (min_pi_score/max_prediction_error/min_bias_suppression), bypassing the
    real governed chain (NexusCandidateStore -> RationalControlValidator ->
    PromotionPolicy) nexus_scheduler.py._execute_job() actually uses -- and
    which correctly holds every candidate at needs_review with this
    project's default settings (PromotionPolicy.allow_automatic_promotion
    is False), rather than auto-deciding "promoted" the way rehearse() did
    on its own. Verified before removing: neither method was called by
    anything except each other and a test exercising rehearse() directly,
    never by _execute_job() or any other real caller -- generate_candidate()
    alone is, and remains, the real interface.
    """

    vector_store: InMemoryVectorStore = field(default_factory=InMemoryVectorStore)

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