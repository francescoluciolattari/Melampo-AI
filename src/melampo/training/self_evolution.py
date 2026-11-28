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
        self,
        case_context: dict,
        discordances: dict | None = None,
        nexus: dict | None = None,
        governance_scores: dict | None = None,
    ) -> dict:
        """The case's rehearsal, packaged for the candidate store; its ``candidate_score`` is 0.

        Until 28 November 2026 the score was a sum of pi_score, convergence and plasticity with hand-set
        weights, measures of the coherence between areas that was never measured; removed with it. The chain
        is closed: no candidate reaches the threshold of ``promotion_policy`` until the lateral cycle of the
        intuition engine has criteria measured on published cases.
        """
        nexus = nexus or {}
        discordances = discordances or {}
        case_id = case_context.get("case_id", "unknown_case")
        text = (
            f"Nexus rehearsal for {case_id}. "
            f"Discordances: {discordances.get('kinds', {})}. "
            f"Report: {case_context.get('report_text', '')}. "
            f"Complaints: {case_context.get('patient_complaints', '')}."
        )
        return {
            "record_id": f"nexus-{case_id}-{len(self.vector_store.records) + 1}",
            "text": text,
            "metadata": {
                "case_id": case_id,
                "discordances": int(discordances.get("count", 0) or 0),
                "reasoning_mode": nexus.get("rehearsal_profile", {}).get(
                    "replay_mode", "nexus_rehearsal"
                ),
                "candidate_score": 0.0,
            },
            "source": "nexus_self_evolution_loop",
        }
