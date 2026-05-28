from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _round_unit(value: float) -> float:
    return round(_clamp(float(value)), 3)


@dataclass(frozen=True, slots=True)
class MelampoMetrics:
    """Typed metrics emitted by the Melampo diagnostic orchestration layer.

    The metrics are computational abstractions inspired by predictive
    processing, active inference, recurrent cortical integration, inhibitory
    control and action-potential-like gating. They are not literal biological
    measurements and require dataset-driven calibration before clinical use.
    """

    pi_score: float = 0.0
    precision_weighted_coherence: float = 0.0
    prediction_error: float = 0.0
    convergence_index: float = 0.0
    mismatch_index: float = 0.0
    deductive_gate: float = 0.0
    candidate_temperature: float = 1.0
    belief_update_rate: float = 0.0
    cross_area_synchrony: float = 0.0
    conflict_load: float = 0.0
    inhibitory_control: float = 0.0
    revision_pressure: float = 0.0
    dream_plasticity: float = 0.0
    bias_suppression_score: float = 0.0
    interdependence_index: float = 0.0
    evidence_integration_score: float = 0.0
    noise_suppression_score: float = 0.0
    action_potential_gate: float = 0.0
    synaptic_plasticity_index: float = 0.0
    deep_inference_score: float = 0.0
    deductive_stability: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "pi_score": _round_unit(self.pi_score),
            "precision_weighted_coherence": _round_unit(self.precision_weighted_coherence),
            "prediction_error": _round_unit(self.prediction_error),
            "convergence_index": _round_unit(self.convergence_index),
            "mismatch_index": _round_unit(self.mismatch_index),
            "deductive_gate": _round_unit(self.deductive_gate),
            "candidate_temperature": round(max(0.0, float(self.candidate_temperature)), 3),
            "belief_update_rate": _round_unit(self.belief_update_rate),
            "cross_area_synchrony": _round_unit(self.cross_area_synchrony),
            "conflict_load": _round_unit(self.conflict_load),
            "inhibitory_control": _round_unit(self.inhibitory_control),
            "revision_pressure": _round_unit(self.revision_pressure),
            "dream_plasticity": _round_unit(self.dream_plasticity),
            "bias_suppression_score": _round_unit(self.bias_suppression_score),
            "interdependence_index": _round_unit(self.interdependence_index),
            "evidence_integration_score": _round_unit(self.evidence_integration_score),
            "noise_suppression_score": _round_unit(self.noise_suppression_score),
            "action_potential_gate": _round_unit(self.action_potential_gate),
            "synaptic_plasticity_index": _round_unit(self.synaptic_plasticity_index),
            "deep_inference_score": _round_unit(self.deep_inference_score),
            "deductive_stability": _round_unit(self.deductive_stability),
            "interpretation": "computational_neuro_inspired_research_metrics_not_literal_biology",
        }


@dataclass(frozen=True, slots=True)
class IntuitionSummary:
    selected: str = "none"
    reasoning_mode: str = "unknown"
    candidate_scores: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "reasoning_mode": self.reasoning_mode,
            "candidate_scores": list(self.candidate_scores),
        }


@dataclass(frozen=True, slots=True)
class DreamSummary:
    accepted: bool = False
    auto_evolution_plan: dict[str, Any] = field(default_factory=dict)
    alternative_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    promotion_policy: dict[str, Any] = field(
        default_factory=lambda: {
            "dream_outputs_are_candidate_only": True,
            "automatic_clinical_promotion_allowed": False,
            "promotion_requires": [
                "source_grounding",
                "contradiction_review",
                "human_review_before_clinical_use",
                "audit_trace",
            ],
        }
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "auto_evolution_plan": dict(self.auto_evolution_plan),
            "alternative_hypotheses": list(self.alternative_hypotheses),
            "promotion_policy": dict(self.promotion_policy),
        }


@dataclass(frozen=True, slots=True)
class DiagnosticResult:
    """Enterprise-grade typed final research diagnostic result.

    This is not a clinical diagnosis. It is an auditable research output owned
    by MelampoDiagnosticOrchestrator. External models are signal providers or
    critics only and cannot become final diagnostic arbiters.
    """

    case_id: str
    result_label: str
    top_hypothesis: dict[str, Any]
    differential: list[dict[str, Any]]
    intuition: IntuitionSummary
    melampo_metrics: MelampoMetrics
    support: dict[str, Any]
    policy: dict[str, Any]
    critique: dict[str, Any]
    dream: DreamSummary
    model_capability_decision_record: dict[str, Any]
    audit_trace: dict[str, Any]
    schema_version: str = "diagnostic_result.v1"
    created_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "case_id": self.case_id,
            "result_label": self.result_label,
            "top_hypothesis": dict(self.top_hypothesis),
            "differential": list(self.differential),
            "intuition": self.intuition.as_dict(),
            "melampo_metrics": self.melampo_metrics.as_dict(),
            "support": dict(self.support),
            "policy": dict(self.policy),
            "critique": dict(self.critique),
            "dream": self.dream.as_dict(),
            "model_capability_decision_record": dict(self.model_capability_decision_record),
            "audit_trace": {
                **dict(self.audit_trace),
                "final_authority": "MelampoDiagnosticOrchestrator",
                "external_models_are_not_final_arbiters": True,
                "clinical_warning": "Research output; not a validated medical device or standalone diagnosis.",
            },
        }
