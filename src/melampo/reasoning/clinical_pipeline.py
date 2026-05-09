from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..areas.case_context_area import CaseContextArea
from ..areas.epidemiology_area import EpidemiologyArea
from ..areas.language_listening_area import LanguageListeningArea
from ..areas.visual_diagnostic_area import VisualDiagnosticArea
from ..evaluation.quantum_gate import QuantumResearchGate
from ..memory.retriever import MemoryRetriever
from ..models.abstention import AbstentionPolicy
from ..models.evidence_ranker import EvidenceRanker
from ..models.quantum_belief_layer import QuantumBeliefLayer
from ..models.risk_gate import RiskGate
from ..orchestration.runtime_services import RuntimeServices
from ..training.counterfactual_sampler import CounterfactualSampler
from ..training.dream_trainer import DreamTrainer
from ..training.replay_filter import ReplayFilter
from .area_coherence import AreaCoherenceAnalyzer
from .diagnostic_orchestrator import MelampoDiagnosticOrchestrator
from .differential_engine import DifferentialEngine
from .escalation import EscalationPolicy
from .intuition_engine import IntuitionEngine
from .pipeline_coordinator import PipelineCoordinator
from .policy_stack import PolicyStack


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _area_uncertainty(area_signals: dict[str, Any]) -> float:
    values = []
    for payload in area_signals.values():
        if isinstance(payload, dict):
            salience = _safe_float(payload.get("salience_score", 0.0))
            values.append(_safe_float(payload.get("uncertainty_score", 1.0 - min(salience, 1.0))))
    return _clamp(_mean(values, default=0.65))


def _derive_governance_scores(
    payload: dict[str, Any],
    area_dynamics: dict[str, Any],
    retrieval: dict[str, Any],
    ranked_evidence: list[dict[str, Any]],
    area_signals: dict[str, Any],
) -> dict[str, Any]:
    """Derive risk, uncertainty and dream coherence from runtime signals.

    Replaces P0 hardcoded constants with an auditable approximation based on
    retrieval coverage, model/area uncertainty, mismatch, prediction error,
    provenance quality and optional case severity supplied by callers.
    """

    neuro_metrics = area_dynamics.get("neuro_dynamic_metrics", {}) if isinstance(area_dynamics, dict) else {}
    mismatch_index = _safe_float(neuro_metrics.get("mismatch_index", area_dynamics.get("mismatch_score", 0.0)))
    prediction_error = _safe_float(neuro_metrics.get("prediction_error", area_dynamics.get("prediction_error", 0.0)))
    convergence_index = _safe_float(neuro_metrics.get("convergence_index", area_dynamics.get("convergence_index", 0.0)))
    coherence_score = _safe_float(area_dynamics.get("coherence_score", 0.0))
    retrieval_quality = retrieval.get("retrieval_quality", {}) if isinstance(retrieval, dict) else {}
    coverage = _clamp(_safe_float(retrieval_quality.get("coverage", min(_safe_float(retrieval.get("evidence_count", 0.0)) / 5.0, 1.0))))
    memory_backed = bool(retrieval_quality.get("memory_backed", False))
    fallback_penalty = 0.2 if retrieval_quality.get("fallback_used", False) or not memory_backed else 0.0
    mean_grounding = _clamp(_safe_float(retrieval_quality.get("mean_grounding_score", 0.0)))
    evidence_strength = _clamp(_mean([_safe_float(item.get("weight", 0.0)) / 3.0 for item in ranked_evidence[:3]], default=0.0))
    mean_area_uncertainty = _area_uncertainty(area_signals)
    provenance = payload.get("provenance", {}) if isinstance(payload, dict) else {}
    weak_provenance = 0.0 if isinstance(provenance, dict) and provenance else 0.25
    clinical_severity = _clamp(_safe_float(payload.get("clinical_severity", payload.get("risk_hint", 0.0))))
    missing_evidence = _clamp(1.0 - coverage)

    uncertainty = _clamp(
        missing_evidence * 0.30
        + mean_area_uncertainty * 0.25
        + mismatch_index * 0.20
        + prediction_error * 0.15
        + fallback_penalty * 0.10
        - evidence_strength * 0.10
        - mean_grounding * 0.05
    )
    risk = _clamp(
        clinical_severity * 0.30
        + mismatch_index * 0.25
        + uncertainty * 0.20
        + prediction_error * 0.15
        + weak_provenance * 0.10
    )
    dream_coherence = _clamp(convergence_index * 0.55 + coherence_score * 0.25 + coverage * 0.20)
    return {
        "risk": round(risk, 3),
        "uncertainty": round(uncertainty, 3),
        "dream_coherence": round(dream_coherence, 3),
        "missing_evidence": round(missing_evidence, 3),
        "retrieval_coverage": round(coverage, 3),
        "mean_grounding_score": round(mean_grounding, 3),
        "mean_area_uncertainty": round(mean_area_uncertainty, 3),
        "mismatch_index": round(mismatch_index, 3),
        "prediction_error": round(prediction_error, 3),
        "convergence_index": round(convergence_index, 3),
        "memory_backed_retrieval": memory_backed,
        "fallback_penalty": round(fallback_penalty, 3),
        "weak_provenance": round(weak_provenance, 3),
        "clinical_severity": round(clinical_severity, 3),
        "derivation": "runtime_governance_scores_not_hardcoded_constants",
    }


@dataclass
class ClinicalInferencePipeline:
    ingestion: object
    normalizer: object
    router: object
    volume_encoder: object
    pathology_encoder: object
    text_encoder: object
    fusion: object
    episodic_memory: object
    semantic_memory: object
    knowledge_graph: object
    workspace: object
    critique: object
    metacognition: object
    quantum_layer: object
    replay_engine: object
    logger: object

    def run(self, payload: dict) -> dict:
        case = self.ingestion.from_payload(payload)
        bundle = self.normalizer.to_fhir_bundle(case)

        runtime_services = RuntimeServices.build(config=getattr(self.metacognition, "config", object()), logger=self.logger)
        retriever = MemoryRetriever(memory_store=self.semantic_memory)
        evidence_ranker = EvidenceRanker()
        coordinator = PipelineCoordinator(
            differential_engine=DifferentialEngine(),
            policy_stack=PolicyStack(
                abstention=AbstentionPolicy(threshold=0.65),
                risk_gate=RiskGate(threshold=0.35),
                escalation=EscalationPolicy(),
            ),
        )
        quantum_gate = QuantumResearchGate()
        dream_trainer = DreamTrainer(
            replay_filter=ReplayFilter(),
            sampler=CounterfactualSampler(),
            belief_layer=QuantumBeliefLayer(),
        )
        intuition_engine = IntuitionEngine(belief_layer=QuantumBeliefLayer())
        visual_area = VisualDiagnosticArea()
        language_area = LanguageListeningArea()
        context_area = CaseContextArea()
        epidemiology_area = EpidemiologyArea()
        area_coherence = AreaCoherenceAnalyzer()
        diagnostic_orchestrator = MelampoDiagnosticOrchestrator()

        text_features = self.text_encoder.encode(case.report_text or case.ehr_text or case.case_id)
        if case.imaging:
            first_study = case.imaging[0]
            first_study_id = first_study.study_id
            volume_features = self.volume_encoder.encode(
                first_study_id,
                series_paths=list(first_study.series_paths),
                metadata=dict(first_study.metadata),
            )
            pathology_features = self.pathology_encoder.encode(first_study_id)
        else:
            volume_features = {"study_id": "none", "series_paths": [], "image_count": 0, "has_local_images": False}
            pathology_features = {"slide_id": "none"}

        fused = self.fusion.fuse(
            {
                "text": text_features,
                "volume": volume_features,
                "pathology": pathology_features,
            }
        )
        query_text = case.report_text or case.ehr_text or case.case_id
        retrieval = retriever.retrieve(
            query_text,
            top_k=5,
            case_context={
                "case_id": case.case_id,
                "demographics": case.demographics,
                "provenance": case.provenance,
                "exposures": payload.get("exposures", {}),
            },
            target_areas=["visual_diagnostic", "language_listening", "case_context", "epidemiology"],
        )
        ranked_evidence = evidence_ranker.rank(retrieval["evidence"])
        resolved = runtime_services.resolve("volume_encoder")
        quantum_allowed = quantum_gate.allow(contextuality_score=0.7)

        area_signals = {
            "visual_diagnostic": visual_area.integrate(
                volume_features=volume_features,
                pathology_features=pathology_features,
                patient_visual=payload.get("patient_visual", {}),
                labs_snapshot=payload.get("labs_snapshot", {}),
            ),
            "language_listening": language_area.integrate(
                report_text=case.report_text,
                ehr_text=case.ehr_text,
                patient_complaints=payload.get("patient_complaints", ""),
                voice_features=payload.get("voice_features", {}),
            ),
            "case_context": context_area.integrate(
                {
                    "demographics": case.demographics,
                    "provenance": case.provenance,
                    "bundle_keys": list(bundle.keys()),
                }
            ),
            "epidemiology": epidemiology_area.integrate(
                demographics=case.demographics,
                provenance=case.provenance,
                exposures=payload.get("exposures", {}),
            ),
        }
        area_dynamics = area_coherence.analyze(area_signals)
        governance_scores = _derive_governance_scores(
            payload=payload,
            area_dynamics=area_dynamics,
            retrieval=retrieval,
            ranked_evidence=ranked_evidence,
            area_signals=area_signals,
        )

        dream = dream_trainer.run(
            case_context={
                "case_id": case.case_id,
                "bundle_keys": list(bundle.keys()),
                "demographics": case.demographics,
                "provenance": case.provenance,
                "report_text": case.report_text,
                "patient_complaints": payload.get("patient_complaints", ""),
                "exposures": payload.get("exposures", {}),
                "area_dynamics": area_dynamics,
                "governance_scores": governance_scores,
            },
            coherence=governance_scores["dream_coherence"],
            risk=governance_scores["risk"],
        )

        intuition = intuition_engine.infer(
            case_id=case.case_id,
            ranked_evidence=ranked_evidence,
            dream=dream,
            quantum_allowed=quantum_allowed,
            area_signals=area_signals,
            area_dynamics=area_dynamics,
        )
        evidence = [
            {"source": "bundle", "kind": "bundle_keys", "value": list(bundle.keys())},
            {"source": "retrieval", "kind": retrieval["retrieval_mode"], "value": retrieval["evidence_count"]},
            {"source": "fusion", "kind": "engine", "value": fused.get("engine", fused.get("provider", "none"))},
            {"source": "service", "kind": "provider", "value": resolved["service"].get("provider", "none")},
            {"source": "intuition", "kind": "candidate", "value": intuition_engine.summarize_for_trace(intuition)},
        ]
        evidence.extend(ranked_evidence)
        coordinated = coordinator.run(
            case_id=case.case_id,
            evidence=evidence,
            risk=governance_scores["risk"],
            uncertainty=governance_scores["uncertainty"],
            intuition=intuition,
            dream=dream,
            area_dynamics=area_dynamics,
        )
        critique_result = self.critique.review({"coordinated": coordinated, "intuition": intuition, "areas": area_signals, "area_dynamics": area_dynamics, "dream": dream})
        pipeline_result = {
            "case_id": case.case_id,
            "bundle_keys": list(bundle.keys()),
            "text_features": text_features,
            "volume_features": volume_features,
            "pathology_features": pathology_features,
            "fused": fused,
            "retrieval": retrieval,
            "ranked_evidence": ranked_evidence,
            "area_signals": area_signals,
            "area_dynamics": area_dynamics,
            "governance_scores": governance_scores,
            "services": resolved,
            "intuition": intuition,
            "coordinated": coordinated,
            "critique": critique_result,
            "quantum_allowed": quantum_allowed,
            "dream": dream,
        }
        pipeline_result["diagnostic_result"] = diagnostic_orchestrator.orchestrate(pipeline_result)
        return pipeline_result
