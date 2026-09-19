from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..areas.case_context_area import CaseContextArea
from ..areas.epidemiology_area import EpidemiologyArea
from ..areas.language_listening_area import LanguageListeningArea
from ..areas.visual_diagnostic_area import VisualDiagnosticArea
from ..evaluation.quantum_gate import QuantumResearchGate
from ..memory.candidate_retrieval import retrieve_candidates
from ..memory.differential_ranking import rank_differential
from ..memory.graph_sources import load_verification_graph
from ..memory.information_content import InformationContentTable
from ..memory.retriever import MemoryRetriever
from ..memory.visual_imprint import VisualImprintBuilder
from ..models.abstention import AbstentionPolicy
from ..models.evidence_ranker import EvidenceRanker
from ..models.quantum_belief_layer import QuantumBeliefLayer
from ..models.risk_gate import RiskGate
from ..orchestration.runtime_services import RuntimeServices
from ..orchestration.specialist_runtime import SpecialistRuntime
from ..training.counterfactual_sampler import CounterfactualSampler
from ..training.mechanism_enumeration import MechanismEnumerator
from ..training.nexus_scheduler import NexusScheduler
from ..training.nexus_trainer import NexusTrainer
from ..training.replay_filter import ReplayFilter
from ..types import CaseContext
from .area_coherence import AreaCoherenceAnalyzer
from .diagnostic_orchestrator import MelampoDiagnosticOrchestrator
from .differential_engine import DifferentialEngine
from .escalation import EscalationPolicy
from .intuition_engine import IntuitionEngine
from .pipeline_coordinator import PipelineCoordinator
from .policy_stack import PolicyStack

# Capped for MechanismEnumerator.run()'s real per-candidate cost -- see the
# comment at its call site in _nexus_case_context for the measurement.
NEXUS_ENUMERATION_CANDIDATE_CAP = 8



class IngestionProtocol(Protocol):
    def from_payload(self, payload: dict) -> CaseContext: ...


class NormalizerProtocol(Protocol):
    def to_fhir_bundle(self, case: CaseContext) -> dict[str, Any]: ...


class EncoderProtocol(Protocol):
    def encode(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...


class FusionProtocol(Protocol):
    def fuse(self, inputs: dict[str, Any]) -> dict[str, Any]: ...


class CritiqueProtocol(Protocol):
    def review(self, payload: dict[str, Any]) -> dict[str, Any]: ...

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
    """Derive risk, uncertainty and nexus coherence from runtime signals.

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
    nexus_coherence = _clamp(convergence_index * 0.55 + coherence_score * 0.25 + coverage * 0.20)
    return {
        "risk": round(risk, 3),
        "uncertainty": round(uncertainty, 3),
        "nexus_coherence": round(nexus_coherence, 3),
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
    ingestion: IngestionProtocol
    normalizer: NormalizerProtocol
    router: object
    volume_encoder: EncoderProtocol
    pathology_encoder: EncoderProtocol
    text_encoder: EncoderProtocol
    fusion: FusionProtocol
    episodic_memory: object
    semantic_memory: object
    knowledge_graph: object
    workspace: object
    critique: CritiqueProtocol
    metacognition: object
    quantum_layer: object
    replay_engine: object
    logger: object
    # Lazily populated cache for the real concept graph and the
    # MechanismEnumerator built from it -- loaded once per pipeline
    # instance, not per case. _build_runtime_components() runs inside
    # run(), called once per request; without this cache, a 1.27M-edge
    # graph load (roughly six seconds, measured against the real data)
    # would repeat on every single case.
    _nexus_graph_source: Any = None
    _nexus_enumerator: Any = None
    _nexus_ic_table: Any = None
    _nexus_scheduler: Any = None

    def _build_runtime_components(self) -> dict[str, Any]:
        diagnostic_orchestrator = MelampoDiagnosticOrchestrator()
        return {
            "runtime_services": RuntimeServices.build(config=getattr(self.metacognition, "config", object()), logger=self.logger),
            "retriever": MemoryRetriever(memory_store=self.semantic_memory),
            "evidence_ranker": EvidenceRanker(),
            "coordinator": PipelineCoordinator(
                differential_engine=DifferentialEngine(),
                policy_stack=PolicyStack(
                    abstention=AbstentionPolicy(threshold=0.65),
                    risk_gate=RiskGate(threshold=0.35),
                    escalation=EscalationPolicy(),
                ),
            ),
            "quantum_gate": QuantumResearchGate(),
            "nexus_trainer": NexusTrainer(
                replay_filter=ReplayFilter(),
                sampler=CounterfactualSampler(),
                belief_layer=QuantumBeliefLayer(),
                # enumerator stays unset here, deliberately -- see
                # _nexus_case_context, which attaches a real one only when
                # a case actually supplies findings. Loading the real graph
                # unconditionally on every call regressed the test suite
                # from ~13s to ~99s: a dozen pre-existing tests exercise
                # this pipeline with no findings at all, and each one paid
                # the real graph's ~6.6s load cost for an enumerator it was
                # never going to use.
            ),
            "intuition_engine": IntuitionEngine(belief_layer=QuantumBeliefLayer()),
            "visual_area": VisualDiagnosticArea(),
            "language_area": LanguageListeningArea(),
            "context_area": CaseContextArea(),
            "epidemiology_area": EpidemiologyArea(),
            "area_coherence": AreaCoherenceAnalyzer(),
            "diagnostic_orchestrator": diagnostic_orchestrator,
            "specialist_runtime": SpecialistRuntime(registry=diagnostic_orchestrator.registry),
        }

    def _encode_modalities(self, case: CaseContext) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
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
        return text_features, volume_features, pathology_features, fused

    def _retrieve_and_rank(self, components: dict[str, Any], case: CaseContext, payload: dict[str, Any], query_text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        retrieval = components["retriever"].retrieve(
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
        return retrieval, components["evidence_ranker"].rank(retrieval["evidence"])

    def _specialist_signals(
        self,
        components: dict[str, Any],
        case: CaseContext,
        query_text: str,
        volume_features: dict[str, Any],
        retrieval: dict[str, Any],
        ranked_evidence: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        specialist_runtime = components["specialist_runtime"]
        radiology_specialist_signal = specialist_runtime.radiology_signal(
            study_id=str(volume_features.get("study_id", "none")),
            series_paths=list(volume_features.get("series_paths", [])),
            metadata=dict(volume_features.get("metadata", {})) if isinstance(volume_features.get("metadata", {}), dict) else {},
        )
        grounded_text_specialist_signal = specialist_runtime.grounded_text_signal(
            case_id=case.case_id,
            text=query_text,
            grounding={
                "retrieval": retrieval,
                "ranked_evidence": ranked_evidence[:5],
                "case_provenance": case.provenance,
            },
        )
        return radiology_specialist_signal, grounded_text_specialist_signal

    def _build_area_signals(
        self,
        components: dict[str, Any],
        payload: dict[str, Any],
        case: CaseContext,
        bundle: dict[str, Any],
        volume_features: dict[str, Any],
        pathology_features: dict[str, Any],
        radiology_specialist_signal: dict[str, Any],
        grounded_text_specialist_signal: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "visual_diagnostic": components["visual_area"].integrate(
                volume_features=volume_features,
                pathology_features=pathology_features,
                patient_visual=payload.get("patient_visual", {}),
                labs_snapshot=payload.get("labs_snapshot", {}),
                specialist_signal=radiology_specialist_signal,
            ),
            "language_listening": components["language_area"].integrate(
                report_text=case.report_text,
                ehr_text=case.ehr_text,
                patient_complaints=payload.get("patient_complaints", ""),
                voice_features=payload.get("voice_features", {}),
                specialist_signal=grounded_text_specialist_signal,
            ),
            "case_context": components["context_area"].integrate(
                {
                    "demographics": case.demographics,
                    "provenance": case.provenance,
                    "bundle_keys": list(bundle.keys()),
                }
            ),
            "epidemiology": components["epidemiology_area"].integrate(
                demographics=case.demographics,
                provenance=case.provenance,
                exposures=payload.get("exposures", {}),
            ),
        }

    def _graph_candidates_for(self, findings: list[str]) -> list[dict[str, Any]]:
        """Real, IC-weighted candidate diagnoses for IntuitionEngine, or an empty list without findings.

        Reuses the same cached graph `_nexus_enumerator_instance` already
        loads for B1 -- no second real-graph load. `InformationContentTable`
        is cached the same way, measured at 0.43s to build against the real
        graph (far cheaper than the graph load itself, but still worth not
        repeating on every request). Empty findings mean an empty list, not
        an error: `intuition_engine.infer` already treats an empty
        `graph_candidates` as "nothing to promote", falling back to its
        previous placeholder behaviour exactly.
        """
        if not findings:
            return []
        enumerator = self._nexus_enumerator_instance()
        if self._nexus_ic_table is None:
            self._nexus_ic_table = InformationContentTable.from_graph_structure(enumerator.graph)
        report = retrieve_candidates(findings, enumerator.graph, max_candidates=NEXUS_ENUMERATION_CANDIDATE_CAP)
        candidate_names = [item.condition for item in report.candidates]
        ranked = rank_differential(findings, candidate_names, enumerator.graph, self._nexus_ic_table)
        return [item.as_dict() for item in ranked]

    def _nexus_scheduler_instance(self) -> NexusScheduler:
        """The queue NexusTrainer's output feeds, for offline promotion during low-activity windows.

        Cached per pipeline instance, the same pattern as the enumerator:
        the queue is in-memory state that must persist across requests on
        the same running instance, not be rebuilt (and emptied) each time.

        Connects the two halves of what was one intended pipeline, split
        into a live stage and an offline one, that had never actually been
        wired together: NexusTrainer runs synchronously on every case and
        NexusScheduler.enqueue() accepts exactly its output shape as a
        parameter -- the two were designed to compose, and nothing called
        the second with the first's result.

        Deliberately does NOT call `run_once()` here, or anywhere in this
        class. `run_once()` is gated by `LowActivityPolicy`, checking real
        activity metrics (active requests, idle seconds) this
        request-handling method has no business supplying -- calling it
        synchronously inside a request would run the validation and
        promotion work at exactly the wrong time, defeating the reason it
        exists as a separate low-activity stage at all. Queued jobs
        accumulate here; a genuine low-activity trigger (a scheduled job,
        matching the daily/weekly workflow pattern this project already
        uses for literature refresh) is separate, not-yet-built
        infrastructure, not part of this change.
        """
        if self._nexus_scheduler is None:
            self._nexus_scheduler = NexusScheduler()
        return self._nexus_scheduler

    def _nexus_enumerator_instance(self) -> Any:
        """The real MechanismEnumerator, built once against the real concept graph and cached.

        Deliberately independent of the diagnostic_assembly.py reconciliation
        question (an open architectural decision -- see ROADMAP.md, H1): this
        wires only the enumerator itself, a read-only graph traversal for
        hypothesis generation, using graph_sources.load_verification_graph()
        directly rather than diagnostic_assembly.py's full assembly (which
        also carries the learned-edge store and conjecture ledger, a larger
        question this change does not settle). Loaded once per pipeline
        instance, not reimplemented via diagnostic_assembly.nexus_context_for
        to avoid pulling in that module's own dependency chain for a helper
        this pipeline can compute directly from what it already imports.

        Before this, self.knowledge_graph (KnowledgeGraphClient) was the only
        graph-shaped object available here -- a seven-line placeholder whose
        `.lookup()` returns a fixed dict, not a real graph. Wiring the
        enumerator to it would have produced hypotheses from a graph with no
        real edges, indistinguishable from the rehearsal-label fallback this
        change replaces.
        """
        if self._nexus_enumerator is None:
            self._nexus_graph_source = load_verification_graph()
            self._nexus_enumerator = MechanismEnumerator(graph=self._nexus_graph_source.graph)
        return self._nexus_enumerator

    def _nexus_case_context(
        self, *, case: CaseContext, payload: dict[str, Any], bundle: dict[str, Any],
        area_dynamics: dict[str, Any], governance_scores: dict[str, Any], visual_imprints: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """The nexus branch's case context, enriched with findings and candidate conditions when available.

        `payload.get("findings")` is the caller-supplied finding list --
        absent, this enriches nothing and the nexus trainer's enumerator
        stays unset (see `_run_nexus_branch`), falling back to rehearsal
        labels exactly as before. Present, `candidate_conditions` is
        derived from the real graph via `retrieve_candidates`, matching
        `diagnostic_assembly.nexus_context_for`'s own logic without
        importing that module.
        """
        findings = [str(item) for item in (payload.get("findings") or []) if str(item).strip()]
        candidate_conditions: list[str] = []
        if findings:
            # Capped rather than passing every candidate through: verified
            # directly against the real graph that MechanismEnumerator.run()
            # costs roughly 3.5-4 seconds per candidate (2 candidates ~7s, 5
            # ~20s, 10 ~40s -- linear, not a fixed overhead), a real
            # performance limitation discovered while wiring this, not
            # previously measured because MechanismEnumerator had only ever
            # been exercised against small fixture graphs. Uncapped, a case
            # with retrieve_candidates' typical yield (dozens of candidates)
            # would make this branch take minutes. The cap keeps the branch
            # usable now; the underlying per-candidate cost is a distinct,
            # deeper question -- see ROADMAP.md, H3.
            report = retrieve_candidates(
                findings, self._nexus_enumerator_instance().graph, max_candidates=NEXUS_ENUMERATION_CANDIDATE_CAP
            )
            candidate_conditions = [item.condition for item in report.candidates]
        return {
            "case_id": case.case_id,
            "bundle_keys": list(bundle.keys()),
            "demographics": case.demographics,
            "provenance": case.provenance,
            "report_text": case.report_text,
            "patient_complaints": payload.get("patient_complaints", ""),
            "exposures": payload.get("exposures", {}),
            "area_dynamics": area_dynamics,
            "governance_scores": governance_scores,
            "visual_imprints": visual_imprints,
            "diagnostic_visual_imprints": visual_imprints,
            "concept_memory_imprints": payload.get("concept_memory_imprints", []),
            "findings": findings,
            "candidate_conditions": candidate_conditions,
        }

    def _run_nexus_branch(
        self,
        components: dict[str, Any],
        payload: dict[str, Any],
        case: CaseContext,
        bundle: dict[str, Any],
        area_dynamics: dict[str, Any],
        governance_scores: dict[str, Any],
        visual_imprints: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # The real graph and MechanismEnumerator load only when there is
        # something for them to do -- attached here, not at
        # _build_runtime_components() time, so the dozens of existing
        # callers that never supply findings never pay for a graph they
        # will not use. Once attached, the same instance is cached on
        # self._nexus_enumerator (_nexus_enumerator_instance) and reused
        # for the rest of this pipeline instance's lifetime.
        if payload.get("findings"):
            components["nexus_trainer"].enumerator = self._nexus_enumerator_instance()
        return components["nexus_trainer"].run(
            case_context=self._nexus_case_context(
                case=case, payload=payload, bundle=bundle, area_dynamics=area_dynamics,
                governance_scores=governance_scores, visual_imprints=visual_imprints,
            ),
            coherence=governance_scores["nexus_coherence"],
            risk=governance_scores["risk"],
        )

    def _coordination_evidence(
        self,
        bundle: dict[str, Any],
        retrieval: dict[str, Any],
        fused: dict[str, Any],
        resolved: dict[str, Any],
        intuition_engine: IntuitionEngine,
        intuition: dict[str, Any],
        ranked_evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        evidence = [
            {"source": "bundle", "kind": "bundle_keys", "value": list(bundle.keys())},
            {"source": "retrieval", "kind": retrieval["retrieval_mode"], "value": retrieval["evidence_count"]},
            {"source": "fusion", "kind": "engine", "value": fused.get("engine", fused.get("provider", "none"))},
            {"source": "service", "kind": "provider", "value": resolved["service"].get("provider", "none")},
            {"source": "intuition", "kind": "candidate", "value": intuition_engine.summarize_for_trace(intuition)},
        ]
        evidence.extend(ranked_evidence)
        return evidence

    def _finalize_diagnostic_result(self, components: dict[str, Any], pipeline_result: dict[str, Any], retrieval: dict[str, Any], ranked_evidence: list[dict[str, Any]]) -> None:
        pipeline_result["diagnostic_result"] = components["diagnostic_orchestrator"].orchestrate(pipeline_result)
        external_critique = components["specialist_runtime"].external_critique(
            diagnostic_result=pipeline_result["diagnostic_result"],
            literature_context={
                "retrieval": retrieval,
                "ranked_evidence": ranked_evidence,
            },
        )
        pipeline_result["external_critique"] = external_critique
        pipeline_result["diagnostic_result"]["external_critique"] = external_critique
        pipeline_result["diagnostic_result"]["audit_trace"]["external_critic_is_final_arbiter"] = False

    def run(self, payload: dict) -> dict:
        case = self.ingestion.from_payload(payload)
        bundle = self.normalizer.to_fhir_bundle(case)
        components = self._build_runtime_components()
        text_features, volume_features, pathology_features, fused = self._encode_modalities(case)
        query_text = case.report_text or case.ehr_text or case.case_id
        retrieval, ranked_evidence = self._retrieve_and_rank(components, case, payload, query_text)
        resolved = components["runtime_services"].resolve("volume_encoder")
        quantum_allowed = components["quantum_gate"].allow(contextuality_score=0.7)
        radiology_specialist_signal, grounded_text_specialist_signal = self._specialist_signals(
            components=components,
            case=case,
            query_text=query_text,
            volume_features=volume_features,
            retrieval=retrieval,
            ranked_evidence=ranked_evidence,
        )
        area_signals = self._build_area_signals(
            components=components,
            payload=payload,
            case=case,
            bundle=bundle,
            volume_features=volume_features,
            pathology_features=pathology_features,
            radiology_specialist_signal=radiology_specialist_signal,
            grounded_text_specialist_signal=grounded_text_specialist_signal,
        )
        visual_imprints = VisualImprintBuilder().from_visual_area(
            signal=area_signals["visual_diagnostic"],
            volume_features=volume_features,
        )
        area_dynamics = components["area_coherence"].analyze(area_signals)
        governance_scores = _derive_governance_scores(
            payload=payload,
            area_dynamics=area_dynamics,
            retrieval=retrieval,
            ranked_evidence=ranked_evidence,
            area_signals=area_signals,
        )
        nexus = self._run_nexus_branch(
            components=components,
            payload=payload,
            case=case,
            bundle=bundle,
            area_dynamics=area_dynamics,
            governance_scores=governance_scores,
            visual_imprints=visual_imprints,
        )
        # The connection found missing while reading both modules in full:
        # NexusScheduler.enqueue()'s `nexus` parameter accepts exactly this
        # dict's shape, unmodified. A minimal case_context is built here
        # rather than reusing NexusTrainer's own enriched one (which also
        # carries findings/candidate_conditions the scheduler's own
        # candidate-text generation does not read) -- the scheduler only
        # needs case_id, report_text and patient_complaints.
        self._nexus_scheduler_instance().enqueue(
            case_context={
                "case_id": case.case_id,
                "report_text": case.report_text,
                "patient_complaints": payload.get("patient_complaints", ""),
            },
            area_dynamics=area_dynamics,
            nexus=nexus,
            retrieval_context=retrieval,
            governance_scores=governance_scores,
        )
        intuition_engine = components["intuition_engine"]
        intuition = intuition_engine.infer(
            case_id=case.case_id,
            ranked_evidence=ranked_evidence,
            nexus=nexus,
            quantum_allowed=quantum_allowed,
            area_signals=area_signals,
            area_dynamics=area_dynamics,
            graph_candidates=self._graph_candidates_for(payload.get("findings") or []),
        )
        evidence = self._coordination_evidence(
            bundle=bundle,
            retrieval=retrieval,
            fused=fused,
            resolved=resolved,
            intuition_engine=intuition_engine,
            intuition=intuition,
            ranked_evidence=ranked_evidence,
        )
        coordinated = components["coordinator"].run(
            case_id=case.case_id,
            evidence=evidence,
            risk=governance_scores["risk"],
            uncertainty=governance_scores["uncertainty"],
            intuition=intuition,
            nexus=nexus,
            area_dynamics=area_dynamics,
        )
        critique_result = self.critique.review({"coordinated": coordinated, "intuition": intuition, "areas": area_signals, "area_dynamics": area_dynamics, "nexus": nexus})
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
            "visual_imprints": visual_imprints,
            "specialist_signals": {
                "radiology": radiology_specialist_signal,
                "grounded_text": grounded_text_specialist_signal,
            },
            "services": resolved,
            "intuition": intuition,
            "coordinated": coordinated,
            "critique": critique_result,
            "quantum_allowed": quantum_allowed,
            "nexus": nexus,
        }
        self._finalize_diagnostic_result(
            components=components,
            pipeline_result=pipeline_result,
            retrieval=retrieval,
            ranked_evidence=ranked_evidence,
        )
        return pipeline_result
