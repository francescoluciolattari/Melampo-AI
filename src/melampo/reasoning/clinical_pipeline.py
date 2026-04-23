from dataclasses import dataclass

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
from .differential_engine import DifferentialEngine
from .escalation import EscalationPolicy
from .intuition_engine import IntuitionEngine
from .pipeline_coordinator import PipelineCoordinator
from .policy_stack import PolicyStack


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
        retriever = MemoryRetriever()
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

        text_features = self.text_encoder.encode(case.report_text or case.ehr_text or case.case_id)
        if case.imaging:
            first_study_id = case.imaging[0].study_id
            volume_features = self.volume_encoder.encode(first_study_id)
            pathology_features = self.pathology_encoder.encode(first_study_id)
        else:
            volume_features = {"study_id": "none"}
            pathology_features = {"slide_id": "none"}

        fused = self.fusion.fuse(
            {
                "text": text_features,
                "volume": volume_features,
                "pathology": pathology_features,
            }
        )
        retrieval = retriever.retrieve(case.report_text or case.ehr_text or case.case_id)
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
            },
            coherence=0.9,
            risk=0.1,
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
            {"source": "retrieval", "kind": retrieval["status"], "value": retrieval["evidence_count"]},
            {"source": "fusion", "kind": "engine", "value": fused.get("engine", fused.get("provider", "none"))},
            {"source": "service", "kind": "provider", "value": resolved["service"].get("provider", "none")},
            {"source": "intuition", "kind": "candidate", "value": intuition_engine.summarize_for_trace(intuition)},
        ]
        evidence.extend(ranked_evidence)
        coordinated = coordinator.run(
            case_id=case.case_id,
            evidence=evidence,
            risk=0.2,
            uncertainty=0.1,
        )
        critique_result = self.critique.review({"coordinated": coordinated, "intuition": intuition, "areas": area_signals, "area_dynamics": area_dynamics, "dream": dream})
        return {
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
            "services": resolved,
            "intuition": intuition,
            "coordinated": coordinated,
            "critique": critique_result,
            "quantum_allowed": quantum_allowed,
            "dream": dream,
        }
