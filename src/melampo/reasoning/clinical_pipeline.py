from dataclasses import dataclass

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
from .differential_engine import DifferentialEngine
from .escalation import EscalationPolicy
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
        evidence = [
            {"source": "bundle", "kind": "bundle_keys", "value": list(bundle.keys())},
            {"source": "retrieval", "kind": retrieval["status"], "value": retrieval["evidence_count"]},
            {"source": "fusion", "kind": "engine", "value": fused.get("engine", fused.get("provider", "none"))},
            {"source": "service", "kind": "provider", "value": resolved["service"].get("provider", "none")},
        ]
        evidence.extend(ranked_evidence)
        coordinated = coordinator.run(
            case_id=case.case_id,
            evidence=evidence,
            risk=0.2,
            uncertainty=0.1,
        )
        critique_result = self.critique.review({"coordinated": coordinated})
        quantum_allowed = quantum_gate.allow(contextuality_score=0.7)
        dream = dream_trainer.run(
            case_context={"case_id": case.case_id, "bundle_keys": list(bundle.keys())},
            coherence=0.9,
            risk=0.1,
        )
        return {
            "case_id": case.case_id,
            "bundle_keys": list(bundle.keys()),
            "text_features": text_features,
            "volume_features": volume_features,
            "pathology_features": pathology_features,
            "fused": fused,
            "retrieval": retrieval,
            "ranked_evidence": ranked_evidence,
            "services": resolved,
            "coordinated": coordinated,
            "critique": critique_result,
            "quantum_allowed": quantum_allowed,
            "dream": dream,
        }
