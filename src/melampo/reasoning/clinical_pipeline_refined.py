from dataclasses import dataclass

from ..evaluation.quantum_gate import QuantumResearchGate
from ..memory.retriever import MemoryRetriever
from ..orchestration.runtime_services import RuntimeServices
from ..training.counterfactual_sampler import CounterfactualSampler
from ..training.dream_trainer import DreamTrainer
from ..training.replay_filter import ReplayFilter
from ..models.quantum_belief_layer import QuantumBeliefLayer
from .critique_loop import CritiqueLoop
from .pipeline_coordinator import PipelineCoordinator


@dataclass
class RefinedClinicalInferencePipeline:
    ingestion: object
    normalizer: object
    runtime_services: RuntimeServices
    retriever: MemoryRetriever
    coordinator: PipelineCoordinator
    critique: CritiqueLoop
    quantum_gate: QuantumResearchGate
    dream_trainer: DreamTrainer

    def run(self, payload: dict) -> dict:
        case = self.ingestion.from_payload(payload)
        bundle = self.normalizer.to_fhir_bundle(case)
        retrieval = self.retriever.retrieve(case.report_text or case.ehr_text or case.case_id)
        resolved = self.runtime_services.resolve("volume_encoder")
        evidence = [
            f"bundle:{','.join(bundle.keys())}",
            f"retrieval:{retrieval['status']}",
            f"service:{resolved['service'].get('provider', 'none')}",
        ]
        coordinated = self.coordinator.run(
            case_id=case.case_id,
            evidence=evidence,
            risk=0.2,
            uncertainty=0.1,
        )
        critique = self.critique.review({"coordinated": coordinated})
        quantum_allowed = self.quantum_gate.allow(contextuality_score=0.7)
        dream = self.dream_trainer.run(
            case_context={"case_id": case.case_id, "bundle_keys": list(bundle.keys())},
            coherence=0.9,
            risk=0.1,
        )
        return {
            "case_id": case.case_id,
            "bundle_keys": list(bundle.keys()),
            "retrieval": retrieval,
            "services": resolved,
            "coordinated": coordinated,
            "critique": critique,
            "quantum_allowed": quantum_allowed,
            "dream": dream,
        }


def build_default_refined_pipeline(ingestion: object, normalizer: object, runtime_services: RuntimeServices, coordinator: PipelineCoordinator, critique: CritiqueLoop) -> RefinedClinicalInferencePipeline:
    return RefinedClinicalInferencePipeline(
        ingestion=ingestion,
        normalizer=normalizer,
        runtime_services=runtime_services,
        retriever=MemoryRetriever(),
        coordinator=coordinator,
        critique=critique,
        quantum_gate=QuantumResearchGate(),
        dream_trainer=DreamTrainer(
            replay_filter=ReplayFilter(),
            sampler=CounterfactualSampler(),
            belief_layer=QuantumBeliefLayer(),
        ),
    )
