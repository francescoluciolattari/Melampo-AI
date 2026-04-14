from dataclasses import dataclass


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
        return {"case_id": case.case_id, "bundle_keys": list(bundle.keys())}
