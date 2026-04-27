from __future__ import annotations

from dataclasses import dataclass

from .config import RuntimeConfig, build_default_config
from .data.ingestion import ClinicalIngestionPipeline
from .data.normalization import ClinicalNormalizer
from .evaluation.validation_matrix import ValidationMatrix
from .memory.episodic_memory import EpisodicMemoryStore
from .memory.knowledge_graph import KnowledgeGraphClient
from .memory.semantic_memory import SemanticMemoryStore
from .models.fusion import MultimodalFusionEngine
from .models.pathology_encoder import PathologyEncoder
from .models.quantum_research import QuantumResearchLayer
from .models.report_encoder import ClinicalTextEncoder
from .models.volume_encoder import VolumeEncoder
from .orchestration.model_router import ModelRouter
from .reasoning.clinical_pipeline import ClinicalInferencePipeline
from .reasoning.critique_loop import CritiqueLoop
from .reasoning.metacognition import MetacognitiveController
from .reasoning.workspace import DifferentialWorkspace
from .training.generative_replay import GenerativeReplayEngine
from .utils.logging_utils import build_logger


@dataclass(slots=True)
class MelampoRuntime:
    config: RuntimeConfig
    pipeline: ClinicalInferencePipeline
    validator: ValidationMatrix

    def describe(self) -> dict:
        return {
            "config": self.config.describe(),
            "pipeline": self.pipeline.__class__.__name__,
            "validator": self.validator.summarize(),
        }


def build_default_runtime(config: RuntimeConfig | None = None) -> MelampoRuntime:
    """Assemble the default research runtime for Melampo."""
    config = config or build_default_config()
    logger = build_logger("melampo")

    ingestion = ClinicalIngestionPipeline()
    normalizer = ClinicalNormalizer()
    router = ModelRouter(config=config, logger=logger)

    volume_encoder = VolumeEncoder(config=config)
    pathology_encoder = PathologyEncoder(config=config)
    text_encoder = ClinicalTextEncoder(config=config)
    fusion = MultimodalFusionEngine(config=config)

    episodic_memory = EpisodicMemoryStore()
    semantic_memory = SemanticMemoryStore()
    knowledge_graph = KnowledgeGraphClient(config=config)
    workspace = DifferentialWorkspace()
    critique = CritiqueLoop(config=config, logger=logger)
    metacognition = MetacognitiveController(config=config)
    quantum_layer = QuantumResearchLayer(config=config)
    replay = GenerativeReplayEngine(config=config, logger=logger)

    pipeline = ClinicalInferencePipeline(
        ingestion=ingestion,
        normalizer=normalizer,
        router=router,
        volume_encoder=volume_encoder,
        pathology_encoder=pathology_encoder,
        text_encoder=text_encoder,
        fusion=fusion,
        episodic_memory=episodic_memory,
        semantic_memory=semantic_memory,
        knowledge_graph=knowledge_graph,
        workspace=workspace,
        critique=critique,
        metacognition=metacognition,
        quantum_layer=quantum_layer,
        replay_engine=replay,
        logger=logger,
    )

    validator = ValidationMatrix(config=config, logger=logger)
    return MelampoRuntime(config=config, pipeline=pipeline, validator=validator)
