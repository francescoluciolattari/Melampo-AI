from melampo.reasoning.metacognition import MetacognitiveController
from melampo.config import build_default_config


def test_metacognition_threshold():
    config = build_default_config()
    controller = MetacognitiveController(config=config)
    assert controller.should_abstain(0.8) is True
    assert controller.should_abstain(0.1) is False
