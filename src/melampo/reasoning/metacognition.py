from dataclasses import dataclass


@dataclass
class MetacognitiveController:
    """Carries the runtime configuration to the pipeline.

    Its ``should_abstain`` (uncertainty >= 0.65, a threshold chosen by hand and never measured) was removed on
    28 November 2026: the case's policy is in ``reasoning.case_signals``.
    """

    config: object
