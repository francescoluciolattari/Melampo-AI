from dataclasses import dataclass, field


@dataclass
class PipelineState:
    """Shared runtime state for the clinical inference pipeline."""

    case_id: str = ""
    evidence: list = field(default_factory=list)
    hypotheses: list = field(default_factory=list)
    uncertainty: float = 0.0
    risk: float = 0.0
    abstain: bool = False
    escalate: bool = False

    def summary(self) -> dict:
        return {
            "case_id": self.case_id,
            "evidence_count": len(self.evidence),
            "hypothesis_count": len(self.hypotheses),
            "risk": self.risk,
            "uncertainty": self.uncertainty,
            "abstain": self.abstain,
            "escalate": self.escalate,
        }
