from dataclasses import dataclass, field


@dataclass
class PipelineState:
    """Shared runtime state for the clinical inference pipeline."""

    case_id: str = ""
    evidence: list = field(default_factory=list)
    hypotheses: list = field(default_factory=list)
    abstain: bool = False
    escalate: bool = False
    reasons: list = field(default_factory=list)
    flags: list = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "case_id": self.case_id,
            "evidence_count": len(self.evidence),
            "hypothesis_count": len(self.hypotheses),
            "abstain": self.abstain,
            "escalate": self.escalate,
            "reasons": list(self.reasons),
            "flags": list(self.flags),
        }
