from dataclasses import dataclass


@dataclass
class ReplayFilter:
    """Replay filter for accepting and characterizing synthetic replay cases."""

    min_coherence: float = 0.7
    max_risk: float = 0.3

    def assess(self, coherence: float, risk: float) -> dict:
        accepted = coherence >= self.min_coherence and risk <= self.max_risk
        coherence_margin = round(coherence - self.min_coherence, 3)
        risk_margin = round(self.max_risk - risk, 3)
        acceptance_score = round(coherence_margin + risk_margin, 3)
        replay_mode = "stabilizing_replay"
        if not accepted:
            replay_mode = "corrective_replay"
        elif coherence < (self.min_coherence + 0.1) or risk > (self.max_risk - 0.1):
            replay_mode = "boundary_replay"
        return {
            "accepted": accepted,
            "coherence_margin": coherence_margin,
            "risk_margin": risk_margin,
            "acceptance_score": acceptance_score,
            "replay_mode": replay_mode,
        }

    def accept(self, coherence: float, risk: float) -> bool:
        return self.assess(coherence=coherence, risk=risk)["accepted"]
