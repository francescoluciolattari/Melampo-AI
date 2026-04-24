from dataclasses import dataclass


@dataclass
class RiskGate:
    """Risk gate for abstention and escalation."""

    threshold: float = 0.35

    def assess(self, risk: float) -> dict:
        allow = risk <= self.threshold
        margin = round(self.threshold - risk, 3)
        band = "high" if risk > (self.threshold + 0.15) else "guarded" if not allow else "low"
        reasons = ["risk_above_gate_threshold"] if not allow else []
        return {
            "allow": allow,
            "threshold": self.threshold,
            "risk": risk,
            "margin": margin,
            "band": band,
            "reasons": reasons,
        }

    def allow(self, risk: float) -> bool:
        return self.assess(risk=risk)["allow"]
