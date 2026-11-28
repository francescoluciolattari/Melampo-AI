from dataclasses import dataclass

from .case_signals import rules_of


@dataclass
class PolicyStack:
    """The policy of the case from its signals (``reasoning.case_signals``): explicit rules, no weights.

    Until 28 November 2026 three gates compared a hand-weighted risk and uncertainty with hand-set thresholds
    (``AbstentionPolicy`` 0.65, ``RiskGate`` 0.35, ``EscalationPolicy`` 0.35/0.65), never measured; removed. Nothing
    here withholds a result: a case escalates to a person for a reason written in a sentence, and its flags go on the
    result (principle of doubt).
    """

    def evaluate(self, signals: dict | None) -> dict:
        signals = signals or {}
        escalate = bool(signals.get("escalate", False))
        return {
            "abstain": False,
            "allow": True,
            "escalate": escalate,
            "reasons": rules_of(signals),
            "flags": rules_of(signals, "flags"),
            "sentences": [item["sentence"] for item in signals.get("reasons", []) + signals.get("flags", [])],
            "decision_band": "guarded" if escalate else "clear",
        }
