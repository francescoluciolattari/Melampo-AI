from dataclasses import dataclass


@dataclass
class CaseContextArea:
    def integrate(self, case_context: dict | None = None) -> dict:
        case_context = case_context or {}
        keys = sorted(case_context.keys())
        return {
            "area": "case_context",
            "focus": "multimodal_context",
            "case_context": case_context,
            "keys": keys,
            "signal_count": len(keys),
            "salience_score": round(0.1 * len(keys), 3),
        }
