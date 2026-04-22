from dataclasses import dataclass


@dataclass
class CaseContextArea:
    def integrate(self, case_context: dict | None = None) -> dict:
        case_context = case_context or {}
        return {"area": "case_context", "case_context": case_context, "keys": sorted(case_context.keys())}
