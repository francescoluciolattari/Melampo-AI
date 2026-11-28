from dataclasses import dataclass
from typing import Any


@dataclass
class SupportContradictionAnalyzer:
    """Build lightweight support and contradiction signals from evidence and the discordances of the case's reading."""

    GROUNDED_SUPPORT_SOURCES = {
        "bundle",
        "retrieval",
        "fusion",
        "service",
        "semantic_memory",
        "vector_memory",
        "weaviate",
        "knowledge_graph",
        "clinical_document_processor",
    }

    def _payload(self, item: dict) -> dict:
        if isinstance(item, dict) and isinstance(item.get("item"), dict):
            return item["item"]
        return item if isinstance(item, dict) else {}

    def analyze(
        self,
        evidence: list,
        intuition: dict | None = None,
        nexus: dict | None = None,
        discordances: dict | None = None,
    ) -> dict:
        nexus = nexus or {}
        discordances = discordances or {}
        rehearsal_profile = nexus.get("rehearsal_profile", {})

        support_profiles: list[dict[str, Any]] = []
        contradiction_profiles: list[dict[str, Any]] = []

        for item in evidence:
            payload = self._payload(item)
            if not payload:
                continue
            source = payload.get("source", "unknown")
            kind = payload.get("kind", "signal")
            grounding_score = float(payload.get("grounding_score", 0.5))
            if source in self.GROUNDED_SUPPORT_SOURCES:
                support_profiles.append(
                    {
                        "label": f"{source}:{kind}",
                        "class": "grounded_support",
                        "strength": round(0.15 + min(grounding_score, 1.0) * 0.18, 3),
                    }
                )
            if payload.get("contradiction_score", 0.0):
                contradiction_profiles.append(
                    {
                        "label": f"{source}:explicit_contradiction",
                        "class": "useful_contradiction",
                        "strength": round(
                            min(float(payload.get("contradiction_score", 0.0)), 1.0)
                            * 0.4,
                            3,
                        ),
                    }
                )

        # what the reading of the case sent to a person (reasoning.case_discordances), one profile per kind.
        # Until 28 November 2026 these were the "mismatch" and "coherence" pairs between areas, a lexical
        # measure with hand-set weights, never measured.
        for kind, count in sorted((discordances.get("kinds") or {}).items()):
            if count:
                contradiction_profiles.append(
                    {
                        "label": f"reading:{kind}_discordance",
                        "class": "useful_contradiction",
                        "strength": 0.35,
                        "count": int(count),
                    }
                )

        if rehearsal_profile.get("boundary_case_hint", False):
            contradiction_profiles.append(
                {
                    "label": "nexus:boundary_case",
                    "class": "weak_contradiction",
                    "strength": 0.15,
                }
            )

        support_signals = [item["label"] for item in support_profiles]
        contradiction_signals = [item["label"] for item in contradiction_profiles]
        support_strength = round(
            sum(float(item.get("strength", 0.0) or 0.0) for item in support_profiles), 3
        )
        contradiction_strength = round(
            sum(
                float(item.get("strength", 0.0) or 0.0)
                for item in contradiction_profiles
            ),
            3,
        )

        return {
            "support_profiles": support_profiles,
            "contradiction_profiles": contradiction_profiles,
            "support_signals": support_signals,
            "contradiction_signals": contradiction_signals,
            "support_strength": support_strength,
            "contradiction_strength": contradiction_strength,
        }
