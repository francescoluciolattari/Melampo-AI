"""The intuition engine, rapid part (28 November 2026): the candidates the case's findings reach in the graph.

The rapid part is ``rank_differential``: candidate diseases ranked by the summed information content of the findings
they actually share (``memory.differential_ranking``). The same candidates go to ``case_diagnosis`` as proposals,
where they are checked against the profiles from certain sources; here they become the differential's first
hypothesis. Nothing in this part is weighted by hand.

Until 28 November 2026 the engine chose among three "modes" (rapid intuition, rational revision, contradiction
revision) by sums of twenty metrics with hand-set weights, all derived from the coherence between "areas"
(``area_coherence``, a lexical overlap never measured against anything), with an optional "quantum-like" belief
update. Removed, with their files.

The lateral part (``reasoning.lateral_intuition``, 28 November 2026) widens the spectrum when the diagnosis is
unclear: salience, generation (every profile that accounts for an observation, and pairs bound by waves), evaluation
against the whole case by the energy of the profiles from certain sources, and the guard (a proposal enters only as a
hypothesis to exclude). It runs inside the diagnostic path, where the case graph and the profiles are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..utils.values import safe_float as _safe_float

RAPID = "rank_differential"


@dataclass
class IntuitionEngine:
    """The rapid part: ``rank_differential``'s candidates, in its order."""

    max_candidates: int = 5

    def infer(
        self,
        case_id: str,
        ranked_evidence: list,
        nexus: dict | None = None,
        graph_candidates: list | None = None,
    ) -> dict[str, Any]:
        """The candidates of ``rank_differential`` (``graph_candidates``), in its order.

        Without graph candidates (no findings, or none resolved in the graph) the positions of the ranked evidence
        stay as placeholders, ``candidate_N``: the differential engine knows them and puts a hypothesis found in
        the graph by the nexus branch before them.
        """
        nexus = nexus or {}
        candidates: list[dict[str, Any]] = []
        for item in (graph_candidates or [])[: self.max_candidates]:
            if not isinstance(item, dict) or not item.get("condition"):
                continue
            candidates.append(
                {
                    "label": str(item["condition"]),
                    "score": round(_safe_float(item.get("specificity_score")), 4),
                    "coverage": round(_safe_float(item.get("coverage")), 4),
                    "matched_findings": list(item.get("matched_findings") or []),
                    "unmatched_findings": list(item.get("unmatched_findings") or []),
                    "source": RAPID,
                }
            )
        if not candidates:
            for index, item in enumerate(ranked_evidence[:3]):
                candidates.append(
                    {
                        "label": f"candidate_{index + 1}",
                        "score": round(_safe_float((item or {}).get("weight"), 1.0), 4),
                        "source": str(((item or {}).get("item") or {}).get("source", "unknown")),
                    }
                )
        return {
            "case_id": case_id,
            "intuition": candidates[0]["label"] if candidates else "no_candidate",
            "source": RAPID if candidates and candidates[0]["source"] == RAPID else "placeholder",
            "candidate_scores": candidates,
            "nexus_alternatives": list(nexus.get("alternative_hypotheses", []) or []),
        }

    def summarize_for_trace(self, intuition_payload: dict) -> dict:
        return {
            "intuition": intuition_payload.get("intuition", "none"),
            "source": intuition_payload.get("source", "none"),
            "candidate_count": len(intuition_payload.get("candidate_scores", [])),
        }


__all__ = ["RAPID", "IntuitionEngine"]
