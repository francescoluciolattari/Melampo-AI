"""Reconcile one-shot (RAG) and recursive (RLM) retrieval into a single dossier.

Running both strategies is only worth its cost because they fail differently.
One-shot hybrid search fails by omission: the relevant chunk falls outside the
``top_k`` cut and is never seen. Recursive retrieval fails by overreach: the root
model composes fragments into a claim broader than any single fragment supports,
with every citation individually valid. One under-claims, the other over-claims.

That asymmetry is what makes their disagreement informative. This module treats
divergence between the two paths as an empirical uncertainty estimate and feeds
it to the neuro-dynamic conflict metrics, which until now were derived from
internal heuristics alone.

The reconciliation is deliberately deterministic. No model adjudicates between
the two paths; a claim surviving only on the recursive path must present
verifiable offsets to be admitted, and unresolved contradiction raises conflict
rather than being silently resolved in favour of either side.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..memory.retrieval_contract import RETRIEVAL_MODE_DUAL

DISPOSITION_CONFIRMED = "confirmed_by_both"
DISPOSITION_RAG_ONLY = "one_shot_only"
DISPOSITION_RLM_ONLY_VERIFIED = "recursive_only_verified"
DISPOSITION_RLM_ONLY_UNVERIFIED = "recursive_only_unverified"


@dataclass
class ReconciliationVerdict:
    """Outcome of merging the two retrieval paths."""

    evidence: list[dict[str, Any]] = field(default_factory=list)
    discarded: list[dict[str, Any]] = field(default_factory=list)
    agreement_ratio: float = 0.0
    conflict_signal: float = 0.0
    recall_gain: int = 0
    overreach_blocked: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "agreement_ratio": round(self.agreement_ratio, 3),
            "conflict_signal": round(self.conflict_signal, 3),
            "recall_gain": self.recall_gain,
            "overreach_blocked": self.overreach_blocked,
            "evidence_count": len(self.evidence),
            "discarded_count": len(self.discarded),
            "notes": list(self.notes),
        }


def reconcile(
    one_shot: dict[str, Any],
    recursive: dict[str, Any],
    *,
    require_offsets_for_recursive_only: bool = True,
) -> ReconciliationVerdict:
    """Merge a one-shot and a recursive retrieval payload into one verdict.

    Disposition matrix:

    - found by both        -> confirmed, highest confidence
    - one-shot only        -> admitted; fast path reached it first
    - recursive only, with verifiable offsets -> admitted as recall gain
    - recursive only, without offsets         -> discarded as probable overreach
    """
    rag_evidence = _evidence_of(one_shot)
    rlm_evidence = _evidence_of(recursive)

    rag_index = {_identity(item): item for item in rag_evidence}
    rlm_index = {_identity(item): item for item in rlm_evidence}

    verdict = ReconciliationVerdict()
    confirmed = 0

    for key, item in rag_index.items():
        merged = dict(item)
        if key in rlm_index:
            merged["reconciliation"] = DISPOSITION_CONFIRMED
            merged["grounding_score"] = _boost(item, rlm_index[key])
            confirmed += 1
        else:
            merged["reconciliation"] = DISPOSITION_RAG_ONLY
        verdict.evidence.append(merged)

    for key, item in rlm_index.items():
        if key in rag_index:
            continue
        merged = dict(item)
        if not require_offsets_for_recursive_only or _has_offsets(item):
            merged["reconciliation"] = DISPOSITION_RLM_ONLY_VERIFIED
            verdict.evidence.append(merged)
            verdict.recall_gain += 1
        else:
            merged["reconciliation"] = DISPOSITION_RLM_ONLY_UNVERIFIED
            verdict.discarded.append(merged)
            verdict.overreach_blocked += 1

    union = len(rag_index) + len([key for key in rlm_index if key not in rag_index])
    verdict.agreement_ratio = confirmed / union if union else 0.0
    verdict.conflict_signal = _conflict_signal(verdict, union)

    for index, item in enumerate(verdict.evidence, start=1):
        item["rank"] = index

    if verdict.recall_gain:
        verdict.notes.append(f"{verdict.recall_gain} finding(s) recovered only by recursive retrieval")
    if verdict.overreach_blocked:
        verdict.notes.append(f"{verdict.overreach_blocked} unverifiable recursive finding(s) discarded")
    if union and not confirmed:
        verdict.notes.append("no overlap between retrieval paths; treat conclusions as low confidence")

    return verdict


def build_dual_path_payload(
    one_shot: dict[str, Any],
    recursive: dict[str, Any],
    verdict: ReconciliationVerdict,
    *,
    coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render a reconciled payload in the shared retrieval contract shape."""
    evidence = verdict.evidence
    mean_grounding = (
        sum(float(item.get("grounding_score", 0.0) or 0.0) for item in evidence) / len(evidence) if evidence else 0.0
    )
    coverage_ratio = float((coverage or {}).get("coverage_ratio", 0.0))

    return {
        "query": one_shot.get("query") or recursive.get("query", ""),
        "focus": one_shot.get("focus") or recursive.get("focus", "general"),
        "target_areas": one_shot.get("target_areas") or recursive.get("target_areas", []),
        "status": "grounded_retrieval_ready" if evidence else "insufficient_grounded_evidence",
        "retrieval_mode": RETRIEVAL_MODE_DUAL,
        "evidence": evidence,
        "evidence_count": len(evidence),
        "retrieval_quality": {
            "memory_backed": True,
            "coverage": round(coverage_ratio, 4),
            "mean_grounding_score": round(mean_grounding, 3),
            "fallback_used": False,
            "agreement_ratio": round(verdict.agreement_ratio, 3),
            "conflict_signal": round(verdict.conflict_signal, 3),
            "recall_gain": verdict.recall_gain,
            "overreach_blocked": verdict.overreach_blocked,
        },
        "case_context_keys": one_shot.get("case_context_keys", []),
        "reconciliation": verdict.as_dict(),
    }


def conflict_inputs_for_neuro_dynamics(verdict: ReconciliationVerdict) -> dict[str, float]:
    """Expose the reconciliation as inputs to the neuro-dynamic conflict metrics.

    ``conflict_load`` and ``revision_pressure`` are currently derived from area
    mismatch heuristics. Retrieval-path divergence gives them a second, empirical
    source grounded in two systems that fail in opposite directions.
    """
    return {
        "retrieval_conflict_signal": round(verdict.conflict_signal, 3),
        "retrieval_agreement_ratio": round(verdict.agreement_ratio, 3),
        "retrieval_overreach_blocked": float(verdict.overreach_blocked),
    }


def _evidence_of(payload: dict[str, Any]) -> Sequence[dict[str, Any]]:
    evidence = payload.get("evidence") if isinstance(payload, dict) else None
    if not isinstance(evidence, list):
        return []
    return [item for item in evidence if isinstance(item, dict)]


def _identity(item: dict[str, Any]) -> str:
    record_id = item.get("record_id")
    if record_id:
        return str(record_id)
    provenance = item.get("provenance")
    if isinstance(provenance, dict) and provenance.get("document_id"):
        return f"{provenance['document_id']}:{provenance.get('char_start')}-{provenance.get('char_end')}"
    text = str(item.get("text", "")).strip().lower()
    return text[:160] or f"anonymous:{id(item)}"


def _has_offsets(item: dict[str, Any]) -> bool:
    provenance = item.get("provenance")
    if not isinstance(provenance, dict):
        return False
    start = provenance.get("char_start")
    end = provenance.get("char_end")
    return isinstance(start, int) and isinstance(end, int) and end > start


def _boost(first: dict[str, Any], second: dict[str, Any]) -> float:
    scores = [
        float(first.get("grounding_score", 0.0) or 0.0),
        float(second.get("grounding_score", 0.0) or 0.0),
    ]
    best = max(scores)
    return round(min(1.0, best + 0.05), 3)


def _conflict_signal(verdict: ReconciliationVerdict, union: int) -> float:
    if union <= 0:
        return 0.0
    divergence = 1.0 - verdict.agreement_ratio
    overreach_weight = verdict.overreach_blocked / union
    return round(min(1.0, divergence * 0.7 + overreach_weight * 0.3), 3)
