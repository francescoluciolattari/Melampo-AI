"""Wire the recursive engine to the semantic memory and to the audit store.

Two small pieces of plumbing that carry two invariants established elsewhere.

**The environment inherits the quarantine.** `ContextEnvironment` accepts an
injectable search function. Binding it to `WeaviateEnterpriseMemoryAdapter.hybrid_search`
means the engine reaches memory through the same call that already refuses
quarantined classes — so a synthetic candidate is unreachable from inside the
recursive loop for the same reason it is unreachable from the one-shot path.
This is verified by attempt, not by reading the adapter: a test stores a
candidate and asserts the engine cannot surface it.

**Trajectories are health records.** A trajectory holds fragments of the case
with their offsets, sub-model prompts built from case text, and the model's own
navigation. When the case is real that is clinical data, and it goes to the
audit store marked as such — not to a log that a rotation policy will discard
or an access policy will treat as operational telemetry.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..memory.context_environment import EnvironmentDocument
from .rlm_engine import DATA_CLASS_REAL, Trajectory

AUDIT_EVENT_TRAJECTORY = "rlm_trajectory"


def search_via_adapter(adapter: Any, *, limit_multiplier: int = 1) -> Callable[[str, int], Sequence[dict[str, Any]]]:
    """Bind the environment's search primitive to the adapter's hybrid search.

    Hits are reshaped to what `ContextEnvironment._fragment_from_hit` expects:
    a document identifier and, when the adapter provides them, character
    offsets and a score. Quarantine exclusion happens inside `hybrid_search`
    and is not repeated here — repeating a safety check in two places invites
    the two to drift apart.
    """

    def _search(query: str, limit: int) -> Sequence[dict[str, Any]]:
        payload = adapter.hybrid_search(query=query, limit=limit * limit_multiplier)
        if payload.get("status") != "completed":
            return []
        reshaped: list[dict[str, Any]] = []
        for hit in payload.get("hits", []):
            metadata = hit.get("metadata") or {}
            document_id = hit.get("document_id") or metadata.get("document_id") or hit.get("record_id")
            if not document_id:
                continue
            reshaped.append(
                {
                    "document_id": str(document_id),
                    "char_start": int(metadata.get("char_start", 0) or 0),
                    "char_end": int(metadata.get("char_end", 0) or 0),
                    "score": hit.get("score"),
                }
            )
        return reshaped

    return _search


def graph_expand_via_adapter(adapter: Any) -> Callable[[str, int], Sequence[dict[str, Any]]]:
    def _expand(concept: str, depth: int) -> Sequence[dict[str, Any]]:
        payload = adapter.graph_expand(object_key=concept, depth=depth)
        if not isinstance(payload, dict):
            return []
        neighbours = payload.get("neighbours") or payload.get("hits") or payload.get("results") or []
        return [item for item in neighbours if isinstance(item, dict)]

    return _expand


def documents_from_adapter_store(adapter: Any, *, data_class: str) -> list[EnvironmentDocument]:
    """Materialise the adapter's stored records as environment documents.

    ``data_class`` is required rather than read from the records, because the
    engine refuses unmarked documents and the caller is the one who knows what
    corpus was loaded. Quarantined records are excluded here as well as at
    search time, so that even a `describe()` of the environment does not reveal
    a candidate's existence.
    """
    documents: list[EnvironmentDocument] = []
    store = getattr(adapter, "fallback_store", None)
    records = getattr(store, "records", {}) or {}
    for record in records.values():
        metadata = dict(getattr(record, "metadata", {}) or {})
        if adapter.schema.is_quarantined(metadata.get("class_name")):
            continue
        text = str(getattr(record, "text", "") or "")
        record_id = str(metadata.get("document_id") or getattr(record, "record_id", "") or "")
        if not text or not record_id:
            continue
        documents.append(
            EnvironmentDocument(
                document_id=str(record_id),
                text=str(text),
                source=str(metadata.get("source_type", "semantic_memory")),
                metadata={**metadata, "data_class": data_class},
            )
        )
    return documents


@dataclass
class TrajectoryAuditWriter:
    """Append trajectories to the audit store, marked as health data when they are."""

    audit_store: Any

    def write(self, trajectory: Trajectory, *, operator: str | None = None) -> dict[str, Any]:
        payload = trajectory.as_dict()
        metadata = {
            "health_data": trajectory.data_class == DATA_CLASS_REAL,
            "data_class": trajectory.data_class,
            "retention_class": "clinical_record" if trajectory.data_class == DATA_CLASS_REAL else "research",
            "operator": operator,
            "completed": trajectory.completed,
            "stop_reason": trajectory.stop_reason,
        }
        return self.audit_store.append(AUDIT_EVENT_TRAJECTORY, payload, metadata)
