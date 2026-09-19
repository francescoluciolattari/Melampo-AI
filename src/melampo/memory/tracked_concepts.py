"""Which concepts the literature refresh keeps current, and in what order.

Answers a question raised directly: not every one of the graph's 29,053
concepts, and not a blind crawl of all of them -- that count is the graph's
total size, not a sensible literature-refresh scope, and was cited earlier
only to illustrate why an unbounded nightly crawl is the wrong shape at all.
The chosen strategy is Nexus-Engine style: bounded, nightly, working through
concepts this project has actually reasoned about, growing as real use
grows it -- never touching the full ontology.

**A queue, not a list.** A bare set of concept names cannot answer "which
ones are overdue", and PubMed's unauthenticated rate limit (3 requests per
second) means a nightly run can only ever touch a bounded slice regardless
of how many concepts are tracked. Every entry carries when it was added, why
(which part of the system asked for it), and when it was last refreshed --
enough for `next_batch` to prioritise concepts that have never been
refreshed, then the ones refreshed longest ago, which is what "keep this
current" has to mean once a full daily sweep of everything is not possible.

**Growth sources are the same real signals this project already produces.**
A concept enters this file when the vetting bench names it, when a case is
run, or when an edge is promoted -- not from a separate curation pass. The
file grows the same way `graph_store`'s learned layer and
`ConceptDescriptionStore` do: from what the system actually does, not from
someone maintaining a list by hand.
"""

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SOURCE_VETTING_BENCH = "vetting_bench"
SOURCE_CONFIRMED_CASE = "confirmed_case"
SOURCE_PROMOTED_EDGE = "promoted_edge"
SOURCE_MANUAL = "manual"


@dataclass
class TrackedConcept:
    """One concept the literature refresh keeps current, and its refresh history."""

    concept: str
    added_from: str
    added_at: float
    last_refreshed_at: float | None = None
    passage_count: int = 0

    @property
    def is_due(self) -> bool:
        """Never refreshed is always due; a refreshed concept is due again on the next sweep that reaches it."""
        return self.last_refreshed_at is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "concept": self.concept,
            "added_from": self.added_from,
            "added_at": self.added_at,
            "last_refreshed_at": self.last_refreshed_at,
            "passage_count": self.passage_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrackedConcept":
        return cls(
            concept=str(payload["concept"]),
            added_from=str(payload.get("added_from", SOURCE_MANUAL)),
            added_at=float(payload.get("added_at", 0.0)),
            last_refreshed_at=payload.get("last_refreshed_at"),
            passage_count=int(payload.get("passage_count", 0)),
        )


@dataclass
class TrackedConceptStore:
    """The full tracked-concept list, loaded from and saved to one JSON file.

    A single JSON object rather than JSONL: unlike the append-only stores
    elsewhere in this project (learned edges, term history), this file is
    read and rewritten as a whole on every update -- refreshing a concept
    changes an existing entry's timestamp, it does not append a new fact
    that must never be lost. There is nothing here an audit trail would
    protect; there is a queue that needs its current state saved.
    """

    concepts: dict[str, TrackedConcept] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.concepts)

    def track(self, concept: str, *, source: str) -> bool:
        """Add a concept if it is not already tracked. Returns whether it was newly added.

        Re-adding an already-tracked concept is a no-op rather than resetting
        its refresh history -- a concept the vetting bench already tracks
        that a later confirmed case also touches should not lose the fact
        that it was already refreshed once.
        """
        key = concept.strip().lower()
        if not key or key in self.concepts:
            return False
        self.concepts[key] = TrackedConcept(concept=concept.strip(), added_from=source, added_at=time.time())
        return True

    def track_many(self, concepts: Iterable[str], *, source: str) -> int:
        return sum(1 for concept in concepts if self.track(concept, source=source))

    def mark_refreshed(self, concept: str, *, passage_count: int) -> None:
        entry = self.concepts.get(concept.strip().lower())
        if entry is not None:
            entry.last_refreshed_at = time.time()
            entry.passage_count = passage_count

    def next_batch(self, limit: int) -> list[TrackedConcept]:
        """The next concepts due for refresh, oldest-refreshed first.

        Never-refreshed concepts sort first (their sort key is negative
        infinity, ahead of any real timestamp), then the ones refreshed
        longest ago. `limit` is where PubMed's unauthenticated rate limit
        actually bites: 3 requests per second bounds how many concepts one
        nightly run can touch regardless of how many are tracked, so a
        caller passes the batch size its own rate budget allows rather than
        this store trying to guess it.
        """
        ordered = sorted(
            self.concepts.values(),
            key=lambda item: item.last_refreshed_at if item.last_refreshed_at is not None else -1.0,
        )
        return ordered[:limit]

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "concepts": [entry.as_dict() for entry in self.concepts.values()],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(path)

    @classmethod
    def load(cls, path: Path) -> "TrackedConceptStore":
        """Load the tracked-concept file, or start empty if it does not exist yet.

        A missing file is the normal state before the first concept has ever
        been tracked -- not an error, the same posture every other loader in
        this project takes toward its own not-yet-created file.
        """
        store = cls()
        if not path.exists():
            return store
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in payload.get("concepts", []):
            tracked = TrackedConcept.from_dict(entry)
            store.concepts[tracked.concept.strip().lower()] = tracked
        return store


def seed_from_vetting_bench(store: TrackedConceptStore, cases: Sequence[Any]) -> int:
    """Track every factor and target the vetting bench's cases name.

    A concrete starting point with zero curation effort: these ~40 concepts
    are already known to matter, since real cases are already built around
    them, and give the nightly refresh something meaningful to do from the
    first run rather than starting from an empty queue.
    """
    concepts: list[str] = []
    for case in cases:
        concepts.append(case.factor)
        concepts.append(case.target)
    return store.track_many(concepts, source=SOURCE_VETTING_BENCH)
