"""Persist the concept graph, so what the system learns survives a restart.

Identified directly as the bottleneck the rest of the evolution depends on:
`InMemoryConceptGraph` is a Python list, rebuilt from the HPO file on every
start. An edge promoted by `ConjectureLedger` after three independent
confirmations exists until the process exits and is then silently lost --
which makes the dream-replay work, the confirmation ledger, and any weight
adjustment from real outcomes ceremonial rather than real.

**The central design choice: learned edges are stored apart from imported
ones, never merged into one file.** The imported layer is derivable -- it can
be rebuilt from the HPO release at any time, and re-importing a newer release
is a normal operation. The learned layer cannot be rebuilt from anything; it
is the accumulated product of confirmed cases and is the only genuinely
irreplaceable data here. Writing them to one file would mean an HPO refresh
either destroys what was learned or requires a merge that has to get the
distinction right anyway. Keeping them apart makes the refresh trivial and
makes "what has this system actually learned?" a question with a direct
answer: read one small file.

That separation also survives into the loaded graph. `provenance` on a
learned edge records its origin, so a clinician or auditor reading a path can
see which links came from a published ontology and which the system inferred
and had confirmed -- a distinction that matters more, not less, once the
system has been running long enough for the learned layer to be substantial.

**Format: JSON Lines, one edge per line.** Chosen over a single JSON array or
a binary format because it appends without rewriting (promotion adds one
edge, not a rewritten file), it stays readable and diffable in review, and a
truncated write costs one line rather than the whole store. `weight`,
`lower`, and `upper` round-trip exactly, so an interval-valued edge survives
persistence with its epistemic state intact rather than collapsing back to a
point estimate.
"""

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .concept_paths import ConceptEdge, InMemoryConceptGraph

PROVENANCE_LEARNED_PREFIX = "learned:"

DEFAULT_LEARNED_FILENAME = "learned_edges.jsonl"


def edge_to_record(edge: ConceptEdge) -> dict[str, Any]:
    """One edge as a JSON-serialisable record, lossless for every field.

    `lower` and `upper` are written even when None, rather than omitted: an
    edge with no bounds and an edge whose bounds happen to be absent must
    deserialise to the same thing they were, and a missing key would leave
    that to the reader's default rather than the writer's record.
    """
    return {
        "source": edge.source,
        "relation": edge.relation,
        "target": edge.target,
        "weight": edge.weight,
        "provenance": edge.provenance,
        "lower": edge.lower,
        "upper": edge.upper,
    }


def record_to_edge(record: dict[str, Any]) -> ConceptEdge:
    return ConceptEdge(
        source=record["source"],
        relation=record["relation"],
        target=record["target"],
        weight=float(record.get("weight", 1.0)),
        provenance=record.get("provenance"),
        lower=record.get("lower"),
        upper=record.get("upper"),
    )


@dataclass
class LoadReport:
    """What was loaded, and what could not be.

    A corrupt line is reported rather than raised on: a single bad line in a
    learned store should not make the whole system refuse to start, and
    should not vanish silently either. The caller decides which matters more
    for their situation.
    """

    edges: list[ConceptEdge] = field(default_factory=list)
    skipped_lines: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"edges": len(self.edges), "skipped_lines": len(self.skipped_lines)}


class LearnedEdgeStore:
    """Append-only persistence for edges the system learned, apart from imported ones.

    Append-only by design, not by omission. An edge promoted after three
    independent confirmations is a claim about accumulated evidence; silently
    rewriting or deleting one would erase the trail that justifies it. Where
    a learned edge needs superseding, the right operation is appending its
    replacement with its own provenance -- the same discipline the ontology
    import already applies by preserving published intervals rather than
    collapsing them.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(self, edge: ConceptEdge) -> None:
        """Add one learned edge. Creates the file and parent directory if absent."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(edge_to_record(edge), ensure_ascii=False) + "\n")

    def append_many(self, edges: Iterable[ConceptEdge]) -> int:
        """Add several edges in one open, returning how many were written."""
        edges = list(edges)
        if not edges:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for edge in edges:
                handle.write(json.dumps(edge_to_record(edge), ensure_ascii=False) + "\n")
        return len(edges)

    def load(self) -> LoadReport:
        """Read every learned edge. A missing file is empty, not an error.

        A store that has never been written to is the normal state of a new
        installation, not a fault -- raising there would make every caller
        write the same try/except around a first run.
        """
        report = LoadReport()
        if not self.path.exists():
            return report
        for line in self.path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                report.edges.append(record_to_edge(json.loads(stripped)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                report.skipped_lines.append(stripped[:200])
        return report

    def count(self) -> int:
        return len(self.load().edges)


def build_persistent_graph(
    imported_edges: Sequence[ConceptEdge], store: LearnedEdgeStore
) -> tuple[InMemoryConceptGraph, LoadReport]:
    """Combine the derivable imported layer with the irreplaceable learned one.

    Returns the load report alongside the graph rather than only the graph,
    so a caller can log or surface how much of the running graph is learned
    rather than imported -- a number worth watching, since a learned layer
    growing faster than confirmations would justify is a signal something is
    promoting too eagerly.

    Learned edges are appended after imported ones. `InMemoryConceptGraph`
    keeps both, so a learned edge duplicating an imported relation does not
    silently replace it -- the traversal sees both and their provenance
    stays distinguishable, which is the safer behaviour when the two
    disagree.
    """
    report = store.load()
    return InMemoryConceptGraph.from_edges([*imported_edges, *report.edges]), report


def learned_provenance(origin_case: str, confirmations: int) -> str:
    """Provenance string marking an edge as learned, with what justified it.

    Kept as a function rather than left to each call site, so every learned
    edge carries the same recognisable shape and `is_learned` below stays a
    reliable test rather than a guess about string formats.
    """
    return f"{PROVENANCE_LEARNED_PREFIX}{origin_case}:confirmations={confirmations}"


def is_learned(edge: ConceptEdge) -> bool:
    """Whether this edge came from the system's own confirmed inference."""
    return bool(edge.provenance) and str(edge.provenance).startswith(PROVENANCE_LEARNED_PREFIX)
