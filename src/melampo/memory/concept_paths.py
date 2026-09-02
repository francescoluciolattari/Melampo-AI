"""Bounded traversal over the clinical concept graph.

A system that can only repeat what a case document already states is of little
use. Most clinical inference connects a finding to a condition through knowledge
that lives outside the case: bibasilar opacities relate to cardiac failure
through pulmonary oedema, and no single report needs to say so for the relation
to hold.

What matters is therefore not whether the system connects two things, but where
the connection comes from. Three origins, and only the third is a defect:

1. a case fragment asserts the relation — a documented fact;
2. no fragment asserts it, but a path through the concept graph supports it — an
   inference, admissible as a hypothesis with the path as its provenance;
3. neither the case nor the graph supports it — fabrication.

This module supplies the second. A path is either present in the graph or it is
not, which makes the judgement mechanical rather than a question of how
plausible a sentence sounds. Speculation then becomes a gradient rather than a
category: a short path over well-attested edges is a strong inference, a long
path over weak edges is speculative but still traceable, and the absence of any
path is not speculation at all.

Adapter-neutral. ``InMemoryConceptGraph`` keeps the logic testable offline;
production wiring goes through the semantic memory adapter's graph traversal.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ConceptEdge:
    """A typed, weighted relation between two clinical concepts.

    ``weight`` expresses how well attested the relation is, from an established
    mechanism down to a sparsely reported association. It is not a probability
    and must not be read as one.
    """

    source: str
    relation: str
    target: str
    weight: float = 1.0
    provenance: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "weight": round(self.weight, 3),
            "provenance": self.provenance,
        }


@runtime_checkable
class ConceptGraphView(Protocol):
    """Minimal traversal surface required by the grounding judge."""

    def edges_from(self, concept: str) -> Sequence[ConceptEdge]: ...

    def concepts(self) -> set[str]: ...


@dataclass
class InMemoryConceptGraph:
    """Offline graph view. Traversal is undirected: a relation is navigable both ways."""

    edges: list[ConceptEdge] = field(default_factory=list)

    @classmethod
    def from_edges(cls, edges: Iterable[ConceptEdge]) -> "InMemoryConceptGraph":
        return cls(edges=list(edges))

    def edges_from(self, concept: str) -> Sequence[ConceptEdge]:
        key = normalise_concept(concept)
        outgoing = [edge for edge in self.edges if normalise_concept(edge.source) == key]
        incoming = [
            ConceptEdge(
                source=edge.target,
                relation=f"inverse_{edge.relation}",
                target=edge.source,
                weight=edge.weight,
                provenance=edge.provenance,
            )
            for edge in self.edges
            if normalise_concept(edge.target) == key
        ]
        return outgoing + incoming

    def concepts(self) -> set[str]:
        found: set[str] = set()
        for edge in self.edges:
            found.add(normalise_concept(edge.source))
            found.add(normalise_concept(edge.target))
        return found


@dataclass(frozen=True)
class ConceptPath:
    """A traversal connecting two concepts, usable as provenance for an inference."""

    edges: tuple[ConceptEdge, ...]

    @property
    def hops(self) -> int:
        return len(self.edges)

    @property
    def strength(self) -> float:
        """Product of edge weights. Long chains of weak links score low, as they should."""
        total = 1.0
        for edge in self.edges:
            total *= max(0.0, min(1.0, edge.weight))
        return total

    @property
    def intermediates(self) -> tuple[str, ...]:
        return tuple(edge.target for edge in self.edges[:-1])

    def as_provenance(self) -> dict[str, Any]:
        return {
            "kind": "concept_graph_path",
            "hops": self.hops,
            "strength": round(self.strength, 3),
            "intermediates": list(self.intermediates),
            "edges": [edge.as_dict() for edge in self.edges],
        }

    def describe(self) -> str:
        if not self.edges:
            return ""
        parts = [self.edges[0].source]
        for edge in self.edges:
            parts.append(f"-[{edge.relation}]->")
            parts.append(edge.target)
        return " ".join(parts)


def find_paths(
    graph: ConceptGraphView,
    start: str,
    end: str,
    *,
    max_hops: int = 3,
    min_edge_weight: float = 0.0,
    max_paths: int = 8,
) -> list[ConceptPath]:
    """Breadth-first search for paths between two concepts, shortest first.

    Bounded deliberately. Given enough hops almost any two clinical concepts
    connect, and a path that long carries no information: it would turn this
    into a check that never fails, which is the same as having no check.
    """
    start_key, end_key = normalise_concept(start), normalise_concept(end)
    if not start_key or not end_key or start_key == end_key:
        return []

    found: list[ConceptPath] = []
    frontier: list[tuple[str, tuple[ConceptEdge, ...], frozenset[str]]] = [(start_key, (), frozenset({start_key}))]

    for _ in range(max(1, max_hops)):
        next_frontier: list[tuple[str, tuple[ConceptEdge, ...], frozenset[str]]] = []
        for concept, path, visited in frontier:
            for edge in graph.edges_from(concept):
                if edge.weight < min_edge_weight:
                    continue
                target_key = normalise_concept(edge.target)
                if target_key in visited:
                    continue
                extended = path + (edge,)
                if target_key == end_key:
                    found.append(ConceptPath(edges=extended))
                    if len(found) >= max_paths:
                        return _ranked(found)
                    continue
                next_frontier.append((target_key, extended, visited | {target_key}))
        frontier = next_frontier
        if not frontier:
            break

    return _ranked(found)


def shared_mechanisms(
    graph: ConceptGraphView,
    first: str,
    second: str,
    *,
    min_edge_weight: float = 0.0,
) -> list[str]:
    """Concepts sitting between two others on a two-hop path.

    The cascade in its simplest form: the causes of one finding intersected with
    the consequences of another. The intersection is not a coincidence of
    vocabulary, it is a candidate mechanism, and naming it converts an
    unsupported assertion of causation into a claim about a specific pathway
    that can be examined and rejected.
    """
    paths = find_paths(graph, first, second, max_hops=2, min_edge_weight=min_edge_weight, max_paths=32)
    mechanisms: list[str] = []
    for path in paths:
        if path.hops != 2:
            continue
        for concept in path.intermediates:
            if concept not in mechanisms:
                mechanisms.append(concept)
    return mechanisms


def mentioned_concepts(text: str, graph: ConceptGraphView, *, max_results: int = 6) -> list[str]:
    """Graph concepts appearing in a span of text, longest match first.

    Clinical concepts are usually multi-word — "bibasilar opacities", not
    "opacities" — so matching single tokens against graph nodes finds nothing.
    Longest-first ordering keeps the specific concept when a shorter one is
    contained inside it.
    """
    haystack = f" {_strip_punctuation(text)} "
    if not haystack.strip():
        return []
    candidates = sorted(graph.concepts(), key=len, reverse=True)
    found: list[str] = []
    for concept in candidates:
        if not concept:
            continue
        needle = _strip_punctuation(concept)
        if f" {needle} " in haystack and not any(needle in existing for existing in found):
            found.append(concept)
        if len(found) >= max_results:
            break
    return found


def _ranked(paths: list[ConceptPath]) -> list[ConceptPath]:
    return sorted(paths, key=lambda path: (path.hops, -path.strength))


def normalise_concept(value: str) -> str:
    return " ".join(str(value).lower().split())


def _strip_punctuation(value: str) -> str:
    """Lowercase and reduce every non-alphanumeric character to a separator.

    Concept names carry no punctuation, but the prose they are matched against
    does: a trailing full stop is enough to make a match silently fail.
    """
    return " ".join(
        "".join(character if character.isalnum() else " " for character in str(value).lower()).split()
    )
