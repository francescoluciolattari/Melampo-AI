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

STATE_DOCUMENTED = "documented"
STATE_UNCERTAIN_POSITIVE = "uncertain_positive"
STATE_WEAK_NEGATION = "weak_negation"
STATE_DOCUMENTED_EXCLUSION = "documented_exclusion"
STATE_GAP = "gap"

GAP_WIDTH = 0.8
NARROW_WIDTH = 0.2
LOW_CEILING = 0.5


def epistemic_state(lower: float, upper: float) -> str:
    """Name the epistemic condition an interval expresses.

    Descriptive only: nothing in the traversal branches on this label. The
    computation reads the bounds, because bounds compose along a path and
    category names do not.
    """
    width = upper - lower
    if width >= GAP_WIDTH:
        return STATE_GAP
    if width <= NARROW_WIDTH:
        return STATE_DOCUMENTED_EXCLUSION if upper <= LOW_CEILING else STATE_DOCUMENTED
    return STATE_WEAK_NEGATION if upper <= LOW_CEILING else STATE_UNCERTAIN_POSITIVE


@dataclass(frozen=True)
class ConceptEdge:
    """A typed relation between two clinical concepts, held as an interval.

    A single number cannot separate "documented as rare" from "nobody has
    looked": both arrive as a small value, and after multiplication along a path
    both read as near-certain denial. The bounds keep them apart. Their width is
    the epistemic uncertainty — an unknown relation is ``[0.0, 1.0]``, a
    documented exclusion is ``[0.0, 0.05]``, and the two behave differently
    everywhere downstream.

    ``weight`` remains the point estimate and the default reading. When bounds
    are not supplied the edge is treated as exact, so existing graphs keep their
    previous behaviour.
    """

    source: str
    relation: str
    target: str
    weight: float = 1.0
    provenance: str | None = None
    lower: float | None = None
    upper: float | None = None

    @classmethod
    def unknown(cls, source: str, relation: str, target: str, provenance: str | None = None) -> "ConceptEdge":
        """An edge the knowledge base does not have. Maximally wide, not absent.

        Distinct from omitting the edge: an absent edge cannot be traversed at
        all, while an unknown one can be traversed and reported as unknown,
        which is what makes it a candidate for graph completion.
        """
        return cls(source, relation, target, weight=0.5, provenance=provenance, lower=0.0, upper=1.0)

    @property
    def bounds(self) -> tuple[float, float]:
        low = _clamp(self.weight if self.lower is None else self.lower)
        high = _clamp(self.weight if self.upper is None else self.upper)
        return (low, max(low, high))

    @property
    def width(self) -> float:
        low, high = self.bounds
        return high - low

    @property
    def is_gap(self) -> bool:
        return self.width >= GAP_WIDTH

    @property
    def state(self) -> str:
        low, high = self.bounds
        return epistemic_state(low, high)

    def as_dict(self) -> dict[str, Any]:
        low, high = self.bounds
        return {
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "weight": round(self.weight, 3),
            "lower": round(low, 3),
            "upper": round(high, 3),
            "state": self.state,
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
            total *= _clamp(edge.weight)
        return total

    @property
    def strength_bounds(self) -> tuple[float, float]:
        """Interval strength: what the path guarantees, and what it could reach.

        Three consumers read this differently. The diagnostic path reads the
        lower bound — what the evidence guarantees. Hypothesis enumeration reads
        the upper bound — what could be true. Graph maintenance reads the width —
        where the knowledge base does not know, and therefore where looking pays.
        """
        low, high = 1.0, 1.0
        for edge in self.edges:
            edge_low, edge_high = edge.bounds
            low *= edge_low
            high *= edge_high
        return (low, high)

    @property
    def strength_lower(self) -> float:
        return self.strength_bounds[0]

    @property
    def strength_upper(self) -> float:
        return self.strength_bounds[1]

    @property
    def epistemic_width(self) -> float:
        low, high = self.strength_bounds
        return high - low

    @property
    def gap_count(self) -> int:
        """Edges on this path the knowledge base does not have."""
        return sum(1 for edge in self.edges if edge.is_gap)

    @property
    def intermediates(self) -> tuple[str, ...]:
        return tuple(edge.target for edge in self.edges[:-1])

    def as_provenance(self) -> dict[str, Any]:
        low, high = self.strength_bounds
        return {
            "kind": "concept_graph_path",
            "hops": self.hops,
            "strength": round(self.strength, 3),
            "strength_lower": round(low, 3),
            "strength_upper": round(high, 3),
            "epistemic_width": round(high - low, 3),
            "gap_count": self.gap_count,
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
    max_gap_edges: int | None = None,
) -> list[ConceptPath]:
    """Breadth-first search for paths between two concepts, shortest first.

    Bounded deliberately. Given enough hops almost any two clinical concepts
    connect, and a path that long carries no information: it would turn this
    into a check that never fails, which is the same as having no check.

    ``max_gap_edges`` bounds how many unknown edges a path may traverse. One
    unverified link in a chain is an inference; two concatenated unknowns are
    not a weaker inference but a different kind of object, since the second
    unknown is conditioned on the first being true. Leave it unset to ignore the
    distinction, which is the behaviour for graphs of exact edges.
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
                if max_gap_edges is not None and sum(1 for item in extended if item.is_gap) > max_gap_edges:
                    continue
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


@dataclass(frozen=True)
class DensityReport:
    """How well the knowledge base covers the neighbourhood of one case."""

    density: float
    known_edges: int
    gap_edges: int
    concepts_present: tuple[str, ...]
    concepts_absent: tuple[str, ...]

    @property
    def is_sparse_for(self) -> bool:
        return self.density < 0.5

    def as_dict(self) -> dict[str, Any]:
        return {
            "density": round(self.density, 3),
            "known_edges": self.known_edges,
            "gap_edges": self.gap_edges,
            "concepts_present": list(self.concepts_present),
            "concepts_absent": list(self.concepts_absent),
        }


def local_density(graph: ConceptGraphView, concepts: Sequence[str], *, radius: int = 1) -> DensityReport:
    """Fraction of the neighbourhood around these concepts that the graph knows.

    Completeness is the wrong question — a clinical graph never reaches it, so a
    completeness threshold is one that never arrives, and waiting on it
    deadlocks: the completion queue is fed by using the graph. Local density is
    answerable and varies enormously, since a cardiology case may sit in
    well-mapped territory while a rare presentation sits in a desert.
    """
    present: list[str] = []
    absent: list[str] = []
    frontier: set[str] = set()

    for concept in concepts:
        key = normalise_concept(concept)
        if not key:
            continue
        if graph.edges_from(key):
            present.append(key)
            frontier.add(key)
        else:
            absent.append(key)

    visited: set[str] = set()
    known = 0
    gaps = 0
    for _ in range(max(1, radius)):
        next_frontier: set[str] = set()
        for concept in frontier:
            if concept in visited:
                continue
            visited.add(concept)
            for edge in graph.edges_from(concept):
                if edge.is_gap:
                    gaps += 1
                else:
                    known += 1
                next_frontier.add(normalise_concept(edge.target))
        frontier = next_frontier

    total = known + gaps
    density = known / total if total else 0.0
    return DensityReport(
        density=density,
        known_edges=known,
        gap_edges=gaps,
        concepts_present=tuple(sorted(present)),
        concepts_absent=tuple(sorted(absent)),
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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
