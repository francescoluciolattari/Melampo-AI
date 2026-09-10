"""Constrained spreading activation over the concept graph.

The third step of the research report's sequence, and the one that had to
wait for the first two. Spreading activation asks a different question from
`find_paths`: not "is there a route between these two named concepts" but
"starting from these concepts, what else lights up, and how strongly" --
which is what a relevance question needs when the second concept is known
but the mechanism connecting them is not.

**Why constrained, and why that word is doing real work.** The literature is
explicit that classical spreading activation *uses all relations of a node,
which adds unsuitable information*: activation leaks through every edge type
indiscriminately and a large, weakly-related neighbourhood drowns the signal.
A 2018 evaluation of query-oriented constrained spreading against
unconstrained reported improvements of 18.9% and 43.8% in MAP over syntactic
and unconstrained semantic search. Constraint is not an optimisation here; it
is the difference between the technique working and not.

Three constraints are applied, each addressing a documented failure mode:

- **Relation constraint.** Only edge types admitted by the caller propagate.
  The project's own vocabulary supplies the defaults -- `causes`, `enables`,
  `manifests_as` from the illness script, `has_phenotype` from the HPO
  import. A relation absent from the allowed set stops activation dead
  rather than passing it along at reduced strength, because "this edge type
  is not relevant to this question" is a categorical statement, not a
  discount.
- **Decay constraint.** Activation multiplies by edge weight and by a decay
  factor per hop, so distance costs. `ConceptPath.strength` already does the
  first half; the explicit per-hop decay is what stops a long chain of strong
  edges from arriving at full force.
- **Threshold constraint.** Activation below a floor stops propagating. This
  is what bounds the frontier in practice: without it, a graph of 285,598
  arcs would eventually activate almost everything at negligible strength.

**Information Content is applied at the destination, not along the way.**
A concept's specificity says how much its activation *means*, not how well
activation travels through it -- weighting mid-path would conflate the two
and penalise a route for passing through a general concept even when the
general concept is genuinely the connection. So spreading uses edge weights
and decay, and IC weights the final activation of each reached concept.
This is the same separation `information_content.score_path` already makes
between "how strong is this path" and "how specific are the concepts on it".
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..reasoning.illness_script import SCRIPT_RELATIONS
from .concept_paths import ConceptGraphView, normalise_concept
from .information_content import InformationContentTable

# HPO's own relation, assigned by ontology_import to every imported annotation
# arc. Named here rather than imported to avoid a circular import between
# memory modules; a test asserts the two stay identical.
RELATION_HAS_PHENOTYPE = "has_phenotype"

# The project's existing relation vocabulary, not a new one: the illness
# script's enables/causes/manifests_as, plus HPO's has_phenotype. These are
# the edges that carry clinical implication, which is what a relevance
# question is asking about.
DEFAULT_ALLOWED_RELATIONS = frozenset(SCRIPT_RELATIONS | {RELATION_HAS_PHENOTYPE})

# Per-hop decay, separate from edge weight. Distance costs even along strong
# edges: two concepts three strong hops apart are related, but not as
# directly as two one hop apart, and the product of weights alone does not
# express that when every weight is near 1.0.
DEFAULT_DECAY = 0.7

# Activation below this stops propagating. The practical bound on the
# frontier: without it, a graph of 285,598 arcs eventually activates nearly
# everything at negligible strength, which is the "unsuitable information"
# the constrained-spreading literature describes.
DEFAULT_THRESHOLD = 0.05

# Hard ceiling on how far activation travels regardless of decay and
# threshold. Belt and braces: a pathological graph with weight-1.0 cycles
# could otherwise keep activation above threshold indefinitely.
DEFAULT_MAX_HOPS = 4


@dataclass(frozen=True)
class ActivatedConcept:
    """One concept reached by spreading, with how and how strongly."""

    concept: str
    activation: float
    hops: int
    reached_via: tuple[str, ...]
    information_content: float
    weighted_activation: float
    source_count: int = 1

    @property
    def is_multiply_sourced(self) -> bool:
        """Whether more than one origin concept activated this one.

        The property a relevance question cares most about: a concept lit up
        from both the factor and the target is a candidate connection between
        them, which is categorically more interesting than one lit up from
        either alone.
        """
        return self.source_count > 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "concept": self.concept,
            "activation": round(self.activation, 4),
            "weighted_activation": round(self.weighted_activation, 4),
            "information_content": round(self.information_content, 4),
            "hops": self.hops,
            "reached_via": list(self.reached_via),
            "source_count": self.source_count,
            "is_multiply_sourced": self.is_multiply_sourced,
        }


@dataclass
class SpreadingResult:
    """Everything spreading activation reached, and the constraints it ran under."""

    activated: list[ActivatedConcept] = field(default_factory=list)
    origins: tuple[str, ...] = ()
    allowed_relations: tuple[str, ...] = ()
    blocked_relation_count: int = 0
    frontier_stopped_at_threshold: int = 0

    def ranked(self) -> list[ActivatedConcept]:
        """Strongest first, by IC-weighted activation."""
        return sorted(self.activated, key=lambda item: -item.weighted_activation)

    def multiply_sourced(self) -> list[ActivatedConcept]:
        """Concepts reached from more than one origin, strongest first.

        For a relevance question this is the answer set: these are the
        concepts that could mediate between the factor and the target.
        """
        return [item for item in self.ranked() if item.is_multiply_sourced]

    def as_dict(self) -> dict[str, Any]:
        return {
            "origins": list(self.origins),
            "allowed_relations": list(self.allowed_relations),
            "activated_count": len(self.activated),
            "multiply_sourced_count": len(self.multiply_sourced()),
            "blocked_relation_count": self.blocked_relation_count,
            "frontier_stopped_at_threshold": self.frontier_stopped_at_threshold,
            "activated": [item.as_dict() for item in self.ranked()],
        }


def spread(
    graph: ConceptGraphView,
    origins: Sequence[str],
    *,
    table: InformationContentTable | None = None,
    allowed_relations: Iterable[str] | None = None,
    decay: float = DEFAULT_DECAY,
    threshold: float = DEFAULT_THRESHOLD,
    max_hops: int = DEFAULT_MAX_HOPS,
    initial_activation: Mapping[str, float] | None = None,
) -> SpreadingResult:
    """Spread activation outward from origin concepts under three constraints.

    Origins themselves are not reported as activated: a concept is not
    evidence of its own relevance, and including them would put the two
    concepts a relevance question already names at the top of its own answer.

    ``blocked_relation_count`` and ``frontier_stopped_at_threshold`` are
    returned rather than discarded so the constraints are visible in the
    result. A spread that blocked nothing was effectively unconstrained --
    the documented failure mode -- and a reader should be able to see that
    from the output instead of inferring it.
    """
    table = table or InformationContentTable()
    allowed = frozenset(allowed_relations) if allowed_relations is not None else DEFAULT_ALLOWED_RELATIONS
    origin_keys = [normalise_concept(item) for item in origins]
    origin_set = set(origin_keys)

    # concept -> (best activation, hops at best, path to it, origins reaching it)
    best: dict[str, tuple[float, int, tuple[str, ...], set[str]]] = {}
    blocked = 0
    stopped = 0

    for origin in origin_keys:
        start_activation = float((initial_activation or {}).get(origin, 1.0))
        frontier: list[tuple[str, float, int, tuple[str, ...]]] = [(origin, start_activation, 0, ())]
        seen_this_origin: set[str] = {origin}

        while frontier:
            concept, activation, hops, via = frontier.pop(0)
            if hops >= max_hops:
                continue

            for edge in graph.edges_from(concept):
                relation = edge.relation
                # Inverse edges are the same relation traversed backwards; the
                # constraint is about relation *type*, not direction, so the
                # prefix is stripped before checking rather than causing every
                # backward traversal to be blocked as an unknown relation.
                base_relation = relation.removeprefix("inverse_")
                if base_relation not in allowed:
                    blocked += 1
                    continue

                target = normalise_concept(edge.target)
                if target in seen_this_origin:
                    continue

                next_activation = activation * edge.weight * decay
                if next_activation < threshold:
                    stopped += 1
                    continue

                seen_this_origin.add(target)
                next_via = (*via, concept)

                if target not in origin_set:
                    current = best.get(target)
                    if current is None:
                        best[target] = (next_activation, hops + 1, next_via, {origin})
                    else:
                        current_activation, current_hops, current_via, sources = current
                        sources.add(origin)
                        if next_activation > current_activation:
                            best[target] = (next_activation, hops + 1, next_via, sources)
                        else:
                            best[target] = (current_activation, current_hops, current_via, sources)

                frontier.append((target, next_activation, hops + 1, next_via))

    activated = []
    for concept, (activation, hops, via, sources) in best.items():
        ic = table.value(concept)
        activated.append(
            ActivatedConcept(
                concept=concept,
                activation=activation,
                hops=hops,
                reached_via=via,
                information_content=ic,
                weighted_activation=activation * ic,
                source_count=len(sources),
            )
        )

    return SpreadingResult(
        activated=activated,
        origins=tuple(origin_keys),
        allowed_relations=tuple(sorted(allowed)),
        blocked_relation_count=blocked,
        frontier_stopped_at_threshold=stopped,
    )


def mediating_concepts(
    graph: ConceptGraphView,
    factor: str,
    target: str,
    *,
    table: InformationContentTable | None = None,
    **kwargs: Any,
) -> SpreadingResult:
    """Spread from both ends of a relevance question and keep what both reach.

    This is what "does the family history bear on today's aortic measurement"
    needs and `find_paths` alone does not give: not whether a route exists,
    but *what mediates* -- the concepts lit up from both the factor and the
    target, which are the candidate mechanisms.

    Returns the full result rather than only the intersection, because a
    spread that reached nothing from one side is a different finding from one
    that reached plenty on both sides with no overlap, and collapsing them
    would hide which happened.
    """
    return spread(graph, [factor, target], table=table, **kwargs)
