"""Concept specificity as Information Content, and path scoring weighted by it.

An empirical observation earlier in this project: comparing two answers by
character similarity ranked a contradiction above a paraphrase, because
"pulmonary embolism" and "pulmonary oedema" share the word "pulmonary" while
"40 mg daily" and "prednisone 40 mg daily" are penalised for differing in
length. The informal diagnosis at the time was that "pulmonary" discriminates
nothing in a medical vocabulary while "embolism" versus "oedema" discriminates
everything.

That intuition has a formal name. **Information Content** is a concept's
specificity, `-log(p(concept))`: a concept appearing everywhere carries little
information, one appearing rarely carries much. And the comparison against
the alternative is settled empirically rather than by argument — evaluating
semantic similarity measures on the MSH-WSD biomedical disambiguation
dataset, information-content-based measures achieve higher accuracy than
path-based measures including shortest-path and Leacock-Chodorow.

`find_paths` is a path-based measure and inherits that limitation. This
module does not replace it; it weights it. A path through highly specific
concepts is stronger evidence of a real connection than a path of the same
length through general ones, and until now nothing in the scoring said so.

**Two ways to obtain IC, and why both are here.** The classical formulation
derives `p(concept)` from corpus frequency. This project has no annotated
corpus of its own, but it does have an imported ontology with published
frequency ranges preserved as intervals, and an ontology's own structure is
itself a usable proxy: a concept with many descendants is general, one with
none is maximally specific (the "intrinsic IC" approach). `from_frequencies`
takes real frequencies where they exist; `from_graph_structure` derives IC
from connectivity where they do not. Neither is guessed at silently — the
`basis` field on every score records which was used.

**A limitation worth stating plainly.** The MSH-WSD result is on UMLS and
MeSH, for word sense disambiguation, not on HPO for answer comparison. The
direction of the finding is well supported; the magnitude on this project's
graph and task is not established, and `IcWeightedPathScore` deliberately
carries both the unweighted and the weighted strength so the difference the
weighting makes is visible rather than assumed.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .concept_paths import ConceptGraphView, ConceptPath, normalise_concept

BASIS_FREQUENCY = "corpus_frequency"
BASIS_GRAPH_STRUCTURE = "graph_structure"
BASIS_UNKNOWN = "unknown"

# A concept absent from the IC table gets this, the midpoint of the normalised
# range. Neither 0.0 (which would make an unknown concept free to traverse,
# rewarding ignorance) nor 1.0 (which would make it maximally informative,
# rewarding it for the same reason). The midpoint is the honest position for
# "no information about how informative this is", and `basis` records that it
# was a default rather than a measurement.
DEFAULT_IC = 0.5


@dataclass(frozen=True)
class ConceptInformationContent:
    """One concept's specificity, with the basis it was derived from."""

    concept: str
    value: float
    basis: str = BASIS_UNKNOWN

    def as_dict(self) -> dict[str, Any]:
        return {"concept": self.concept, "value": round(self.value, 4), "basis": self.basis}


@dataclass
class InformationContentTable:
    """Information Content per concept, normalised to [0, 1].

    Normalisation is to the observed maximum rather than to a theoretical one:
    raw `-log(p)` has no fixed upper bound, and a path score multiplying
    unbounded factors would be neither comparable across graphs nor readable.
    The relative ordering — which is what the weighting actually uses — is
    unaffected.
    """

    scores: dict[str, ConceptInformationContent] = field(default_factory=dict)

    def get(self, concept: str) -> ConceptInformationContent:
        key = normalise_concept(concept)
        return self.scores.get(key, ConceptInformationContent(concept=key, value=DEFAULT_IC, basis=BASIS_UNKNOWN))

    def value(self, concept: str) -> float:
        return self.get(concept).value

    def __len__(self) -> int:
        return len(self.scores)

    @classmethod
    def from_frequencies(cls, frequencies: Mapping[str, float]) -> "InformationContentTable":
        """Classical IC: -log(p), from observed concept frequencies.

        A frequency of zero would give infinite IC, which is the wrong answer
        for "never observed" -- that is absence of evidence, not maximal
        specificity. Zero and negative frequencies are skipped rather than
        clamped, so they fall through to DEFAULT_IC with basis `unknown`,
        which is what they honestly are.
        """
        positive = {
            normalise_concept(concept): float(count)
            for concept, count in frequencies.items()
            if float(count) > 0
        }
        if not positive:
            return cls()

        total = sum(positive.values())
        raw = {concept: -math.log(count / total) for concept, count in positive.items()}
        return cls(scores=_normalised(raw, BASIS_FREQUENCY))

    @classmethod
    def from_graph_structure(cls, graph: ConceptGraphView) -> "InformationContentTable":
        """Intrinsic IC: specificity from the ontology's own shape.

        A concept reachable from many others is general; one with few
        connections is specific. This is the standard fallback when no corpus
        frequency exists, and it is what this project can compute today from
        an imported ontology without needing an annotated corpus first.

        Deliberately simple -- degree-based rather than descendant-count-based
        -- because the traversal here is undirected and a true descendant count
        would need directed hierarchy edges the graph view does not expose.
        The ordering it produces (hub concepts general, leaf concepts specific)
        is the property the weighting depends on.
        """
        concepts = graph.concepts()
        if not concepts:
            return cls()

        degrees = {normalise_concept(concept): len(graph.edges_from(concept)) for concept in concepts}
        total_degree = sum(degrees.values())
        if total_degree <= 0:
            return cls()

        raw: dict[str, float] = {}
        for concept, degree in degrees.items():
            # A concept with no edges is maximally specific by this measure,
            # but it is also unreachable, so its IC never affects a path score.
            share = (degree + 1) / (total_degree + len(degrees))
            raw[concept] = -math.log(share)
        return cls(scores=_normalised(raw, BASIS_GRAPH_STRUCTURE))


def _normalised(raw: Mapping[str, float], basis: str) -> dict[str, ConceptInformationContent]:
    highest = max(raw.values()) if raw else 0.0
    if highest <= 0:
        return {
            concept: ConceptInformationContent(concept=concept, value=DEFAULT_IC, basis=basis) for concept in raw
        }
    return {
        concept: ConceptInformationContent(concept=concept, value=value / highest, basis=basis)
        for concept, value in raw.items()
    }


@dataclass(frozen=True)
class IcWeightedPathScore:
    """A path's strength, before and after Information Content weighting.

    Both are carried deliberately. The MSH-WSD result supporting IC weighting
    is on UMLS/MeSH for word sense disambiguation, not on this graph for this
    task -- so the difference the weighting makes should be visible in every
    score rather than folded invisibly into one number.
    """

    path_strength: float
    mean_information_content: float
    weighted_strength: float
    concepts: tuple[str, ...]
    bases: tuple[str, ...]

    @property
    def all_concepts_measured(self) -> bool:
        """Whether every concept on the path had a real IC, not the default.

        A weighted score resting mostly on DEFAULT_IC is not measuring
        specificity, it is reporting an assumption, and a reader should be
        able to tell which they have.
        """
        return BASIS_UNKNOWN not in self.bases

    def as_dict(self) -> dict[str, Any]:
        return {
            "path_strength": round(self.path_strength, 4),
            "mean_information_content": round(self.mean_information_content, 4),
            "weighted_strength": round(self.weighted_strength, 4),
            "all_concepts_measured": self.all_concepts_measured,
            "concepts": list(self.concepts),
        }


def path_concepts(path: ConceptPath) -> tuple[str, ...]:
    """Every concept a path passes through, in order, deduplicated."""
    if not path.edges:
        return ()
    seen: list[str] = [normalise_concept(path.edges[0].source)]
    for edge in path.edges:
        target = normalise_concept(edge.target)
        if target != seen[-1]:
            seen.append(target)
    return tuple(seen)


def score_path(path: ConceptPath, table: InformationContentTable) -> IcWeightedPathScore:
    """Weight a path's strength by the mean specificity of the concepts it crosses.

    The mean, not the product: a product would compound with path length and
    conflate "crosses general concepts" with "is long", which `strength`
    already penalises. Keeping the two effects separate means a reader can see
    whether a low score comes from weak links or from unspecific ones.
    """
    concepts = path_concepts(path)
    if not concepts:
        return IcWeightedPathScore(
            path_strength=path.strength, mean_information_content=DEFAULT_IC,
            weighted_strength=path.strength * DEFAULT_IC, concepts=(), bases=(),
        )

    entries = [table.get(concept) for concept in concepts]
    mean_ic = sum(entry.value for entry in entries) / len(entries)
    return IcWeightedPathScore(
        path_strength=path.strength,
        mean_information_content=mean_ic,
        weighted_strength=path.strength * mean_ic,
        concepts=concepts,
        bases=tuple(entry.basis for entry in entries),
    )


def rank_paths(paths: Iterable[ConceptPath], table: InformationContentTable) -> list[IcWeightedPathScore]:
    """Score and order paths by IC-weighted strength, strongest first."""
    scored = [score_path(path, table) for path in paths]
    return sorted(scored, key=lambda item: -item.weighted_strength)


# --------------------------------------------------------------------------
# Converging paths
# --------------------------------------------------------------------------

# ONTOSPREAD, the spreading-activation framework applied to medical
# ontologies, explicitly rewards concepts reached by multiple independent
# paths above what any single path would score. It is the graph-level analogue
# of this project's own cross-check principle: two independent routes to the
# same conclusion are worth more than one, and worth more than their
# individual strengths suggest.
#
# The reward is deliberately sub-additive -- each additional independent path
# adds less than the last. Two routes are much better than one; the tenth adds
# little, and a formulation where it added as much as the first would let a
# densely-connected region of the graph manufacture confidence by sheer
# connectivity, which is the failure mode a convergence reward most needs to
# avoid.
CONVERGENCE_REWARD_BASE = 0.5


@dataclass(frozen=True)
class ConvergenceScore:
    """How many genuinely distinct routes connect two concepts, and how strongly."""

    best_weighted_strength: float
    independent_path_count: int
    convergence_multiplier: float
    converged_strength: float
    shared_intermediate_concepts: tuple[str, ...]

    @property
    def is_single_thread(self) -> bool:
        """Whether the connection rests on one route only.

        The distinction a relevance question actually needs: "these concepts
        are connected" reads very differently when one thin path supports it
        than when four independent ones do.
        """
        return self.independent_path_count <= 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_weighted_strength": round(self.best_weighted_strength, 4),
            "independent_path_count": self.independent_path_count,
            "convergence_multiplier": round(self.convergence_multiplier, 4),
            "converged_strength": round(self.converged_strength, 4),
            "is_single_thread": self.is_single_thread,
            "shared_intermediate_concepts": list(self.shared_intermediate_concepts),
        }


def score_convergence(paths: Sequence[ConceptPath], table: InformationContentTable) -> ConvergenceScore:
    """Reward two concepts being connected by several independent routes.

    "Independent" means not passing through the same intermediate concepts:
    two paths differing only in the direction an edge was traversed, or
    sharing every waypoint, are one route reported twice and must not count
    as corroboration. Intermediates shared across *some* paths are reported
    separately -- a common waypoint is a real feature of the connection
    (often the mechanism itself), not a defect, but a reader should see when
    every route funnels through one node.
    """
    if not paths:
        return ConvergenceScore(0.0, 0, 1.0, 0.0, ())

    scored = rank_paths(paths, table)
    best = scored[0].weighted_strength

    signatures: set[frozenset[str]] = set()
    intermediate_sets: list[set[str]] = []
    for score in scored:
        intermediates = frozenset(score.concepts[1:-1]) if len(score.concepts) > 2 else frozenset()
        signatures.add(intermediates)
        intermediate_sets.append(set(intermediates))

    independent = len(signatures)
    # Sub-additive: 1 path -> 1.0, 2 -> 1.5, 3 -> 1.75, 4 -> 1.875 ...
    multiplier = 1.0 + sum(CONVERGENCE_REWARD_BASE**index for index in range(1, independent))

    shared = set.intersection(*intermediate_sets) if intermediate_sets and all(intermediate_sets) else set()

    return ConvergenceScore(
        best_weighted_strength=best,
        independent_path_count=independent,
        convergence_multiplier=multiplier,
        # Capped at 1.0: a strength is a probability-like quantity and a
        # convergence reward must not push it past certainty, however many
        # routes agree.
        converged_strength=min(1.0, best * multiplier),
        shared_intermediate_concepts=tuple(sorted(shared)),
    )
