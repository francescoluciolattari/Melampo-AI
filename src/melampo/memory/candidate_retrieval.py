"""Find which conditions the graph connects to a case's findings.

The missing half of `MechanismEnumerator`. Its `enumerate(findings,
candidate_conditions)` requires the caller to supply the conditions to weigh
-- it answers "of these conditions, which does the graph connect to these
findings, and how strongly", not "which conditions could explain these
findings at all". Identified directly in discussion: without this, wiring
the enumerator into the pipeline would connect a machine with nothing to
chew on.

The graph already holds what is needed. HPO annotations import as
`disease -> has_phenotype -> finding` edges, and `InMemoryConceptGraph`
generates the inverse of every edge, so walking outward from a finding
reaches the diseases that manifest it. This module does that walk and
returns the conditions worth enumerating over.

**Ranked by how many of the case's findings each condition touches, before
anything else.** A condition linked to four of the presented findings is a
better candidate than one linked to a single finding by a strong edge, and
ranking by edge strength alone would invert that -- the same reasoning the
convergence reward in `information_content` already encodes at path level:
independent corroboration outweighs a single strong link.

**What this deliberately does not do.** It does not decide which conditions
are plausible, score them, or prune to a differential. It gathers what the
graph connects, and `MechanismEnumerator` -- which already has the path
enumeration, the interval arithmetic, and the density-based judgement about
whether to rank at all -- does the rest. Splitting it this way keeps the
retrieval cheap and total: over-gathering here costs the enumerator a little
work and risks nothing, while under-gathering silently removes a diagnosis
from consideration before anything can weigh it.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .concept_paths import ConceptGraphView, normalise_concept, resolve_concept

# A condition reachable from a finding in more hops than this is not being
# "connected to the case" in a useful sense -- at three or four hops nearly
# everything connects to nearly everything, which is the same reason
# find_paths bounds its own traversal.
DEFAULT_MAX_HOPS = 2

# Cap on how many conditions to hand the enumerator. Not a quality judgement
# -- the enumerator does that -- but a bound on work, and a signal when the
# graph is so densely connected around these findings that the retrieval is
# not discriminating. `truncated` on the result says when it bit.
DEFAULT_MAX_CANDIDATES = 40


@dataclass(frozen=True)
class CandidateCondition:
    """One condition the graph connects to the case, with how broadly."""

    condition: str
    findings_linked: tuple[str, ...]
    nearest_hops: int

    @property
    def breadth(self) -> int:
        """How many of the case's findings this condition touches."""
        return len(self.findings_linked)

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "findings_linked": list(self.findings_linked),
            "breadth": self.breadth,
            "nearest_hops": self.nearest_hops,
        }


@dataclass
class RetrievalReport:
    """The candidates found, and what the retrieval could not do."""

    candidates: list[CandidateCondition] = field(default_factory=list)
    unresolved_findings: list[str] = field(default_factory=list)
    truncated: bool = False

    @property
    def condition_names(self) -> list[str]:
        """Just the names, in rank order -- what MechanismEnumerator wants."""
        return [item.condition for item in self.candidates]

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidates": [item.as_dict() for item in self.candidates],
            "unresolved_findings": list(self.unresolved_findings),
            "truncated": self.truncated,
        }


def retrieve_candidates(
    findings: Sequence[str],
    graph: ConceptGraphView,
    *,
    max_hops: int = DEFAULT_MAX_HOPS,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    exclude: Sequence[str] = (),
) -> RetrievalReport:
    """Gather every condition the graph links to any of these findings.

    Findings are resolved through `resolve_concept` first, for the same
    reason `verify_mechanism` does: a finding phrased as a clinician would
    write it ("the patient's chronic kidney disease") is not a graph node,
    and passing it straight to `edges_from` returns nothing silently. A
    finding that resolves to nothing is reported in `unresolved_findings`
    rather than dropped, since "the graph has never heard of this finding"
    is a fact worth surfacing -- it is exactly the coverage gap the density
    check downstream reasons about.
    """
    report = RetrievalReport()
    excluded = {normalise_concept(item) for item in exclude}

    # condition -> (findings that reached it, fewest hops taken to reach it)
    reached: dict[str, tuple[set[str], int]] = {}

    for raw_finding in findings:
        finding = resolve_concept(raw_finding, graph)
        if finding is None:
            report.unresolved_findings.append(raw_finding)
            continue

        # Breadth-first from this finding, bounded, collecting everything
        # reachable. What counts as a "condition" is not decided here by
        # inspecting relation names -- that would hard-code an assumption
        # about the ontology's vocabulary that a second imported source
        # (LOINC, ATC) would break. Instead, direction is used: HPO imports
        # as `disease -> has_phenotype -> finding`, and InMemoryConceptGraph
        # marks the generated reverse of each edge, so a concept reached by
        # traversing an edge *backwards* is on the disease side of that
        # relation. A first run of this retrieval, before this distinction,
        # ranked "night sweats" among the hypotheses -- a finding proposed
        # as a diagnosis, reached because it shares a disease with the
        # case's own findings. Two hops through a disease lands on that
        # disease's other symptoms, and those are not candidates.
        frontier: list[tuple[str, int]] = [(finding, 0)]
        seen = {finding}
        while frontier:
            concept, hops = frontier.pop(0)
            if hops >= max_hops:
                continue
            for edge in graph.edges_from(concept):
                target = normalise_concept(edge.target)
                if target in seen:
                    continue
                seen.add(target)
                reached_by_reverse = edge.relation.startswith("inverse_")
                if target not in excluded and reached_by_reverse:
                    linked, best_hops = reached.get(target, (set(), hops + 1))
                    linked.add(finding)
                    reached[target] = (linked, min(best_hops, hops + 1))
                frontier.append((target, hops + 1))

    resolved_findings = {
        item for item in (resolve_concept(raw, graph) for raw in findings) if item is not None
    }
    candidates = [
        CandidateCondition(condition=name, findings_linked=tuple(sorted(linked)), nearest_hops=hops)
        for name, (linked, hops) in reached.items()
        if name not in resolved_findings
    ]
    # Breadth first, then proximity, then name -- the last purely so the
    # order is stable across runs rather than dependent on dict iteration,
    # which matters for a bench that compares runs to each other.
    candidates.sort(key=lambda item: (-item.breadth, item.nearest_hops, item.condition))

    report.truncated = len(candidates) > max_candidates
    report.candidates = candidates[:max_candidates]
    return report
