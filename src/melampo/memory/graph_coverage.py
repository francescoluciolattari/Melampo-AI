"""Measure how much of a reference set of clinical relations the graph contains.

After knowledge-mediated grounding landed, the quality of every judgement began
to depend on graph coverage: on a thin graph a legitimate inference is reported
as fabrication, and the false alarms look like a defect of the model rather than
of the knowledge base. That risk is only manageable once it is a number.

Coverage is measured against a reference set — relations known to be clinically
real, supplied by whoever curates them — and the result separates three
outcomes, because they call for different work:

- **present** — the graph has a traversable, non-gap path;
- **gap** — a path exists but crosses an unknown edge, so the relation is
  reachable but unattested;
- **absent** — no path at all.

Absent relations are the completion queue. Gap relations are the calibration
queue. Treating them as one number would merge two different jobs.

The same measurement bounds what any evaluation can claim. An A/B run over a
region with low coverage measures the knowledge base while appearing to measure
the architecture.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .concept_paths import ConceptGraphView, find_paths

OUTCOME_PRESENT = "present"
OUTCOME_GAP = "gap"
OUTCOME_ABSENT = "absent"


@dataclass(frozen=True)
class ReferenceRelation:
    """A relation the graph is expected to support, with its source."""

    source: str
    target: str
    provenance: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target, "provenance": self.provenance}


@dataclass
class CoverageResult:
    """Per-relation outcome, with the shortest supporting path when there is one."""

    relation: ReferenceRelation
    outcome: str
    hops: int | None = None
    strength_lower: float = 0.0
    strength_upper: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.relation.as_dict(),
            "outcome": self.outcome,
            "hops": self.hops,
            "strength_lower": round(self.strength_lower, 3),
            "strength_upper": round(self.strength_upper, 3),
        }


@dataclass
class CoverageReport:
    results: list[CoverageResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    def _count(self, outcome: str) -> int:
        return sum(1 for item in self.results if item.outcome == outcome)

    @property
    def present(self) -> int:
        return self._count(OUTCOME_PRESENT)

    @property
    def gap(self) -> int:
        return self._count(OUTCOME_GAP)

    @property
    def absent(self) -> int:
        return self._count(OUTCOME_ABSENT)

    @property
    def coverage(self) -> float:
        """Fraction of reference relations the graph supports without crossing an unknown."""
        return self.present / self.total if self.total else 0.0

    @property
    def reachability(self) -> float:
        """Fraction reachable at all, counting paths that cross an unknown edge."""
        return (self.present + self.gap) / self.total if self.total else 0.0

    def completion_queue(self) -> list[ReferenceRelation]:
        """Relations with no path: the graph is missing them entirely."""
        return [item.relation for item in self.results if item.outcome == OUTCOME_ABSENT]

    def calibration_queue(self) -> list[ReferenceRelation]:
        """Relations reachable only through an unknown edge: present but unattested."""
        return [item.relation for item in self.results if item.outcome == OUTCOME_GAP]

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "present": self.present,
            "gap": self.gap,
            "absent": self.absent,
            "coverage": round(self.coverage, 4),
            "reachability": round(self.reachability, 4),
            "results": [item.as_dict() for item in self.results],
        }


def measure_coverage(
    graph: ConceptGraphView,
    reference: Sequence[ReferenceRelation],
    *,
    max_hops: int = 3,
) -> CoverageReport:
    """Classify each reference relation as present, reachable through a gap, or absent."""
    report = CoverageReport()
    for relation in reference:
        paths = find_paths(graph, relation.source, relation.target, max_hops=max_hops)
        if not paths:
            report.results.append(CoverageResult(relation=relation, outcome=OUTCOME_ABSENT))
            continue
        attested = [path for path in paths if path.gap_count == 0]
        best = attested[0] if attested else paths[0]
        report.results.append(
            CoverageResult(
                relation=relation,
                outcome=OUTCOME_PRESENT if attested else OUTCOME_GAP,
                hops=best.hops,
                strength_lower=best.strength_lower,
                strength_upper=best.strength_upper,
            )
        )
    return report


def evaluation_is_interpretable(report: CoverageReport, *, minimum_coverage: float = 0.6) -> dict[str, Any]:
    """Whether a measurement over this graph would reflect the architecture.

    Guards the case the decision record flags: below the threshold an A/B run
    measures graph coverage while appearing to measure retrieval, and a negative
    result would be predetermined rather than informative.
    """
    sufficient = report.coverage >= minimum_coverage
    return {
        "interpretable": sufficient,
        "coverage": round(report.coverage, 4),
        "minimum_coverage": minimum_coverage,
        "reason": (
            "graph coverage supports interpreting the result as a property of the architecture"
            if sufficient
            else "coverage too low: a result here would reflect the knowledge base, not the retrieval strategy"
        ),
    }
