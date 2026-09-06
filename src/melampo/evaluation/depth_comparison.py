"""Compare depth 0 against depth 1 on the same cases, before committing to recursion.

Depth 0 is the loop without the sub-model: the root navigates, greps, slices
and searches, but never hands a fragment to a second model. Depth 1 adds that
single recursive step. The literature suggests depth 0 is often sufficient at a
fraction of the cost, and that suggestion is worth three days to test before
three weeks are spent on the recursion it might make unnecessary.

The comparison is paired — same case, both depths — because case difficulty
varies far more than the difference between depths, and an unpaired comparison
would be dominated by which cases landed where.

What is compared is not "which answer is better", which would need a judge, but
three quantities the trajectories already carry: coverage of the corpus, number
of distinct fragments surfaced, and cost in iterations and sub-model calls. If
depth 1 surfaces the same evidence at twice the cost, that is the answer.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..memory.context_environment import EnvironmentDocument
from ..reasoning.rlm_engine import Budget, RlmEngine, Trajectory


@dataclass(frozen=True)
class DepthCase:
    case_id: str
    documents: tuple[EnvironmentDocument, ...]
    question: str


@dataclass
class PairedDepthOutcome:
    case_id: str
    depth0: Trajectory
    depth1: Trajectory

    @property
    def coverage_delta(self) -> float:
        return float(self.depth1.coverage.get("coverage_ratio", 0.0)) - float(
            self.depth0.coverage.get("coverage_ratio", 0.0)
        )

    @property
    def evidence_delta(self) -> int:
        return len(self.depth1.evidence()) - len(self.depth0.evidence())

    @property
    def cost_ratio(self) -> float:
        cost0 = max(1, int(self.depth0.budget.get("iterations", 0)))
        cost1 = int(self.depth1.budget.get("iterations", 0)) + int(self.depth1.budget.get("sub_model_calls", 0))
        return cost1 / cost0

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "depth0_completed": self.depth0.completed,
            "depth1_completed": self.depth1.completed,
            "coverage_delta": round(self.coverage_delta, 4),
            "evidence_delta": self.evidence_delta,
            "cost_ratio": round(self.cost_ratio, 2),
        }


@dataclass
class DepthComparisonReport:
    outcomes: list[PairedDepthOutcome] = field(default_factory=list)

    @property
    def mean_coverage_delta(self) -> float:
        return sum(item.coverage_delta for item in self.outcomes) / len(self.outcomes) if self.outcomes else 0.0

    @property
    def mean_evidence_delta(self) -> float:
        return sum(item.evidence_delta for item in self.outcomes) / len(self.outcomes) if self.outcomes else 0.0

    @property
    def mean_cost_ratio(self) -> float:
        return sum(item.cost_ratio for item in self.outcomes) / len(self.outcomes) if self.outcomes else 0.0

    def verdict(self) -> str:
        """Whether the recursive step earned its cost, stated as a sentence."""
        if not self.outcomes:
            return "no cases compared"
        if self.mean_evidence_delta <= 0 and self.mean_coverage_delta <= 0.01:
            return "depth 1 surfaces no more evidence than depth 0; the recursive step is not justified on these cases"
        if self.mean_cost_ratio > 2.0 and self.mean_evidence_delta < 1.0:
            return "depth 1 costs more than twice as much for under one extra fragment per case; marginal"
        return "depth 1 surfaces additional evidence; the recursive step earns its cost on these cases"

    def as_dict(self) -> dict[str, Any]:
        return {
            "cases": len(self.outcomes),
            "mean_coverage_delta": round(self.mean_coverage_delta, 4),
            "mean_evidence_delta": round(self.mean_evidence_delta, 3),
            "mean_cost_ratio": round(self.mean_cost_ratio, 2),
            "verdict": self.verdict(),
            "outcomes": [item.as_dict() for item in self.outcomes],
        }


def compare_depths(
    cases: Sequence[DepthCase],
    root_model: Callable[[str], str],
    sub_model: Callable[[str, str], str],
    *,
    budget_factory: Callable[[], Budget] = Budget,
) -> DepthComparisonReport:
    """Run every case at both depths with the same models and budget."""
    engine0 = RlmEngine(root_model=root_model, sub_model=None, depth=0)
    engine1 = RlmEngine(root_model=root_model, sub_model=sub_model, depth=1)
    report = DepthComparisonReport()
    for case in cases:
        trajectory0 = engine0.run(case.case_id, case.documents, case.question, budget=budget_factory())
        trajectory1 = engine1.run(case.case_id, case.documents, case.question, budget=budget_factory())
        report.outcomes.append(PairedDepthOutcome(case.case_id, trajectory0, trajectory1))
    return report
