"""Paired A/B comparison of one-shot and dual-path retrieval.

Evaluates the refutation criterion registered for
``rlm.dual_path_beats_single_path``:

    Dual-path faithfulness falls below the one-shot baseline on the same case
    set, or its recall gain is not distinguishable from the one-shot path.

Two design choices worth stating.

**Paired, not independent.** Both strategies run on the same cases, so the
comparison is within-case. Case difficulty varies far more than the difference
between strategies, and an unpaired comparison would be dominated by which cases
landed in which arm.

**Distinguishability is decided by a confidence interval, not a mean.** A recall
gain of +0.03 with an interval spanning zero is not a gain, and reporting the
mean alone would let it pass. The interval is produced by a deterministic
bootstrap with a fixed seed, so the same inputs always yield the same verdict —
an evaluation that changes its answer between runs cannot gate a release.

The harness can only refute or corroborate. It does not decide whether to adopt
the architecture, and a corroborated claim is not a clinical validation.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .falsification_program import (
    CLAIM_CORROBORATED,
    CLAIM_REFUTED,
    FalsificationProgram,
)
from .grounding_judge import GroundingJudge

CLAIM_ID = "rlm.dual_path_beats_single_path"

OUTCOME_REFUTED = "refuted"
OUTCOME_CORROBORATED = "corroborated"
OUTCOME_INCONCLUSIVE = "inconclusive"


@dataclass
class PairedCaseResult:
    """One case evaluated under both strategies."""

    case_id: str
    one_shot_faithfulness: float
    dual_path_faithfulness: float
    one_shot_recall: float
    dual_path_recall: float

    @property
    def faithfulness_delta(self) -> float:
        return self.dual_path_faithfulness - self.one_shot_faithfulness

    @property
    def recall_delta(self) -> float:
        return self.dual_path_recall - self.one_shot_recall


@dataclass
class ComparisonReport:
    outcome: str
    faithfulness_delta: float
    faithfulness_interval: tuple[float, float]
    recall_delta: float
    recall_interval: tuple[float, float]
    case_count: int
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": CLAIM_ID,
            "outcome": self.outcome,
            "case_count": self.case_count,
            "faithfulness_delta": round(self.faithfulness_delta, 4),
            "faithfulness_interval": [round(value, 4) for value in self.faithfulness_interval],
            "recall_delta": round(self.recall_delta, 4),
            "recall_interval": [round(value, 4) for value in self.recall_interval],
            "reasons": list(self.reasons),
        }


@dataclass
class DualPathComparison:
    """Evaluate the dual-path claim against its registered refutation criterion."""

    judge: GroundingJudge = field(default_factory=GroundingJudge)
    bootstrap_samples: int = 2000
    seed: int = 20260901
    min_cases: int = 20

    def score_case(
        self,
        case_id: str,
        one_shot_claims: Sequence[tuple[str, Sequence[Any]]],
        dual_path_claims: Sequence[tuple[str, Sequence[Any]]],
        one_shot_recall: float,
        dual_path_recall: float,
    ) -> PairedCaseResult:
        return PairedCaseResult(
            case_id=case_id,
            one_shot_faithfulness=self.judge.faithfulness(one_shot_claims),
            dual_path_faithfulness=self.judge.faithfulness(dual_path_claims),
            one_shot_recall=one_shot_recall,
            dual_path_recall=dual_path_recall,
        )

    def evaluate(self, results: Sequence[PairedCaseResult]) -> ComparisonReport:
        if not results:
            return ComparisonReport(
                outcome=OUTCOME_INCONCLUSIVE,
                faithfulness_delta=0.0,
                faithfulness_interval=(0.0, 0.0),
                recall_delta=0.0,
                recall_interval=(0.0, 0.0),
                case_count=0,
                reasons=["no cases evaluated"],
            )

        faithfulness_deltas = [result.faithfulness_delta for result in results]
        recall_deltas = [result.recall_delta for result in results]

        faithfulness_delta = _mean(faithfulness_deltas)
        recall_delta = _mean(recall_deltas)
        faithfulness_interval = self._bootstrap_interval(faithfulness_deltas)
        recall_interval = self._bootstrap_interval(recall_deltas)

        reasons: list[str] = []

        if len(results) < self.min_cases:
            return ComparisonReport(
                outcome=OUTCOME_INCONCLUSIVE,
                faithfulness_delta=faithfulness_delta,
                faithfulness_interval=faithfulness_interval,
                recall_delta=recall_delta,
                recall_interval=recall_interval,
                case_count=len(results),
                reasons=[f"fewer than {self.min_cases} paired cases; result not interpretable"],
            )

        # Refutation arm one: faithfulness regression.
        if faithfulness_interval[1] < 0:
            reasons.append("dual-path faithfulness is below the one-shot baseline")
        # Refutation arm two: recall gain indistinguishable from zero.
        if recall_interval[0] <= 0 <= recall_interval[1]:
            reasons.append("recall gain is not distinguishable from the one-shot path")

        if reasons:
            outcome = OUTCOME_REFUTED
        elif faithfulness_interval[0] >= 0 and recall_interval[0] > 0:
            outcome = OUTCOME_CORROBORATED
            reasons.append("recall gain is positive and faithfulness does not regress")
        else:
            outcome = OUTCOME_INCONCLUSIVE
            reasons.append("intervals do not settle the claim in either direction")

        return ComparisonReport(
            outcome=outcome,
            faithfulness_delta=faithfulness_delta,
            faithfulness_interval=faithfulness_interval,
            recall_delta=recall_delta,
            recall_interval=recall_interval,
            case_count=len(results),
            reasons=reasons,
        )

    def resolve_claim(self, report: ComparisonReport, program: FalsificationProgram, run_id: str) -> str | None:
        """Record the outcome against the registered claim.

        An inconclusive run leaves the claim open, which is the correct state: a
        study that did not settle the question has not settled it.
        """
        if report.outcome == OUTCOME_REFUTED:
            program.resolve(CLAIM_ID, CLAIM_REFUTED, evidence=f"{run_id}: {'; '.join(report.reasons)}")
            return CLAIM_REFUTED
        if report.outcome == OUTCOME_CORROBORATED:
            program.resolve(CLAIM_ID, CLAIM_CORROBORATED, evidence=f"{run_id}: {'; '.join(report.reasons)}")
            return CLAIM_CORROBORATED
        return None

    def _bootstrap_interval(self, values: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
        if not values:
            return (0.0, 0.0)
        if len(values) == 1:
            return (values[0], values[0])
        generator = random.Random(self.seed)
        size = len(values)
        means = []
        for _ in range(self.bootstrap_samples):
            sample = [values[generator.randrange(size)] for _ in range(size)]
            means.append(_mean(sample))
        means.sort()
        tail = (1.0 - confidence) / 2.0
        lower_index = max(0, int(tail * len(means)) - 1)
        upper_index = min(len(means) - 1, int((1.0 - tail) * len(means)))
        return (means[lower_index], means[upper_index])


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
