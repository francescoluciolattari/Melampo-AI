"""Measure whether a candidate root model can emit the engine's action format.

Published instruction-following benchmarks cannot settle this. IFEval saturates
— models score far higher on its detectable-format subset than on harder format
benchmarks, and the gap between models there is narrow — so a leaderboard
position does not predict whether a given model will write `grep(prednisone)`
rather than `grep prednisone` under this particular grammar. Failure modes are
model-specific, and telling them apart needs the trace rather than the metric.

So this bench produces the trace. Two numbers decide the choice:

- **Adherence**: of the non-empty lines a model emitted, what fraction the
  parser accepted. This is the number that matters, because a rejected line is
  not a degraded action — it is no action at all.
- **Completion**: what fraction of runs reached `final()`. A model can emit
  well-formed actions forever and never declare completion, which the engine
  records as budget exhaustion, not success.

Both come from machinery that already exists: `parse_actions` returns what it
ignored, and every trajectory carries its stop reason. The bench adds counting
and the near-miss inspection, not new measurement.

Near misses are collected separately because they are the actionable half of a
bad result. A model writing prose needs a different prompt; a model writing
`grep prednisone` needs one line of tolerance in the parser. The first is
research, the second is an afternoon, and the raw adherence figure does not
distinguish them.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..memory.context_environment import EnvironmentDocument
from ..reasoning.rlm_engine import STOP_FINAL, Budget, RlmEngine, parse_actions

# A line that names a verb but does not invoke it: the model understood the task
# and missed the syntax. Distinguished from prose because the remedy differs.
_NEAR_MISS = re.compile(
    r"^\s*(describe|grep|slice|search|expand|query|final)\b(?!\s*\()", re.IGNORECASE
)


@dataclass(frozen=True)
class BenchCase:
    case_id: str
    documents: tuple[EnvironmentDocument, ...]
    question: str


@dataclass
class ModelResult:
    """One candidate's behaviour across the bench."""

    model_name: str
    runs: int = 0
    completed_runs: int = 0
    accepted_lines: int = 0
    rejected_lines: int = 0
    near_misses: list[str] = field(default_factory=list)
    prose_lines: list[str] = field(default_factory=list)
    stop_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def adherence(self) -> float:
        total = self.accepted_lines + self.rejected_lines
        return self.accepted_lines / total if total else 0.0

    @property
    def completion_rate(self) -> float:
        return self.completed_runs / self.runs if self.runs else 0.0

    @property
    def near_miss_share(self) -> float:
        """Of the rejected lines, how many were syntax rather than misunderstanding.

        A high share means the prompt is landing and the parser is strict; a low
        share means the model is not attempting actions at all.
        """
        return len(self.near_misses) / self.rejected_lines if self.rejected_lines else 0.0

    def prose_examples_present(self) -> bool:
        return bool(self.prose_lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "runs": self.runs,
            "adherence": round(self.adherence, 4),
            "completion_rate": round(self.completion_rate, 4),
            "accepted_lines": self.accepted_lines,
            "rejected_lines": self.rejected_lines,
            "near_miss_share": round(self.near_miss_share, 4),
            "near_miss_examples": self.near_misses[:5],
            "prose_examples": self.prose_lines[:5],
            "stop_reasons": dict(sorted(self.stop_reasons.items())),
        }


@dataclass
class BenchReport:
    results: list[ModelResult] = field(default_factory=list)
    adherence_target: float = 0.95

    def ranked(self) -> list[ModelResult]:
        return sorted(self.results, key=lambda item: (-item.adherence, -item.completion_rate))

    def verdict(self) -> str:
        """State what the numbers decide, including when they decide nothing."""
        if not self.results:
            return "no models benched"
        best = self.ranked()[0]
        if best.adherence >= self.adherence_target:
            return (
                f"{best.model_name} meets the adherence target "
                f"({best.adherence:.0%}); the choice is settled on these cases"
            )

        producing_output = [item for item in self.results if item.accepted_lines or item.rejected_lines]
        if not producing_output:
            # Every candidate returned nothing at all: this is not a format
            # problem, since there is no output to have a format. A silent
            # 0/0 near-miss share must not be read as "mostly near misses" --
            # that would misdiagnose a connectivity or auth failure as a
            # prompt problem and send the operator down the wrong fix.
            return (
                "no model produced any output at all; check API keys, network reachability and "
                "provider errors before revisiting the prompt or the model choice"
            )

        rejecting = [item for item in producing_output if item.rejected_lines]
        if rejecting and all(item.near_miss_share > 0.5 for item in rejecting):
            return (
                "no model meets the target, but most rejections are near misses: "
                "the models understand the task and miss the syntax, so this is prompt "
                "and parser work rather than a model choice"
            )
        return (
            "no model meets the target and rejections are mostly prose: "
            "the prompt is not conveying the action format, and no model choice fixes that"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "models": len(self.results),
            "adherence_target": self.adherence_target,
            "verdict": self.verdict(),
            "results": [item.as_dict() for item in self.ranked()],
        }


def bench_model(
    model_name: str,
    root_model: Callable[[str], str],
    cases: Sequence[BenchCase],
    *,
    budget_factory: Callable[[], Budget] = Budget,
) -> ModelResult:
    """Run one candidate over the cases and count what the parser made of it."""
    engine = RlmEngine(root_model=_counting(root_model, collector := []), depth=0)
    result = ModelResult(model_name=model_name)

    for case in cases:
        trajectory = engine.run(case.case_id, case.documents, case.question, budget=budget_factory())
        result.runs += 1
        if trajectory.stop_reason == STOP_FINAL:
            result.completed_runs += 1
        reason = trajectory.stop_reason or "unknown"
        result.stop_reasons[reason] = result.stop_reasons.get(reason, 0) + 1

    for output in collector:
        actions, ignored = parse_actions(output)
        result.accepted_lines += len(actions)
        result.rejected_lines += len(ignored)
        for line in ignored:
            (result.near_misses if _NEAR_MISS.match(line) else result.prose_lines).append(line)
    return result


def bench_models(
    candidates: dict[str, Callable[[str], str]],
    cases: Sequence[BenchCase],
    *,
    adherence_target: float = 0.95,
) -> BenchReport:
    """Run every candidate over the same cases, so the comparison is paired."""
    report = BenchReport(adherence_target=adherence_target)
    for name, model in candidates.items():
        report.results.append(bench_model(name, model, cases))
    return report


def _counting(root_model: Callable[[str], str], collector: list[str]) -> Callable[[str], str]:
    """Wrap a model so every raw output is kept for parsing statistics."""

    def _wrapped(prompt: str) -> str:
        output = root_model(prompt)
        collector.append(output)
        return output

    return _wrapped
