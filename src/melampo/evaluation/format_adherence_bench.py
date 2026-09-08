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

# A live run left one candidate's job running 30.3 minutes against a 30-minute
# job timeout, cancelled without a result, while every other candidate in the
# same run finished comfortably inside it. The right response is not a wider
# timeout that waits out slowness -- if a candidate needs minutes for a single
# case, that is not a budget problem to accommodate, it is evidence the
# candidate is impractical for this workload.
#
# The threshold is a FRACTION of each case's own configured wall-clock
# ceiling, not a fixed number of seconds. Two workflows here use different
# per-case budgets (60s default for the full 21-candidate roster, 90s for the
# 8-candidate comparison's wider allowance) -- a fixed absolute threshold
# tuned to look right against one of those would be miscalibrated against the
# other: too low relative to a 90s ceiling flags candidates that are still
# working productively as "slow", too high relative to a 60s ceiling never
# trips at all, silently disabling the whole mechanism for that workflow.
# Comparing against each case's own ceiling stays correctly calibrated
# whichever budget is actually in use, including a budget neither workflow
# uses yet.
#
# The fraction was first set at 1.0 -- "took the case's entire nominal
# allowance" -- and a live run showed exactly the gap that choice left open.
# grok-4.6 was consistently at or beyond its ceiling and tripped correctly
# after three cases (5.8 minutes, clean diagnostic data). gemini-3-pro-preview,
# reached with a corrected slug in the same run, was never that far over --
# apparently reliably using most but not all of its 90s allowance, case after
# case -- so no three consecutive cases ever hit exactly 100%, the job ran
# past the 15-minute job timeout, and was killed externally before writing
# any result at all: a job timeout kills the process itself, which no
# try/except inside that process can catch, unlike every other failure mode
# this bench recovers from. Lowered to 0.75 so a candidate reliably spending
# three-quarters or more of its budget -- not only one exhausting it outright
# -- is recognised and abandoned with real diagnostic data, before the job
# timeout has to be the one to notice, with no data to show for it.
#
# If the last WINDOW consecutive cases each consumed at least this fraction
# of their own ceiling, further cases are not attempted: the candidate has
# already shown what it will keep showing, and running the rest would only
# spend more time and money confirming a conclusion already reached. WINDOW
# is 3 so a single slow case (a network blip, a transient provider queue)
# does not condemn an otherwise-fine candidate.
LATENCY_CIRCUIT_BREAKER_WINDOW = 3
LATENCY_CIRCUIT_BREAKER_THRESHOLD_FRACTION = 0.75



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
    # Iterations actually used, split by outcome. Distinguishes "ran out of
    # budget" from "task genuinely too hard": a model landing at exactly the
    # iteration cap every time was never given the chance to finish, while one
    # stopping well short of the cap and still not calling final() has a
    # different problem the budget cannot fix.
    iterations_on_completion: list[int] = field(default_factory=list)
    # Each incomplete run's (iterations used, that run's own ceiling), so
    # budget_bound compares each run against the budget it actually had
    # rather than assuming every case shares one ceiling.
    iterations_on_incompletion: list[tuple[int, int]] = field(default_factory=list)
    # Real wall-clock seconds actually spent per case, in the order cases were
    # run -- distinct from iteration counts, which say how many turns a model
    # took but nothing about how long each one took. A candidate needing many
    # quick turns and a candidate needing few slow ones can share the same
    # iteration count and look identical without this.
    case_elapsed_seconds: list[float] = field(default_factory=list)
    # Per-case elapsed_seconds / that case's own max_wall_clock_seconds. Kept
    # separate from case_elapsed_seconds because the circuit breaker compares
    # against each case's own ceiling, not an absolute number -- see
    # LATENCY_CIRCUIT_BREAKER_THRESHOLD_FRACTION.
    case_latency_ratios: list[float] = field(default_factory=list)
    # Set when the latency circuit breaker trips: see
    # LATENCY_CIRCUIT_BREAKER_WINDOW/THRESHOLD_SECONDS above. Distinguishes
    # "every case was attempted" from "the bench gave up on this candidate
    # early because its own recent behaviour already answered the question."
    abandoned_for_latency: bool = False
    cases_skipped_for_latency: int = 0

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

    @property
    def mean_iterations_on_completion(self) -> float | None:
        values = self.iterations_on_completion
        return sum(values) / len(values) if values else None

    @property
    def mean_iterations_on_incompletion(self) -> float | None:
        values = [used for used, _ceiling in self.iterations_on_incompletion]
        return sum(values) / len(values) if values else None

    @property
    def mean_case_seconds(self) -> float | None:
        values = self.case_elapsed_seconds
        return sum(values) / len(values) if values else None

    @property
    def max_case_seconds(self) -> float | None:
        return max(self.case_elapsed_seconds) if self.case_elapsed_seconds else None

    @property
    def budget_bound(self) -> bool:
        """Whether every incomplete run used the full iteration budget it was given.

        True suggests the budget was the limiting factor and a wider one might
        change the result; false suggests the model stopped short for another
        reason -- a malformed action it could not recover from, or simply
        never attempting to conclude -- which a wider budget will not fix.

        False is ambiguous on its own, and a real comparison exposed the
        ambiguity directly: mistral-large-openrouter completed every case
        (budget_bound False because there was nothing incomplete to be bound
        by -- the best possible outcome) and mistral-small-openrouter failed
        two cases by exhausting their iteration ceiling (budget_bound True).
        Both are meaningfully different from a candidate that failed cases
        for some OTHER reason -- a malformed action, giving up outright --
        which would also show budget_bound False, looking identical to
        "completed everything" from this field alone. See
        all_cases_completed for the distinction this property cannot make by
        itself: read them together, not this one in isolation.
        """
        return bool(self.iterations_on_incompletion) and all(
            used >= ceiling for used, ceiling in self.iterations_on_incompletion
        )

    @property
    def all_cases_completed(self) -> bool:
        """True only when completion_rate is exactly 100% -- the specific
        thing budget_bound's False cannot distinguish from "failed for a
        non-iteration reason" on its own."""
        return self.runs > 0 and self.completed_runs == self.runs

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
            "mean_iterations_on_completion": (
                None if self.mean_iterations_on_completion is None else round(self.mean_iterations_on_completion, 2)
            ),
            "mean_iterations_on_incompletion": (
                None
                if self.mean_iterations_on_incompletion is None
                else round(self.mean_iterations_on_incompletion, 2)
            ),
            "budget_bound": self.budget_bound,
            "all_cases_completed": self.all_cases_completed,
            "mean_case_seconds": None if self.mean_case_seconds is None else round(self.mean_case_seconds, 1),
            "max_case_seconds": None if self.max_case_seconds is None else round(self.max_case_seconds, 1),
            "abandoned_for_latency": self.abandoned_for_latency,
            "cases_skipped_for_latency": self.cases_skipped_for_latency,
        }


@dataclass
class BenchReport:
    results: list[ModelResult] = field(default_factory=list)
    adherence_target: float = 0.95

    def ranked(self) -> list[ModelResult]:
        """Same three-key order as rank_result_dicts: adherence, completion,
        then mean seconds per completed case as a tiebreaker among ties on
        the first two."""
        return sorted(
            self.results,
            key=lambda item: (
                -item.adherence,
                -item.completion_rate,
                item.mean_case_seconds if item.mean_case_seconds is not None else float("inf"),
            ),
        )

    def verdict(self) -> str:
        """State what the numbers decide, including when they decide nothing."""
        return compute_verdict([item.as_dict() for item in self.results], self.adherence_target)

    def as_dict(self) -> dict[str, Any]:
        """The full payload the script writes to the results file.

        Present since the tool's first commit; lost when BenchReport.verdict()
        was refactored to delegate to compute_verdict() in the parallel-matrix
        change, apparently dropped while the class body was being rewritten
        and never re-added. Every unit test that exercised this class called
        bench_model() directly and asserted on the resulting ModelResult
        (which has its own as_dict() and was never affected), so nothing
        locally exercised BenchReport.as_dict() itself -- the gap surfaced
        only on a live run, where _run() calls exactly this method and
        nothing stood in for it. Restored here, and a regression test now
        calls it directly rather than only through code that happens to
        reach it.
        """
        return {
            "models": len(self.results),
            "adherence_target": self.adherence_target,
            "verdict": self.verdict(),
            "results": [item.as_dict() for item in self.ranked()],
        }


def rank_result_dicts(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort result dicts the way ``BenchReport.ranked()`` sorts ``ModelResult``.

    Operates on the plain-dict shape ``ModelResult.as_dict()`` produces, so the
    same ordering applies whether the results just came from a live run or
    were loaded back from JSON written by a separate, earlier process --
    which is what a parallel, one-candidate-per-job bench run needs when a
    later step combines what each job wrote independently.

    Three keys, in order: adherence, completion, then mean seconds per
    completed case as a tiebreaker. Added after a real run left three
    candidates tied at 100% adherence and 100% completion -- gemma-3-27b,
    mistral-large-openrouter, nemotron-3-super -- with nemotron running
    3-4x faster per case than the other two, a difference already measured
    (mean_case_seconds) but invisible in the verdict because nothing ranked
    on it. A missing mean_case_seconds (an older result predating that
    field, or a candidate with zero completed cases) sorts last among ties
    rather than raising or silently defaulting to "fastest" -- infinity is
    the value that means "the tiebreaker has no answer for this one",
    consistent with adherence/completion still deciding the primary order
    regardless.
    """
    def _tiebreak_seconds(item: dict[str, Any]) -> float:
        value = item.get("mean_case_seconds")
        return value if isinstance(value, (int, float)) else float("inf")

    return sorted(
        results,
        key=lambda item: (-item["adherence"], -item["completion_rate"], _tiebreak_seconds(item)),
    )


def compute_verdict(results: Sequence[dict[str, Any]], adherence_target: float = 0.95) -> str:
    """The verdict logic, as a pure function over result dicts.

    Extracted from ``BenchReport.verdict()`` so a merge step recombining
    results that were computed in separate processes (one per candidate, run
    in parallel) reaches the same conclusion a single sequential run would
    have -- one place decides what the numbers mean, read from either a live
    ``BenchReport`` or a set of files on disk.
    """
    if not results:
        return "no models benched"
    ranked = rank_result_dicts(results)
    best = ranked[0]
    if best["adherence"] >= adherence_target:
        tied = [
            item
            for item in ranked
            if item["adherence"] == best["adherence"] and item["completion_rate"] == best["completion_rate"]
        ]
        if len(tied) > 1:
            # Several candidates reached the same adherence and completion --
            # the case a real run first showed with three candidates at
            # 100%/100%. The tiebreak already happened inside rank_result_dicts
            # (mean seconds per completed case); naming it here means the
            # verdict states why this one specifically, rather than reporting
            # a number tied with others as if it had settled the question
            # alone.
            return (
                f"{best['model_name']} meets the adherence target ({best['adherence']:.0%}) "
                f"and is fastest among {len(tied)} candidates tied on adherence and completion "
                f"({best.get('mean_case_seconds', 'n/a')}s mean per completed case); "
                "the choice is settled on these cases"
            )
        return (
            f"{best['model_name']} meets the adherence target "
            f"({best['adherence']:.0%}); the choice is settled on these cases"
        )

    producing_output = [item for item in results if item["accepted_lines"] or item["rejected_lines"]]
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

    rejecting = [item for item in producing_output if item["rejected_lines"]]
    if rejecting and all(item["near_miss_share"] > 0.5 for item in rejecting):
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
    """Run one candidate over the cases and count what the parser made of it.

    Stops early if the latency circuit breaker trips (see
    LATENCY_CIRCUIT_BREAKER_WINDOW/THRESHOLD_SECONDS): a candidate whose last
    few cases each took real minutes has already shown what running the rest
    would show again, and finishing the full case list would only spend more
    time and money confirming a conclusion already reached.
    """
    engine = RlmEngine(root_model=_counting(root_model, collector := []), depth=0)
    result = ModelResult(model_name=model_name)

    for case in cases:
        budget = budget_factory()
        trajectory = engine.run(case.case_id, case.documents, case.question, budget=budget)
        result.runs += 1
        used = trajectory.budget.get("iterations", 0)
        if trajectory.stop_reason == STOP_FINAL:
            result.completed_runs += 1
            result.iterations_on_completion.append(used)
        else:
            result.iterations_on_incompletion.append((used, budget.max_iterations))
        reason = trajectory.stop_reason or "unknown"
        result.stop_reasons[reason] = result.stop_reasons.get(reason, 0) + 1

        elapsed = float(trajectory.budget.get("elapsed_seconds", 0.0) or 0.0)
        result.case_elapsed_seconds.append(elapsed)

        # Each case against its own ceiling, not a fixed number of seconds --
        # see the module-level comment on LATENCY_CIRCUIT_BREAKER_THRESHOLD_FRACTION
        # for why a fixed threshold would be miscalibrated for one of the two
        # workflows that call this with different per-case wall-clock budgets.
        ceiling = max(budget.max_wall_clock_seconds, 1e-9)  # guard a pathological zero-length budget
        result.case_latency_ratios.append(elapsed / ceiling)

        recent = result.case_latency_ratios[-LATENCY_CIRCUIT_BREAKER_WINDOW:]
        if len(recent) >= LATENCY_CIRCUIT_BREAKER_WINDOW and all(
            ratio >= LATENCY_CIRCUIT_BREAKER_THRESHOLD_FRACTION for ratio in recent
        ):
            result.abandoned_for_latency = True
            result.cases_skipped_for_latency = len(cases) - result.runs
            break

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
    budget_factory: Callable[[], Budget] = Budget,
) -> BenchReport:
    """Run every candidate over the same cases, so the comparison is paired."""
    report = BenchReport(adherence_target=adherence_target)
    for name, model in candidates.items():
        report.results.append(bench_model(name, model, cases, budget_factory=budget_factory))
    return report


def _counting(root_model: Callable[[str], str], collector: list[str]) -> Callable[[str], str]:
    """Wrap a model so every raw output is kept for parsing statistics."""

    def _wrapped(prompt: str) -> str:
        output = root_model(prompt)
        collector.append(output)
        return output

    return _wrapped
