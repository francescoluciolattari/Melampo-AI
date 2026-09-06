"""B4 phase one: does the dream branch catch diagnoses the differential misses?

The claim under test contains two questions, and only the second needs a
clinician. The first — do the enumerated hypotheses capture diagnoses the main
differential missed — is answered by cases with a **documented outcome**, and
that is what this harness measures.

The measure is deliberately narrow. Not "is the branch right", which the main
differential mostly answers already, but: **among the cases where the primary
differential misses the documented diagnosis, in how many does the branch
contain it?** That isolates the value of asking "what else could this be" from
the value of the diagnosis itself, and it is the quantity expertise research
identifies as decisive — a clinician's reasoning succeeds or fails largely on
whether the correct hypothesis entered the initial set at all.

Three further measures, none requiring a clinician:

- **Stratification by density.** Coverage is uneven, so capture is reported per
  band of local graph density. A rate that rises with density is evidence the
  mechanism works and the graph is the limit; a flat rate is evidence the
  mechanism does not.
- **Attention cost.** Hypotheses are not free: each one consumes review. The
  ratio of useful to emitted is computable without judgement, and a branch
  catching one missed diagnosis every thirty hypotheses costs more than it
  returns.
- **Capture@K.** Whether the captured hypothesis leads to the investigation that
  would confirm it, which is what makes a hypothesis actionable rather than
  merely present.

Refutation is against a **base rate**: the capture rate is only evidence if it
exceeds what picking conditions at random from the graph would achieve. Without
that comparison a high rate may reflect a small candidate set rather than a
working mechanism.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

DENSITY_BANDS = ((0.0, 0.25, "sparse"), (0.25, 0.6, "moderate"), (0.6, 1.01, "dense"))


@dataclass(frozen=True)
class EvaluationCase:
    """One case with its documented diagnosis, held apart from the presentation."""

    case_id: str
    presentation: str
    documented_diagnosis: str
    candidate_conditions: tuple[str, ...] = ()
    source: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "documented_diagnosis": self.documented_diagnosis,
            "source": self.source,
            "candidate_count": len(self.candidate_conditions),
        }


@dataclass
class CaseOutcome:
    """What happened on one case."""

    case_id: str
    documented_diagnosis: str
    differential: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    density: float = 0.0
    discriminating_tests: list[str] = field(default_factory=list)

    @property
    def differential_hit(self) -> bool:
        return _contains(self.differential, self.documented_diagnosis)

    @property
    def hypothesis_hit(self) -> bool:
        return _contains(self.hypotheses, self.documented_diagnosis)

    @property
    def is_capture(self) -> bool:
        """The case that matters: the differential missed it and the branch caught it."""
        return not self.differential_hit and self.hypothesis_hit

    @property
    def band(self) -> str:
        return next(label for low, high, label in DENSITY_BANDS if low <= self.density < high)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "differential_hit": self.differential_hit,
            "hypothesis_hit": self.hypothesis_hit,
            "capture": self.is_capture,
            "density": round(self.density, 3),
            "band": self.band,
            "hypotheses_emitted": len(self.hypotheses),
        }


@dataclass
class CaptureReport:
    outcomes: list[CaseOutcome] = field(default_factory=list)
    base_rate: float | None = None

    @property
    def missed_by_differential(self) -> list[CaseOutcome]:
        return [item for item in self.outcomes if not item.differential_hit]

    @property
    def captures(self) -> list[CaseOutcome]:
        return [item for item in self.outcomes if item.is_capture]

    @property
    def capture_rate(self) -> float:
        """Among cases the differential missed, the fraction the branch caught."""
        missed = self.missed_by_differential
        return len(self.captures) / len(missed) if missed else 0.0

    @property
    def hypotheses_emitted(self) -> int:
        return sum(len(item.hypotheses) for item in self.outcomes)

    @property
    def attention_cost(self) -> float:
        """Hypotheses emitted per diagnosis captured. Lower is better."""
        return self.hypotheses_emitted / len(self.captures) if self.captures else float("inf")

    def by_band(self) -> dict[str, dict[str, Any]]:
        bands: dict[str, list[CaseOutcome]] = {}
        for outcome in self.missed_by_differential:
            bands.setdefault(outcome.band, []).append(outcome)
        return {
            band: {
                "missed": len(items),
                "captured": sum(1 for item in items if item.is_capture),
                "capture_rate": round(sum(1 for item in items if item.is_capture) / len(items), 3),
            }
            for band, items in sorted(bands.items())
        }

    def exceeds_base_rate(self) -> bool | None:
        """Whether capture beats picking candidates at random. None if not computed."""
        if self.base_rate is None:
            return None
        return self.capture_rate > self.base_rate

    def as_dict(self) -> dict[str, Any]:
        return {
            "cases": len(self.outcomes),
            "missed_by_differential": len(self.missed_by_differential),
            "captured_by_branch": len(self.captures),
            "capture_rate": round(self.capture_rate, 4),
            "base_rate": None if self.base_rate is None else round(self.base_rate, 4),
            "exceeds_base_rate": self.exceeds_base_rate(),
            "hypotheses_emitted": self.hypotheses_emitted,
            "attention_cost": None if self.attention_cost == float("inf") else round(self.attention_cost, 2),
            "by_density_band": self.by_band(),
            "outcomes": [item.as_dict() for item in self.outcomes],
        }


def evaluate(outcomes: Sequence[CaseOutcome], *, base_rate: float | None = None) -> CaptureReport:
    return CaptureReport(outcomes=list(outcomes), base_rate=base_rate)


def estimate_base_rate(
    cases: Sequence[EvaluationCase], hypotheses_per_case: int, *, seed: int = 20260906, trials: int = 200
) -> float:
    """Capture rate obtained by drawing candidates at random, as the floor to beat.

    A capture rate is only evidence of a working mechanism if it exceeds what
    chance would give on the same candidate sets — a small candidate list makes
    any selector look effective.
    """
    generator = random.Random(seed)
    usable = [case for case in cases if case.candidate_conditions]
    if not usable or hypotheses_per_case <= 0:
        return 0.0
    hits = 0
    for _ in range(trials):
        for case in usable:
            pool = list(case.candidate_conditions)
            drawn = generator.sample(pool, min(hypotheses_per_case, len(pool)))
            if _contains(drawn, case.documented_diagnosis):
                hits += 1
    return hits / (trials * len(usable))


def capture_at_k(outcome: CaseOutcome, confirming_test: str) -> bool:
    """Whether the captured hypothesis led to the investigation that would confirm it.

    A hypothesis present but unactionable has not done the work: the point of
    raising it is to make the next step obvious.
    """
    return outcome.is_capture and _contains(outcome.discriminating_tests, confirming_test)


def _contains(items: Sequence[str], target: str) -> bool:
    needle = _normalise(target)
    return any(_normalise(item) == needle for item in items)


def _normalise(value: str) -> str:
    return " ".join(str(value).lower().split())
