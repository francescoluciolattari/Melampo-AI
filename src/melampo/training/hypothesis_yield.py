"""Learn which hypothesis shapes turn out to matter, from confirmed outcomes.

The dream branch can produce a model the engine then uses, and this is that
model — but what trains it decides whether it is learning or circularity.

A model trained on the hypotheses the system generated would learn to reproduce
its own imagination: fluent, self-consistent, and untethered. A model trained on
**which hypotheses were independently confirmed** learns something real, because
the signal comes from histology, outcome or a blinded review — evidence produced
by something that did not take part in the reasoning.

So the unit of learning is not a hypothesis but a *hypothesis with an outcome*,
and outcomes are admitted only through the confirmation registry.

What is learned is deliberately not a diagnosis. It is the **yield of a shape**:
given a path of two hops over well-attested edges corroborated by three
findings, how often did that pattern turn out to matter? The engine uses it to
decide what is worth a clinician's attention, which is the scarcest resource in
the system and the one an unranked channel spends fastest.

The estimator is empirical rates per feature bucket with Wilson intervals, not a
fitted network. Three reasons, and the third is the one that matters most: the
result is inspectable, a bucket with three observations stays visibly wide
instead of pretending to knowledge, and a rate over counted outcomes can be
explained to a reviewer in a sentence.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..memory.ontology_import import wilson_interval

FEATURE_HOPS = "hops"
FEATURE_SUPPORT = "support"
FEATURE_CORROBORATION = "corroboration"
FEATURE_GAP_COUNT = "gap_count"

# Bucket edges chosen to separate shapes a clinician would describe differently,
# not to optimise a fit: a direct link versus a mechanism versus a long chain;
# a well-attested edge versus a sparsely reported one.
SUPPORT_BUCKETS = ((0.0, 0.2, "weak"), (0.2, 0.6, "moderate"), (0.6, 1.01, "strong"))
HOP_BUCKETS = ((1, 2, "direct"), (2, 3, "mechanism"), (3, 99, "chain"))

MIN_OBSERVATIONS = 5


@dataclass(frozen=True)
class HypothesisFeatures:
    """The shape of a hypothesis, independent of its content."""

    hops: int
    support: float
    corroboration: int = 1
    gap_count: int = 0

    def bucket(self) -> str:
        """A stable key describing the shape, readable in a report."""
        support_label = next(
            label for low, high, label in SUPPORT_BUCKETS if low <= self.support < high
        )
        hop_label = next(label for low, high, label in HOP_BUCKETS if low <= self.hops < high)
        corroboration_label = "corroborated" if self.corroboration >= 2 else "single"
        gap_label = "with_gap" if self.gap_count else "attested"
        return f"{hop_label}|{support_label}|{corroboration_label}|{gap_label}"

    def as_dict(self) -> dict[str, Any]:
        return {
            FEATURE_HOPS: self.hops,
            FEATURE_SUPPORT: round(self.support, 3),
            FEATURE_CORROBORATION: self.corroboration,
            FEATURE_GAP_COUNT: self.gap_count,
            "bucket": self.bucket(),
        }

    @classmethod
    def from_hypothesis(cls, hypothesis: Any) -> "HypothesisFeatures":
        """Read the shape off a ``MechanismHypothesis`` without importing it."""
        return cls(
            hops=int(getattr(hypothesis, "shortest_hops", 1) or 1),
            support=float(getattr(hypothesis, "support", 0.0) or 0.0),
            corroboration=int(getattr(hypothesis, "corroboration", 1) or 1),
            gap_count=int(getattr(hypothesis, "gap_count", 0) or 0),
        )


@dataclass(frozen=True)
class HypothesisOutcome:
    """A hypothesis that was surfaced, and what independently became of it."""

    case_id: str
    condition: str
    features: HypothesisFeatures
    confirmed: bool
    confirmation_source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "condition": self.condition,
            "features": self.features.as_dict(),
            "confirmed": self.confirmed,
            "confirmation_source": self.confirmation_source,
        }


@dataclass(frozen=True)
class YieldEstimate:
    """How often a shape turned out to matter, as an interval."""

    bucket: str
    confirmed: int
    observed: int
    lower: float
    upper: float

    @property
    def is_established(self) -> bool:
        """Whether enough outcomes exist for the estimate to carry weight."""
        return self.observed >= MIN_OBSERVATIONS

    def as_dict(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "confirmed": self.confirmed,
            "observed": self.observed,
            "lower": round(self.lower, 3),
            "upper": round(self.upper, 3),
            "established": self.is_established,
        }


@dataclass
class HypothesisYieldModel:
    """Empirical yield per hypothesis shape, learned from confirmed outcomes.

    Uninformative until outcomes accumulate, and it says so: an unobserved shape
    returns the full interval rather than a guess, so a new pattern is neither
    promoted nor suppressed on no evidence.
    """

    outcomes: list[HypothesisOutcome] = field(default_factory=list)
    min_observations: int = MIN_OBSERVATIONS

    def observe(self, outcome: HypothesisOutcome) -> None:
        self.outcomes.append(outcome)

    def observe_many(self, outcomes: Iterable[HypothesisOutcome]) -> int:
        before = len(self.outcomes)
        self.outcomes.extend(outcomes)
        return len(self.outcomes) - before

    def estimate(self, features: HypothesisFeatures) -> YieldEstimate:
        bucket = features.bucket()
        matching = [item for item in self.outcomes if item.features.bucket() == bucket]
        confirmed = sum(1 for item in matching if item.confirmed)
        if not matching:
            return YieldEstimate(bucket=bucket, confirmed=0, observed=0, lower=0.0, upper=1.0)
        lower, upper = wilson_interval(confirmed, len(matching))
        return YieldEstimate(
            bucket=bucket, confirmed=confirmed, observed=len(matching), lower=lower, upper=upper
        )

    def rank(self, hypotheses: Sequence[Any]) -> list[tuple[Any, YieldEstimate]]:
        """Order hypotheses by the yield their shape has demonstrated.

        Ranked by the **upper** bound, consistent with the rest of the
        hypothesis stream: exploration reads what could be true. An unobserved
        shape therefore sits high rather than low, which is the intended
        behaviour — a pattern nobody has measured is a reason to look, not a
        reason to suppress.
        """
        scored = [(item, self.estimate(HypothesisFeatures.from_hypothesis(item))) for item in hypotheses]
        scored.sort(key=lambda pair: (-pair[1].upper, -pair[1].lower))
        return scored

    def suppressed(self, features: HypothesisFeatures) -> bool:
        """Whether a shape has demonstrated enough futility to stop surfacing it.

        Requires an established estimate: a shape is only suppressed once enough
        outcomes exist, never on a handful. Silence on thin evidence would hide
        exactly the rare patterns the branch exists to surface.
        """
        estimate = self.estimate(features)
        return estimate.is_established and estimate.upper < 0.05

    def report(self) -> dict[str, Any]:
        buckets: dict[str, list[HypothesisOutcome]] = {}
        for outcome in self.outcomes:
            buckets.setdefault(outcome.features.bucket(), []).append(outcome)
        rows = []
        for bucket, items in sorted(buckets.items()):
            confirmed = sum(1 for item in items if item.confirmed)
            lower, upper = wilson_interval(confirmed, len(items))
            rows.append(
                YieldEstimate(bucket, confirmed, len(items), lower, upper).as_dict()
            )
        return {
            "outcomes": len(self.outcomes),
            "buckets": len(buckets),
            "established_buckets": sum(1 for row in rows if row["established"]),
            "by_bucket": rows,
        }


def outcomes_from_confirmations(
    surfaced: Sequence[dict[str, Any]], confirmations: Sequence[Any]
) -> list[HypothesisOutcome]:
    """Pair surfaced hypotheses with independent confirmations for the same case.

    Only confirmations that passed the registry are used, so what trains the
    model is always evidence from outside the system's own reasoning. A surfaced
    hypothesis with no confirmation for its case yields nothing: absence of a
    confirmation is not evidence that the hypothesis was wrong, and counting it
    as a failure would teach the model from silence.
    """
    by_case: dict[str, Any] = {}
    for confirmation in confirmations:
        if getattr(confirmation, "is_independent", False):
            by_case[getattr(confirmation, "case_id", "")] = confirmation

    outcomes: list[HypothesisOutcome] = []
    for item in surfaced:
        case_id = str(item.get("case_id", ""))
        confirmation = by_case.get(case_id)
        if confirmation is None:
            continue
        condition = str(item.get("condition", "")).strip()
        features = item.get("features")
        if not condition or not isinstance(features, HypothesisFeatures):
            continue
        outcomes.append(
            HypothesisOutcome(
                case_id=case_id,
                condition=condition,
                features=features,
                confirmed=_normalise(condition) == _normalise(getattr(confirmation, "diagnosis", "")),
                confirmation_source=str(getattr(confirmation, "source", "")),
            )
        )
    return outcomes


def _normalise(value: str) -> str:
    return " ".join(str(value).lower().split())
