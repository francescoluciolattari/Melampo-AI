"""Propose investigations that discriminate between competing hypotheses.

``DifferentialEngine`` already returns ``recommended_tests``, but they come from
``_recommended_actions_for_domain`` — fixed strings chosen by domain, such as
"repeat targeted imaging review" or "expand corroborating evidence". They never
inspect the hypotheses in contention, so they are process reminders rather than
discriminating investigations: the same list is returned whether the
differential holds one hypothesis or five.

The question worth answering is different. Given several hypotheses and their
current weights, which observation would most change the picture? That is
expected information gain, and it is computable here because the concept graph
already encodes which findings attach to which conditions and how strongly.

Three properties follow from computing it rather than listing it:

- A test attaching equally to every hypothesis scores zero and is dropped from
  the suggestions. It may still be clinically necessary; it simply does not
  discriminate, and ``include_non_discriminating`` reports it with a gain of
  zero for callers who want that stated rather than implied.
- A test attaching to exactly one of two evenly weighted hypotheses scores
  highest, which is what a clinician means by a decisive investigation.
- When one hypothesis already dominates, every score collapses toward zero.
  There is little left to learn, and the system stops proposing work.

Information and burden are reported separately. Collapsing them into one score
hides the trade-off a clinician is entitled to make, and the same reasoning
applies here as to novelty and support in hypothesis enumeration.

**Scope.** This module ranks observations by how much they would resolve
uncertainty. It is not a care recommendation, carries no view on whether an
investigation is appropriate for a patient, and its output is decision support
requiring clinical judgement. Only concepts present in the graph are proposed:
nothing is invented.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..memory.concept_paths import ConceptEdge, ConceptGraphView, normalise_concept

DEFAULT_TEST_RELATIONS = frozenset(
    {
        "indicates",
        "detects",
        "diagnoses",
        "confirms",
        "positive_in",
        "abnormal_in",
        "elevated_in",
        "finding_of",
        "investigation_for",
    }
)

MIN_PROBABILITY = 1e-9


@dataclass(frozen=True)
class WeightedHypothesis:
    """A hypothesis under consideration, with its current weight."""

    label: str
    weight: float

    @classmethod
    def from_differential(cls, differential: dict[str, Any]) -> list["WeightedHypothesis"]:
        """Extract hypotheses from a ``DifferentialEngine`` payload."""
        hypotheses = differential.get("hypotheses") if isinstance(differential, dict) else None
        if not isinstance(hypotheses, list):
            return []
        extracted: list[WeightedHypothesis] = []
        for item in hypotheses:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip()
            if not label:
                continue
            extracted.append(cls(label=label, weight=max(0.0, _as_float(item.get("score"), 0.0))))
        return extracted


@dataclass(frozen=True)
class Investigation:
    """A proposed observation, ranked by how much uncertainty it would resolve."""

    name: str
    information_gain: float
    burden: float
    discriminates_between: tuple[str, ...]
    likelihoods: dict[str, float] = field(default_factory=dict)
    provenance: tuple[dict[str, Any], ...] = ()
    information_gain_lower: float = 0.0

    @property
    def gain_per_burden(self) -> float:
        """Reported alongside raw gain, never instead of it."""
        return round(self.information_gain / self.burden, 4) if self.burden > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": "discriminating_test",
            "information_gain_bits": round(self.information_gain, 4),
            "information_gain_lower_bits": round(self.information_gain_lower, 4),
            "burden": round(self.burden, 3),
            "gain_per_burden": self.gain_per_burden,
            "discriminates_between": list(self.discriminates_between),
            "likelihoods": {key: round(value, 3) for key, value in sorted(self.likelihoods.items())},
            "provenance": [dict(item) for item in self.provenance],
            "decision_support_only": True,
            "requires_clinician_judgement": True,
        }


@dataclass
class DiscriminatingTestSelector:
    """Rank candidate investigations by expected information gain."""

    graph: ConceptGraphView
    test_relations: frozenset[str] = DEFAULT_TEST_RELATIONS
    absent_likelihood: float = 0.05
    absent_bounds: tuple[float, float] = (0.0, 1.0)
    min_information_gain: float = 0.01
    max_suggestions: int = 5
    burdens: dict[str, float] = field(default_factory=dict)
    default_burden: float = 1.0

    def candidate_tests(self, hypotheses: Sequence[WeightedHypothesis]) -> list[str]:
        """Concepts attached to at least one hypothesis by a test-like relation.

        Discovered from the graph rather than supplied, so the candidate set
        cannot contain an investigation the knowledge base does not know about.
        """
        found: list[str] = []
        for hypothesis in hypotheses:
            for edge in self.graph.edges_from(hypothesis.label):
                if not self._is_test_relation(edge):
                    continue
                name = normalise_concept(edge.target)
                if name and name != normalise_concept(hypothesis.label) and name not in found:
                    found.append(name)
        return found

    def rank(
        self,
        hypotheses: Sequence[WeightedHypothesis],
        candidate_tests: Iterable[str] | None = None,
        *,
        include_non_discriminating: bool = False,
    ) -> list[Investigation]:
        """Rank candidate investigations, most informative first.

        Tests below ``min_information_gain`` are dropped by default: a list of
        suggestions should contain suggestions. Set
        ``include_non_discriminating`` to keep them, ranked last. A test that
        attaches equally to every hypothesis may still be clinically necessary,
        and reporting it with a gain of zero states that plainly rather than
        leaving the caller to infer it from an absence.
        """
        prior = _normalise_weights(hypotheses)
        if len(prior) < 2:
            return []

        tests = list(candidate_tests) if candidate_tests is not None else self.candidate_tests(hypotheses)
        investigations: list[Investigation] = []

        for test in tests:
            likelihoods, provenance = self._likelihoods(test, prior)
            bounds = self._likelihood_bounds(test, prior)
            gain = expected_information_gain(prior, likelihoods)
            gain_lower = guaranteed_information_gain(prior, bounds)
            if gain < self.min_information_gain and not include_non_discriminating:
                continue
            discriminating = tuple(
                label
                for label, value in sorted(likelihoods.items())
                if abs(value - self.absent_likelihood) > 1e-9
            )
            investigations.append(
                Investigation(
                    name=test,
                    information_gain=gain,
                    burden=self.burdens.get(normalise_concept(test), self.default_burden),
                    discriminates_between=discriminating,
                    likelihoods=likelihoods,
                    provenance=provenance,
                    information_gain_lower=gain_lower,
                )
            )

        investigations.sort(
            key=lambda item: (-item.information_gain_lower, -item.information_gain, item.burden, item.name)
        )
        if include_non_discriminating:
            return investigations
        return investigations[: self.max_suggestions]

    def suggest(self, differential: dict[str, Any]) -> dict[str, Any]:
        """Convenience wrapper over a ``DifferentialEngine`` payload."""
        hypotheses = WeightedHypothesis.from_differential(differential)
        ranked = self.rank(hypotheses)
        return {
            "discriminating_tests": [item.as_dict() for item in ranked],
            "hypothesis_count": len(hypotheses),
            "prior_entropy_bits": round(entropy(list(_normalise_weights(hypotheses).values())), 4),
            "decision_support_only": True,
        }

    def _likelihood_bounds(self, test: str, prior: dict[str, float]) -> dict[str, tuple[float, float]]:
        """Likelihood intervals per hypothesis, widest matching edge winning."""
        bounds: dict[str, tuple[float, float]] = {}
        test_key = normalise_concept(test)
        for label in prior:
            found: tuple[float, float] | None = None
            for edge in self.graph.edges_from(label):
                if not self._is_test_relation(edge) or normalise_concept(edge.target) != test_key:
                    continue
                low, high = edge.bounds
                found = (low, high) if found is None else (min(found[0], low), max(found[1], high))
            bounds[label] = found if found is not None else self.absent_bounds
        return bounds

    def _is_test_relation(self, edge: ConceptEdge) -> bool:
        relation = str(edge.relation).lower()
        return relation.removeprefix("inverse_") in self.test_relations

    def _likelihoods(
        self, test: str, prior: dict[str, float]
    ) -> tuple[dict[str, float], tuple[dict[str, Any], ...]]:
        """P(test positive | hypothesis), read off the graph edge weights.

        A missing edge yields ``absent_likelihood`` rather than zero. Zero would
        assert that the finding is impossible under that hypothesis, which the
        absence of an edge does not establish — the graph being silent is not the
        same as the graph denying.

        The point estimate still reads silence as a low value, which is a
        softened version of the same error. ``_likelihood_bounds`` is the honest
        reading: a missing edge is the full interval, so it guarantees nothing
        and cannot win the ranking on the strength of what is unknown about it.
        """
        likelihoods = {label: self.absent_likelihood for label in prior}
        provenance: list[dict[str, Any]] = []
        test_key = normalise_concept(test)

        for label in prior:
            for edge in self.graph.edges_from(label):
                if not self._is_test_relation(edge):
                    continue
                if normalise_concept(edge.target) != test_key:
                    continue
                weight = max(0.0, min(1.0, edge.weight))
                if weight > likelihoods[label]:
                    likelihoods[label] = weight
                    provenance.append({"hypothesis": label, **edge.as_dict()})
        return likelihoods, tuple(provenance)


def guaranteed_information_gain(prior: dict[str, float], bounds: dict[str, tuple[float, float]]) -> float:
    """Worst-case information gain over the likelihood intervals.

    Ranking by a point estimate lets ignorance masquerade as diagnostic power: a
    test whose likelihood is unknown on one side scores higher than one
    documented on both, because the missing value is read as an extreme rather
    than as a blank. The guarantee is what the test delivers regardless of how
    the unknowns resolve.

    Expected information gain falls to zero when every likelihood coincides, so
    if the intervals share any common value that configuration is achievable and
    the guarantee is zero — a wide interval offers no floor. Otherwise the
    intervals are disjoint and the worst case is the configuration in which they
    are closest, obtained by clamping each toward the others.
    """
    if not bounds:
        return 0.0
    highest_lower = max(low for low, _ in bounds.values())
    lowest_upper = min(high for _, high in bounds.values())
    if highest_lower <= lowest_upper:
        return 0.0
    midpoint = (highest_lower + lowest_upper) / 2.0
    closest = {label: min(max(midpoint, low), high) for label, (low, high) in bounds.items()}
    return expected_information_gain(prior, closest)


def entropy(probabilities: Sequence[float]) -> float:
    """Shannon entropy in bits. Zero-probability outcomes contribute nothing."""
    total = 0.0
    for value in probabilities:
        if value > MIN_PROBABILITY:
            total -= value * math.log2(value)
    return total


def expected_information_gain(prior: dict[str, float], likelihoods: dict[str, float]) -> float:
    """Prior entropy minus expected posterior entropy, over a binary outcome.

    Never negative: on average an observation cannot increase uncertainty, and a
    negative result here would indicate a bug rather than a finding.
    """
    labels = list(prior)
    prior_values = [prior[label] for label in labels]

    positive_probability = sum(prior[label] * likelihoods.get(label, 0.0) for label in labels)
    negative_probability = 1.0 - positive_probability

    posterior_positive = _posterior(prior, likelihoods, positive=True, evidence=positive_probability)
    posterior_negative = _posterior(prior, likelihoods, positive=False, evidence=negative_probability)

    expected = positive_probability * entropy(posterior_positive) + negative_probability * entropy(posterior_negative)
    return max(0.0, entropy(prior_values) - expected)


def _posterior(
    prior: dict[str, float], likelihoods: dict[str, float], *, positive: bool, evidence: float
) -> list[float]:
    if evidence <= MIN_PROBABILITY:
        return []
    values = []
    for label, probability in prior.items():
        likelihood = likelihoods.get(label, 0.0)
        values.append(probability * (likelihood if positive else 1.0 - likelihood) / evidence)
    return values


def _normalise_weights(hypotheses: Sequence[WeightedHypothesis]) -> dict[str, float]:
    """Convert hypothesis weights to a probability distribution.

    Weights arrive as engine scores that need not sum to one. Non-positive
    totals fall back to a uniform distribution, which is the honest reading of
    scores that carry no information about relative standing.
    """
    positive = [(item.label, max(0.0, item.weight)) for item in hypotheses if item.label]
    if not positive:
        return {}
    total = sum(weight for _, weight in positive)
    if total <= MIN_PROBABILITY:
        share = 1.0 / len(positive)
        return {label: share for label, _ in positive}
    return {label: weight / total for label, weight in positive}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
