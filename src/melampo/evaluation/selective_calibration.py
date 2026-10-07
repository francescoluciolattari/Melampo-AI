"""Which links to accept so that the error among accepted links is certified (T4 of the linker plan).

The linker gives every accepted link a convergence score (``LinkResult.convergence``: independent
mechanisms that support it minus the conflicts left). On a gold set, this module finds the loosest
score threshold at which the error rate among the links kept stays at or below ``alpha`` with
probability at least ``1 - delta``, by **fixed-sequence testing** (Learn-then-Test, Angelopoulos et al.,
arXiv 2110.01052; the split fixed-sequence procedure of MAPIE for precision control):

* thresholds are tested from the strictest to the loosest, an order fixed before looking at the
  labels; for each the null hypothesis is "the error among the links kept is above alpha";
* the p-value is the binomial tail P(Bin(n, alpha) <= errors) on the n links kept;
* testing stops at the first threshold that is not rejected; every threshold rejected before it is
  certified, and the family-wise error is at most ``delta`` with no correction, because the
  sequence is fixed.

**Mondrian** (per stratum: language, frame, stage...): the same test inside each stratum, so that a
good average cannot hide a bad stratum. ``simultaneous=True`` divides delta by the number of
strata (Bonferroni), for one guarantee over all of them at once.

What the guarantee is and is not. It holds for the population the gold set was drawn from
(exchangeable items: same hospitals, templates, models, lexicon) and for the links the system
accepts, not for one link. A new hospital, model, prompt or lexicon needs a new calibration. With
zero errors, certifying alpha = 1% at delta = 5% needs at least 299 accepted links in the stratum
(``links_needed``).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any


def binomial_cdf(k: int, n: int, p: float) -> float:
    """P(Bin(n, p) <= k), summed in log space so that n in the thousands stays exact enough."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    total = 0.0
    for i in range(k + 1):
        total += math.exp(
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * log_p
            + (n - i) * log_q
        )
    return min(1.0, total)


def p_value(errors: int, n: int, alpha: float) -> float:
    """Evidence against "the error rate is above alpha", from ``errors`` in ``n`` accepted links."""
    if n == 0:
        return 1.0
    return binomial_cdf(errors, n, alpha)


def links_needed(alpha: float, delta: float, errors: int = 0) -> int:
    """Fewest accepted links with ``errors`` wrong that reject "error > alpha" at level delta."""
    if errors == 0:
        return math.ceil(math.log(delta) / math.log1p(-alpha))
    n = errors + 1
    while p_value(errors, n, alpha) > delta:
        n += 1
    return n


@dataclass(frozen=True)
class Step:
    threshold: float
    kept: int
    errors: int
    p_value: float
    certified: bool


@dataclass
class Certificate:
    """The loosest certified threshold (None: nothing certified) and every step tested."""

    alpha: float
    delta: float
    threshold: float | None
    kept: int
    errors: int
    total: int
    steps: list[Step] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return self.kept / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "delta": self.delta,
            "threshold": self.threshold,
            "kept": self.kept,
            "errors": self.errors,
            "total": self.total,
            "coverage_of_accepted": round(self.coverage, 4),
            "links_needed_with_no_error": links_needed(self.alpha, self.delta),
            "steps": [step.__dict__ for step in self.steps],
        }


def certify(
    scores: Sequence[float],
    correct: Sequence[bool],
    alpha: float = 0.01,
    delta: float = 0.05,
) -> Certificate:
    """Fixed-sequence test of the score thresholds, strictest first."""
    if len(scores) != len(correct):
        raise ValueError("one label per score")
    steps: list[Step] = []
    best: Step | None = None
    for threshold in sorted(set(scores), reverse=True):
        kept = [ok for score, ok in zip(scores, correct) if score >= threshold]
        errors = sum(1 for ok in kept if not ok)
        p = p_value(errors, len(kept), alpha)
        step = Step(threshold, len(kept), errors, p, p <= delta)
        steps.append(step)
        if not step.certified:
            break
        best = step
    return Certificate(
        alpha=alpha,
        delta=delta,
        threshold=best.threshold if best else None,
        kept=best.kept if best else 0,
        errors=best.errors if best else 0,
        total=len(scores),
        steps=steps,
    )


def certify_by_stratum(
    rows: Sequence[tuple[Hashable, float, bool]],
    alpha: float = 0.01,
    delta: float = 0.05,
    simultaneous: bool = False,
) -> dict[Hashable, Certificate]:
    """Mondrian certification: (stratum, score, correct) rows, one certificate per stratum."""
    grouped: dict[Hashable, tuple[list[float], list[bool]]] = defaultdict(
        lambda: ([], [])
    )
    for stratum, score, ok in rows:
        grouped[stratum][0].append(score)
        grouped[stratum][1].append(ok)
    level = delta / len(grouped) if simultaneous and grouped else delta
    return {
        stratum: certify(scores, oks, alpha, level)
        for stratum, (scores, oks) in sorted(grouped.items(), key=lambda kv: str(kv[0]))
    }
