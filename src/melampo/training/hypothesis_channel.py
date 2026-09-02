"""Isolated channel delivering dream candidates as exclusion hypotheses.

A synthetic candidate is not evidence and must never enter the differential as
support for a conclusion. It can legitimately enter as a *hypothesis to be
excluded*, which is what a differential diagnosis consists of in the first
place: structured possibilities that direct the search for further evidence, not
established facts.

The distinction is epistemic, not textual. The same candidate is useful framed
as "consider also X, synthetically generated, not observed here" and is
contamination framed as "memory supports X". Enforcing that distinction with a
metadata flag alone is not sufficient: candidates sharing a collection with
clinical evidence compete for the same ``top_k`` slots and are ranked by the same
similarity function, so a single downstream caller that forgets to filter
reintroduces them as evidence silently. The separation therefore has to be
structural — a distinct namespace, a distinct retrieval channel and a single
authorised consumption point.

Hypotheses are also gated on diagnostic indeterminacy. When one hypothesis
already dominates, additional synthetic alternatives add noise rather than
information; they earn their place only when the differential is flat and the
areas disagree.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

HYPOTHESIS_NAMESPACE = "melampo_hypothesis_candidates"
HYPOTHESIS_ROLE = "exclusion_hypothesis"


@dataclass(frozen=True)
class IndeterminacyGate:
    """Decide whether the hypothesis channel may open for a case.

    Thresholds operate on metrics already produced by ``neuro_dynamics``:
    ``convergence_index`` (how strongly the differential has settled),
    ``conflict_load`` (how much the functional areas disagree) and a case risk
    score.
    """

    max_convergence_index: float = 0.55
    min_conflict_load: float = 0.35
    min_risk: float = 0.30

    def evaluate(self, dynamics: dict[str, Any] | None, *, risk: float = 0.0) -> dict[str, Any]:
        metrics = dynamics or {}
        convergence = _safe_float(metrics.get("convergence_index"), 1.0)
        conflict = _safe_float(metrics.get("conflict_load"), 0.0)
        risk_value = _safe_float(risk, 0.0)

        reasons: list[str] = []
        if convergence > self.max_convergence_index:
            reasons.append("differential already converged")
        if conflict < self.min_conflict_load:
            reasons.append("insufficient conflict between areas")
        if risk_value < self.min_risk:
            reasons.append("case risk below relevance threshold")

        return {
            "open": not reasons,
            "blocking_reasons": reasons,
            "convergence_index": round(convergence, 3),
            "conflict_load": round(conflict, 3),
            "risk": round(risk_value, 3),
        }

    def is_open(self, dynamics: dict[str, Any] | None, *, risk: float = 0.0) -> bool:
        return bool(self.evaluate(dynamics, risk=risk)["open"])


@dataclass
class HypothesisEnvelope:
    """A dream candidate wrapped for consumption by the differential engine only."""

    label: str
    rationale: str = ""
    origin: str = "dream_branch"
    novelty_score: float = 0.0
    supporting_patterns: list[str] = field(default_factory=list)

    def as_exclusion_hypothesis(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "role": HYPOTHESIS_ROLE,
            "rationale": self.rationale,
            "origin": self.origin,
            "novelty_score": round(float(self.novelty_score), 3),
            "supporting_patterns": list(self.supporting_patterns),
            "namespace": HYPOTHESIS_NAMESPACE,
            "learning_status": "candidate",
            "source_type": "synthetic_dream_candidate",
            "synthetic_candidate_not_clinical_truth": True,
            "usable_as_evidence": False,
            "human_review_before_clinical_use": True,
        }


@dataclass
class HypothesisChannel:
    """Gated delivery of exclusion hypotheses into the differential.

    The channel never writes to semantic memory and never returns evidence. It
    is a read path with one authorised consumer.
    """

    gate: IndeterminacyGate = field(default_factory=IndeterminacyGate)
    max_hypotheses: int = 3

    def open_for(
        self,
        candidates: Iterable[HypothesisEnvelope],
        *,
        dynamics: dict[str, Any] | None = None,
        risk: float = 0.0,
    ) -> dict[str, Any]:
        decision = self.gate.evaluate(dynamics, risk=risk)
        if not decision["open"]:
            return {
                "channel_open": False,
                "gate": decision,
                "hypotheses": [],
                "namespace": HYPOTHESIS_NAMESPACE,
            }

        ordered = sorted(candidates, key=lambda item: float(item.novelty_score), reverse=True)
        selected = [envelope.as_exclusion_hypothesis() for envelope in ordered[: self.max_hypotheses]]
        return {
            "channel_open": True,
            "gate": decision,
            "hypotheses": selected,
            "namespace": HYPOTHESIS_NAMESPACE,
        }


def assert_not_evidence(items: Sequence[dict[str, Any]]) -> None:
    """Guard used at the evidence boundary.

    Raises when a synthetic hypothesis reaches a path reserved for evidence. The
    contamination this prevents is self-reinforcing: a candidate admitted as
    evidence becomes retrievable in later cases, and the system starts citing its
    own generated material with formally intact provenance.
    """
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        if item.get("role") == HYPOTHESIS_ROLE or item.get("synthetic_candidate_not_clinical_truth") is True:
            raise ValueError(
                f"item {index} is a synthetic exclusion hypothesis and cannot be used as evidence"
            )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
