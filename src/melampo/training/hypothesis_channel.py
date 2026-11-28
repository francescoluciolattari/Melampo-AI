"""Isolated channel delivering nexus candidates as exclusion hypotheses.

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
information; they earn their place only when the diagnosis is unclear. Who says
it is unclear is the caller (the lateral cycle of the intuition engine): the gate
on the coherence between areas that decided it until 28 November 2026 was
removed with that measure, never measured.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

HYPOTHESIS_NAMESPACE = "melampo_hypothesis_candidates"
HYPOTHESIS_ROLE = "exclusion_hypothesis"


@dataclass
class HypothesisEnvelope:
    """A nexus candidate wrapped for consumption by the differential engine only."""

    label: str
    rationale: str = ""
    origin: str = "nexus_branch"
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
            "source_type": "synthetic_nexus_candidate",
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

    max_hypotheses: int = 3

    def open_for(
        self,
        candidates: Iterable[HypothesisEnvelope],
        *,
        unclear: bool,
        reasons: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Open the channel when the caller says the diagnosis is unclear, with its reasons."""
        decision = {"open": bool(unclear), "reasons": list(reasons)}
        if not decision["open"]:
            return {
                "channel_open": False,
                "gate": decision,
                "hypotheses": [],
                "namespace": HYPOTHESIS_NAMESPACE,
            }

        ordered = sorted(
            candidates, key=lambda item: float(item.novelty_score), reverse=True
        )
        selected = [
            envelope.as_exclusion_hypothesis()
            for envelope in ordered[: self.max_hypotheses]
        ]
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
        if (
            item.get("role") == HYPOTHESIS_ROLE
            or item.get("synthetic_candidate_not_clinical_truth") is True
        ):
            raise ValueError(
                f"item {index} is a synthetic exclusion hypothesis and cannot be used as evidence"
            )


