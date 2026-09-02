"""Registry of experimental claims and the criteria that would refute them.

A decision record mixes claims of different epistemic types. Architectural
boundaries are constraints: they hold because the project chose them, and no
measurement can refute them. Sequencing is a plan: revisable on new information,
binding until then. But a decision record also carries empirical predictions —
that one retrieval strategy outperforms another, that a signal is informative —
and those are not settled by being written down.

Marking a prediction "accepted" is the failure this module exists to prevent.
Predictions belong here, each paired with the observation that would refute it,
so the release gate can distinguish what has been measured from what has been
assumed.
"""

from dataclasses import dataclass, field
from typing import Any, Final

CLAIM_OPEN = "open"
CLAIM_CORROBORATED = "corroborated"
CLAIM_REFUTED = "refuted"
CLAIM_WITHDRAWN = "withdrawn"


@dataclass
class FalsifiableClaim:
    """An empirical claim paired with an explicit refutation criterion.

    A claim without a refutation criterion is not a claim, it is a preference.
    ``refutation_criterion`` must describe an observation that would settle the
    matter against the claim.
    """

    claim_id: str
    statement: str
    refutation_criterion: str
    decision_record: str | None = None
    status: str = CLAIM_OPEN
    blocking: bool = False
    evidence: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "refutation_criterion": self.refutation_criterion,
            "decision_record": self.decision_record,
            "status": self.status,
            "blocking": self.blocking,
            "evidence": list(self.evidence),
        }


def _default_claims() -> list[FalsifiableClaim]:
    """Empirical claims carried by the RLM-on-Memory decision record.

    None of these is settled by the decision to implement the architecture. Each
    is stated with the observation that would refute it.
    """
    record = "docs/rlm_on_memory_decision_record.md"
    return [
        FalsifiableClaim(
            claim_id="rlm.dual_path_beats_single_path",
            statement=(
                "Running one-shot and recursive retrieval in parallel and reconciling them yields better "
                "grounded evidence than either path alone on complex cases."
            ),
            refutation_criterion=(
                "Dual-path faithfulness falls below the one-shot baseline on the same case set, or its recall "
                "gain is not distinguishable from the one-shot path."
            ),
            decision_record=record,
            blocking=True,
        ),
        FalsifiableClaim(
            claim_id="rlm.disagreement_is_informative",
            statement=(
                "Divergence between the two retrieval paths is an informative uncertainty signal, because the "
                "paths fail in opposite directions: one-shot by omission, recursive by overreach."
            ),
            refutation_criterion=(
                "Path divergence shows no association with independently adjudicated case difficulty, or the two "
                "paths are observed to fail on the same items rather than complementarily."
            ),
            decision_record=record,
            blocking=True,
        ),
        FalsifiableClaim(
            claim_id="rlm.coverage_predicts_grounding",
            statement=(
                "Measured corpus coverage predicts grounding quality, making low coverage with high confidence a "
                "detectable failure mode."
            ),
            refutation_criterion=(
                "Coverage shows no relationship with adjudicated grounding quality across the validation set."
            ),
            decision_record=record,
        ),
        FalsifiableClaim(
            claim_id="rlm.recursive_helps_only_on_complex_cases",
            statement=(
                "Recursive retrieval improves outcomes on multi-document longitudinal cases and degrades them on "
                "simple factual lookups, justifying a complexity gate rather than uniform adoption."
            ),
            refutation_criterion=(
                "Recursive retrieval matches or exceeds one-shot performance on simple lookups, removing the "
                "rationale for the gate."
            ),
            decision_record=record,
        ),
        FalsifiableClaim(
            claim_id="rlm.dream_hypotheses_add_value",
            statement=(
                "Grounded enumeration of corpus variants produces exclusion hypotheses that improve the "
                "differential under high diagnostic indeterminacy."
            ),
            refutation_criterion=(
                "Hypotheses admitted through the channel are judged clinically relevant no more often than "
                "chance, or their presence does not change the differential under review."
            ),
            decision_record=record,
        ),
        FalsifiableClaim(
            claim_id="rlm.open_weight_root_is_sufficient",
            statement=(
                "A self-hosted open-weight root model is sufficient for the recursive loop at depth 1, so the "
                "on-premise transition does not materially degrade the API baseline."
            ),
            refutation_criterion=(
                "The self-hosted root model loses more than the pre-registered tolerance against the API baseline "
                "on the same case set."
            ),
            decision_record=record,
            blocking=True,
        ),
    ]


LEGACY_CRITERIA: Final[dict[str, str]] = {
    "quantum_like": "requires comparative benchmark",
    "open_systems": "requires replicated gains",
    "biophysical_frontier": "requires independent laboratory evidence",
}


@dataclass
class FalsificationProgram:
    """Track experimental claims and minimum falsification criteria."""

    claims: list[FalsifiableClaim] = field(default_factory=_default_claims)

    LEGACY_CRITERIA = LEGACY_CRITERIA

    def summarize(self) -> dict:
        payload = dict(self.LEGACY_CRITERIA)
        for claim in self.claims:
            payload[claim.claim_id] = claim.refutation_criterion
        return payload

    def register(self, claim: FalsifiableClaim) -> None:
        if any(existing.claim_id == claim.claim_id for existing in self.claims):
            raise ValueError(f"claim already registered: {claim.claim_id}")
        self.claims.append(claim)

    def get(self, claim_id: str) -> FalsifiableClaim:
        for claim in self.claims:
            if claim.claim_id == claim_id:
                return claim
        raise KeyError(f"unknown claim: {claim_id}")

    def resolve(self, claim_id: str, status: str, evidence: str) -> FalsifiableClaim:
        """Record an outcome against a claim. Evidence is mandatory."""
        if status not in {CLAIM_CORROBORATED, CLAIM_REFUTED, CLAIM_WITHDRAWN}:
            raise ValueError(f"invalid resolution status: {status}")
        if not evidence:
            raise ValueError("resolving a claim requires evidence")
        claim = self.get(claim_id)
        claim.status = status
        claim.evidence.append(evidence)
        return claim

    def open_claims(self) -> list[FalsifiableClaim]:
        return [claim for claim in self.claims if claim.status == CLAIM_OPEN]

    def blocking_claims(self) -> list[FalsifiableClaim]:
        """Open claims that must be resolved before the strategy leaves research use."""
        return [claim for claim in self.open_claims() if claim.blocking]

    def refuted_claims(self) -> list[FalsifiableClaim]:
        return [claim for claim in self.claims if claim.status == CLAIM_REFUTED]

    def report(self) -> dict[str, Any]:
        return {
            "total": len(self.claims),
            "open": len(self.open_claims()),
            "blocking_open": len(self.blocking_claims()),
            "refuted": len(self.refuted_claims()),
            "claims": [claim.as_dict() for claim in self.claims],
        }
