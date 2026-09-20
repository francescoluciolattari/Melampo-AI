"""D1: decides how many reasoning paths a case needs, and whether it is new or already pending.

Replaces a 12-line stub that routed on a task NAME string
("router"/"mcp"/anything else) to a transport protocol hint -- unrelated
to clinical complexity, never called by anything, verified before this
rewrite (grep found zero callers). This is the router the September RLM
decision record specifies: it decides *how many* reasoning paths run, not
which one, per the table reproduced below (docs/rlm_on_memory_decision_record.md,
"Authority boundary", CONSTRAINT, accepted):

| Case profile              | Mode                                     |
|----------------------------|-------------------------------------------|
| Factual lookup, low risk   | One-shot only                             |
| Complex or high risk       | Dual path with reconciliation             |
| Unresolved area mismatch   | Dual path, extended recursive budget      |
| Nexus branch (low activity)| Recursive only; latency is not binding    |

**Two things this router does NOT yet do, stated plainly rather than
silently attempted.** First: "dual path" here means routing through
RlmEngine's recursive navigation into DifferentialEngine
(via rlm_graph_bridge.py) alongside IntuitionEngine's one-shot path --
that connection does not exist in clinical_pipeline.py today (verified:
RlmEngine is never called from there), so a "dual_path" verdict is
reported, not yet actable on -- the same "recognised but not executed"
posture already used for pending_case_router.py's confirm_and_train
branch, for the same reason: building the missing connection is separate,
larger work, not something to improvise inside this router. Second: the
Nexus-branch row describes NexusTrainer's own background rehearsal
context, which already runs unconditionally per case regardless of this
router's verdict (see nexus_trainer.py) -- this router's decision applies
to the live, per-case reasoning path, not that background branch.

**The pending-case check runs first, ahead of complexity routing**, per
the design worked through directly: before deciding how many paths a
case needs, the router must know whether this payload is a brand new
case, more data for one already at needs_review, or a confirmation for
one -- a different question with a different answer, decided by
pending_case_router.route_payload() before complexity is even assessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..training.nexus_candidate_store import NexusCandidateStore
from ..training.pending_case_router import RoutingDecision, route_payload

# Thresholds, explicit and justified rather than tuned to a hidden
# dataset -- there is none to tune against yet. Each one names the exact
# signal it reads and why that value, not a number chosen to "feel right":
# review these against real outcomes once any exist (see
# recursive_engine_decision_record.md on why every neuro-dynamic metric
# in this project needs the same scrutiny before being trusted).
LOW_RISK_THRESHOLD = 0.3
HIGH_RISK_THRESHOLD = 0.5
FEW_FINDINGS_THRESHOLD = 2
UNRESOLVED_MISMATCH_THRESHOLD = 0.5

MODE_ONE_SHOT = "one_shot"
MODE_DUAL_PATH = "dual_path"
MODE_DUAL_PATH_EXTENDED_BUDGET = "dual_path_extended_budget"


@dataclass
class ModelRouter:
    """D1. Two entry points, called at two different points in clinical_pipeline.run() -- see pick_pending_case and pick_mode."""

    candidate_store: NexusCandidateStore

    def pick_pending_case(self, payload: dict[str, Any]) -> RoutingDecision:
        """Whether this payload is new, more data for a pending case, or a confirmation -- callable early.

        Split from mode selection deliberately: this needs nothing but the
        payload itself and the candidate store, so it can run at the very
        start of clinical_pipeline.run(), before report_text is resolved
        for the rest of the pipeline to use -- area_dynamics and
        governance_scores, which pick_mode() needs, do not exist yet at
        that point.
        """
        return route_payload(payload, self.candidate_store)

    def pick_mode(
        self,
        *,
        findings: list[str] | None = None,
        area_dynamics: dict[str, Any] | None = None,
        governance_scores: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """How many reasoning paths this case needs, and why -- callable once real signals exist, later in run()."""
        findings = findings or []
        area_dynamics = area_dynamics or {}
        governance_scores = governance_scores or {}
        risk = float(governance_scores.get("risk", 0.0))
        mismatch_score = float(area_dynamics.get("mismatch_score", 0.0))

        if mismatch_score >= UNRESOLVED_MISMATCH_THRESHOLD:
            return MODE_DUAL_PATH_EXTENDED_BUDGET, f"unresolved area mismatch ({mismatch_score:.2f} >= {UNRESOLVED_MISMATCH_THRESHOLD})"
        if risk >= HIGH_RISK_THRESHOLD or len(findings) > FEW_FINDINGS_THRESHOLD:
            reason = f"risk {risk:.2f} >= {HIGH_RISK_THRESHOLD}" if risk >= HIGH_RISK_THRESHOLD else f"{len(findings)} findings > {FEW_FINDINGS_THRESHOLD}"
            return MODE_DUAL_PATH, reason
        if risk <= LOW_RISK_THRESHOLD and len(findings) <= FEW_FINDINGS_THRESHOLD:
            return MODE_ONE_SHOT, f"low risk ({risk:.2f}) and few findings ({len(findings)})"
        # Between the low- and high-risk bands with a moderate finding
        # count: the table names only the two extremes plus the mismatch
        # case, not this middle ground -- dual_path is the safer default
        # of the two live-path modes when neither extreme clearly applies,
        # not a silent guess dressed up as a rule.
        return MODE_DUAL_PATH, f"neither clearly low-risk/few-findings nor high-risk/many-findings (risk={risk:.2f}, findings={len(findings)})"
