"""Wire HypothesisYieldModel to the confirmation registry, and expose it as a query tool.

Closes a gap found by checking the code rather than recalling a summary:
`HypothesisYieldModel` (training/hypothesis_yield.py) and `ConfirmationRegistry`
(governance/confirmation_registry.py) both existed, both well-designed, both
never connected to each other and never instantiated anywhere production code
would reach. The same shape of gap found five times earlier in this project
(Muse Glimmer, root_model_cross_check, mechanism_verification, the dream
trainer's enumerator hook, the conjecture ledger) — a module built, tested,
and never wired to a caller.

**What this module deliberately does not do.** `HypothesisYieldModel` learns
bucketed empirical rates, not weights -- "not a fitted network", by its own
docstring, and for a stated reason: an inspectable rate over counted outcomes,
not an opaque function. Because it is data rather than weights, it does not
need weight-level integration with whatever model does the vetting. Whether
that model is a closed API (Claude, Gemini) or an open one hosted on-premise
(GPT-OSS-120B, Nemotron) makes no difference here: neither can have weights
merged into it over an API, and neither needs to -- both can call a tool.
`yield_rate_tool()` below is that tool: a plain, inspectable function plus a
tool-spec dict in the same shape function-calling and MCP tool definitions
use, so it can be exposed to any calling model through whatever mechanism that
model's integration already uses, without this module taking a position on
which protocol carries it.

The question of whether to eventually fine-tune a model's actual weights on
confirmed outcomes is real and distinct from this -- it would need an
open-weight, on-premise-hosted candidate (the E4 roadmap item), and it is not
what this module does.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..governance.confirmation_registry import ConfirmationRegistry
from .hypothesis_yield import (
    HypothesisFeatures,
    HypothesisYieldModel,
    outcomes_from_confirmations,
)


@dataclass
class SyncResult:
    """What happened when the yield model was synced against the registry."""

    new_observations: int
    total_observations: int
    learning_set_size: int
    unmatched_confirmations: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "new_observations": self.new_observations,
            "total_observations": self.total_observations,
            "learning_set_size": self.learning_set_size,
            "unmatched_confirmations": self.unmatched_confirmations,
        }


def sync_from_registry(
    model: HypothesisYieldModel,
    registry: ConfirmationRegistry,
    surfaced: Sequence[dict[str, Any]],
) -> SyncResult:
    """Pull the registry's admitted confirmations into the yield model.

    ``surfaced`` is the record of hypotheses actually shown to a clinician for
    each case -- ``{"case_id": ..., "condition": ..., "features": HypothesisFeatures(...)}``,
    the shape ``outcomes_from_confirmations`` already expects. Only entries
    whose case has an admitted, independent confirmation produce an outcome;
    everything else is silently not observed rather than counted as a miss,
    for the reason ``outcomes_from_confirmations`` states directly: absence of
    a confirmation is not evidence the hypothesis was wrong.

    Idempotent in effect but not in bookkeeping: calling this again with the
    same registry state re-adds the same outcomes, since ``HypothesisYieldModel``
    keeps no identity on its own list. Call once per new registry state, not
    on a timer -- ``new_observations`` is returned specifically so a caller can
    tell whether the call actually added anything.
    """
    confirmed_case_ids = {item.case_id for item in registry.learning_set()}
    outcomes = outcomes_from_confirmations(surfaced, registry.learning_set())
    added = model.observe_many(outcomes)
    matched_case_ids = {outcome.case_id for outcome in outcomes}
    unmatched = len(confirmed_case_ids - matched_case_ids)
    return SyncResult(
        new_observations=added,
        total_observations=len(model.outcomes),
        learning_set_size=len(registry.learning_set()),
        unmatched_confirmations=unmatched,
    )


# --------------------------------------------------------------------------
# Exposing the model as a tool a diagnostic model can call
# --------------------------------------------------------------------------

YIELD_RATE_TOOL_NAME = "yield_rate"

YIELD_RATE_TOOL_SPEC: dict[str, Any] = {
    "name": YIELD_RATE_TOOL_NAME,
    "description": (
        "Look up how often a hypothesis of this shape has been independently confirmed in the "
        "past. Returns a confidence interval, not a point estimate -- a shape with few observations "
        "reports a wide interval rather than a false-precise rate. An unobserved shape returns the "
        "full [0, 1] interval: absence of data, not evidence the shape is unreliable."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "hops": {"type": "integer", "description": "Length of the supporting graph path."},
            "support": {
                "type": "number",
                "description": "Point strength of the supporting path, 0.0 to 1.0.",
            },
            "corroboration": {
                "type": "integer",
                "description": "How many independent findings support this hypothesis. Default 1.",
                "default": 1,
            },
            "gap_count": {
                "type": "integer",
                "description": "How many unknown-strength edges the supporting path crosses. Default 0.",
                "default": 0,
            },
        },
        "required": ["hops", "support"],
    },
}


def yield_rate_tool(model: HypothesisYieldModel) -> Any:
    """Bind a HypothesisYieldModel into a callable matching YIELD_RATE_TOOL_SPEC's parameters.

    Returns a plain function, not a class, so it drops into whatever calling
    convention is already in use -- direct Python call, a dict-dispatch tool
    router, an MCP server's handler registration -- without this module
    depending on any of them.
    """

    def _call(hops: int, support: float, corroboration: int = 1, gap_count: int = 0) -> dict[str, Any]:
        features = HypothesisFeatures(
            hops=int(hops), support=float(support), corroboration=int(corroboration), gap_count=int(gap_count)
        )
        return model.estimate(features).as_dict()

    return _call
