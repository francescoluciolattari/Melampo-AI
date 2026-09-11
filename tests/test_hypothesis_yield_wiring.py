"""Tests for wiring HypothesisYieldModel to the confirmation registry and exposing it as a tool."""

from melampo.governance.confirmation_registry import (
    SOURCE_HISTOPATHOLOGY,
    SOURCE_SYSTEM_ACCEPTED,
    Confirmation,
    ConfirmationRegistry,
)
from melampo.training.hypothesis_yield import HypothesisFeatures, HypothesisYieldModel
from melampo.training.hypothesis_yield_wiring import (
    YIELD_RATE_TOOL_SPEC,
    sync_from_registry,
    yield_rate_tool,
)


def _surfaced(case_id: str, condition: str = "pericarditis", **feature_kwargs) -> dict:
    defaults = {"hops": 2, "support": 0.7, "corroboration": 2}
    defaults.update(feature_kwargs)
    return {"case_id": case_id, "condition": condition, "features": HypothesisFeatures(**defaults)}


# --------------------------------------------------------------------------
# Syncing: only independent, admitted confirmations feed the model
# --------------------------------------------------------------------------


def test_an_independent_confirmation_is_observed():
    registry = ConfirmationRegistry()
    registry.register(Confirmation(case_id="c1", diagnosis="pericarditis", source=SOURCE_HISTOPATHOLOGY))
    model = HypothesisYieldModel()

    result = sync_from_registry(model, registry, [_surfaced("c1")])

    assert result.new_observations == 1
    assert result.total_observations == 1


def test_a_non_independent_confirmation_is_not_observed():
    """This is the exact automation-bias guard confirmation_registry.py exists
    for: a system-accepted suggestion must not silently teach the model that
    its own guesses were right."""
    registry = ConfirmationRegistry()
    registry.register(Confirmation(case_id="c1", diagnosis="pericarditis", source=SOURCE_SYSTEM_ACCEPTED))
    model = HypothesisYieldModel()

    result = sync_from_registry(model, registry, [_surfaced("c1")])

    assert result.new_observations == 0
    assert result.learning_set_size == 0


def test_a_surfaced_hypothesis_with_no_confirmation_is_silently_skipped():
    """Absence of a confirmation is not evidence the hypothesis was wrong --
    it must not be counted as a miss."""
    registry = ConfirmationRegistry()
    model = HypothesisYieldModel()

    result = sync_from_registry(model, registry, [_surfaced("c1")])

    assert result.new_observations == 0
    assert result.unmatched_confirmations == 0


def test_a_confirmation_with_no_matching_surfaced_hypothesis_is_reported_unmatched():
    registry = ConfirmationRegistry()
    registry.register(Confirmation(case_id="c1", diagnosis="pericarditis", source=SOURCE_HISTOPATHOLOGY))
    model = HypothesisYieldModel()

    result = sync_from_registry(model, registry, [])

    assert result.unmatched_confirmations == 1


def test_sync_result_reports_growing_totals_across_calls():
    registry = ConfirmationRegistry()
    registry.register(Confirmation(case_id="c1", diagnosis="pericarditis", source=SOURCE_HISTOPATHOLOGY))
    model = HypothesisYieldModel()
    sync_from_registry(model, registry, [_surfaced("c1")])

    registry.register(Confirmation(case_id="c2", diagnosis="pericarditis", source=SOURCE_HISTOPATHOLOGY))
    result = sync_from_registry(model, registry, [_surfaced("c1"), _surfaced("c2")])

    assert result.total_observations == 3, "c1 observed twice (once per call) plus c2 once"


# --------------------------------------------------------------------------
# The tool: data, not weights -- must work identically regardless of which
# model calls it
# --------------------------------------------------------------------------


def test_the_tool_returns_the_same_estimate_the_model_itself_would():
    model = HypothesisYieldModel()
    registry = ConfirmationRegistry()
    registry.register(Confirmation(case_id="c1", diagnosis="pericarditis", source=SOURCE_HISTOPATHOLOGY))
    sync_from_registry(model, registry, [_surfaced("c1")])

    tool = yield_rate_tool(model)
    via_tool = tool(hops=2, support=0.7, corroboration=2)
    via_direct = model.estimate(HypothesisFeatures(hops=2, support=0.7, corroboration=2)).as_dict()

    assert via_tool == via_direct


def test_the_tool_is_a_plain_function_not_a_class():
    """Deliberately a bare callable so it drops into any calling convention --
    direct call, dict-dispatch router, MCP handler registration -- without
    this module depending on any of them."""
    tool = yield_rate_tool(HypothesisYieldModel())
    assert callable(tool)
    assert not isinstance(tool, type)


def test_the_tool_defaults_corroboration_and_gap_count():
    tool = yield_rate_tool(HypothesisYieldModel())
    result = tool(hops=1, support=0.9)
    assert result["bucket"] == "direct|strong|single|attested"


def test_an_unobserved_shape_returns_the_full_interval_via_the_tool():
    """No data is not suppression: an unobserved shape must not read as
    unreliable."""
    tool = yield_rate_tool(HypothesisYieldModel())
    result = tool(hops=5, support=0.1)
    assert result["lower"] == 0.0
    assert result["upper"] == 1.0
    assert result["established"] is False


def test_the_tool_spec_names_match_the_callables_parameters():
    """The spec is what an MCP server or function-calling schema would
    register; it must actually describe the callable it accompanies."""
    import inspect

    tool = yield_rate_tool(HypothesisYieldModel())
    spec_params = set(YIELD_RATE_TOOL_SPEC["parameters"]["properties"])
    actual_params = set(inspect.signature(tool).parameters)
    assert spec_params == actual_params


def test_the_tool_spec_required_fields_have_no_default_in_the_callable():
    import inspect

    tool = yield_rate_tool(HypothesisYieldModel())
    sig = inspect.signature(tool)
    for name in YIELD_RATE_TOOL_SPEC["parameters"]["required"]:
        assert sig.parameters[name].default is inspect.Parameter.empty
