"""Tests for TaskProtocolRouter: the task-name to transport-protocol
routing split out of what used to be ModelRouter, before ModelRouter was
rebuilt into D1 (clinical complexity routing). This is the original
routing logic, unchanged, just under its own name and used only by
RuntimeServices.resolve().
"""

from melampo.orchestration.task_protocol_router import TaskProtocolRouter


def _router():
    return TaskProtocolRouter(config=object(), logger=object())


def test_a_task_name_containing_router_gets_the_a2a_protocol():
    result = _router().pick("volume_encoder_router")
    assert result["protocol_hint"] == "a2a"


def test_a_task_name_containing_mcp_gets_the_mcp_protocol():
    result = _router().pick("mcp_tool_call")
    assert result["protocol_hint"] == "mcp"


def test_any_other_task_name_gets_the_service_protocol():
    result = _router().pick("volume_encoder")
    assert result["protocol_hint"] == "service"


def test_the_task_name_is_echoed_back():
    result = _router().pick("volume_encoder")
    assert result["task"] == "volume_encoder"
