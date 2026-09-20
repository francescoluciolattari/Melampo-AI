"""Task-name to transport-protocol routing, used by RuntimeServices.resolve().

Split out of what used to be ModelRouter, which this file's routing logic
occupied entirely before ModelRouter was rebuilt into D1 (clinical
complexity routing, orchestration/model_router.py) -- genuinely two
different concerns that happened to share one class and one name.
RuntimeServices.resolve() needs a transport protocol hint for a task
name (a2a/mcp/service), which has nothing to do with how many clinical
reasoning paths a case needs; conflating them under one class would have
made D1 harder to reason about for no benefit to either concern.
"""

from dataclasses import dataclass


@dataclass
class TaskProtocolRouter:
    config: object
    logger: object

    def pick(self, task_name: str) -> dict:
        protocol_hint = "service"
        if "router" in task_name:
            protocol_hint = "a2a"
        elif "mcp" in task_name:
            protocol_hint = "mcp"
        return {
            "task": task_name,
            "router": "api_for_service_a2a_router",
            "protocol_hint": protocol_hint,
            "routing_mode": "static_research_router",
        }
