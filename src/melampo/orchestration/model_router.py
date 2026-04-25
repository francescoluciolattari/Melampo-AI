from dataclasses import dataclass


@dataclass
class ModelRouter:
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
