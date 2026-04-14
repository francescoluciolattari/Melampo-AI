from dataclasses import dataclass


@dataclass
class ModelRouter:
    config: object
    logger: object

    def pick(self, task_name: str) -> dict:
        return {"task": task_name, "router": "api_for_service_a2a_router"}
