from dataclasses import dataclass


@dataclass
class A2AAdapter:
    endpoint: str = "api_for_service_a2a_router"

    def describe(self) -> dict:
        return {"endpoint": self.endpoint, "protocol": "a2a"}
