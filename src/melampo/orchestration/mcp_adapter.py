from dataclasses import dataclass


@dataclass
class MCPAdapter:
    endpoint: str = "api_for_service_mcp_server"

    def describe(self) -> dict:
        return {"endpoint": self.endpoint, "protocol": "mcp"}
