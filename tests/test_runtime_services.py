from melampo.config import build_default_config
from melampo.orchestration.runtime_services import RuntimeServices
from melampo.utils.logging_utils import build_logger


def test_runtime_services_resolve_known_task():
    services = RuntimeServices.build(
        config=build_default_config(), logger=build_logger("test")
    )
    result = services.resolve("volume_encoder")
    assert result["service"]["provider"] == "api_for_service_volume_encoder"


class _Config:
    pass


class _Logger:
    def info(self, *args, **kwargs):
        return None


def test_runtime_services_expose_structured_resolution_metadata():
    runtime = RuntimeServices.build(config=_Config(), logger=_Logger())
    resolved = runtime.resolve("volume_encoder")
    assert resolved["task"] == "volume_encoder"
    assert resolved["available"] is True
    assert resolved["resolution_mode"] == "direct_registry_match"
    assert resolved["protocol"] in ["service", "mcp", "a2a"]
    assert resolved["route"]["routing_mode"] == "static_research_router"

    unresolved = runtime.resolve("unknown_task")
    assert unresolved["available"] is False
    assert unresolved["resolution_mode"] == "router_only"
