from melampo.orchestration.bootstrap import RegistryBootstrap


def test_registry_bootstrap_contains_core_services():
    registry = RegistryBootstrap().build()
    assert registry.get("volume_encoder")["provider"] == "api_for_service_volume_encoder"
    assert registry.get("mcp_server")["protocol"] == "mcp"
