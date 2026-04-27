from melampo.orchestration.bootstrap import RegistryBootstrap
from melampo.orchestration.contracts import ServiceContract
from melampo.orchestration.service_registry import ServiceRegistry


def test_registry_bootstrap_exposes_contract_inventory_and_summary():
    contract = ServiceContract("volume_encoder", "api_for_service_volume_encoder", "service", "encoder")
    described = contract.describe()
    assert described["name"] == "volume_encoder"
    assert described["role"] == "encoder"

    registry = ServiceRegistry()
    registry.register("volume_encoder", "api_for_service_volume_encoder", "service", "encoder")
    assert registry.get("volume_encoder")["role"] == "encoder"
    summary = registry.describe()
    assert summary["service_count"] == 1
    assert summary["service_names"] == ["volume_encoder"]

    bootstrap = RegistryBootstrap()
    contracts = bootstrap.contracts()
    assert contracts
    built = bootstrap.build()
    built_summary = built.describe()
    assert built_summary["service_count"] >= 3
    assert "volume_encoder" in built_summary["service_names"]
