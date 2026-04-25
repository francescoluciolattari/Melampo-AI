from melampo.evaluation.quantum_gate import QuantumResearchGate
from melampo.orchestration.runtime_services import RuntimeServices


class _Config:
    pass


class _Logger:
    def info(self, *args, **kwargs):
        return None


def test_runtime_services_and_quantum_gate_expose_structured_resolution_metadata():
    gate = QuantumResearchGate(min_contextuality_score=0.6).assess(0.75)
    assert gate["allow"] is True
    assert gate["level"] in ["high", "guarded", "low"]
    assert gate["reasons"]

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
