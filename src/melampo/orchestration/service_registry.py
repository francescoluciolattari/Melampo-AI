from dataclasses import dataclass, field


@dataclass
class ServiceRegistry:
    """In-memory registry for provider-neutral service declarations."""

    services: dict = field(default_factory=dict)

    def register(self, name: str, provider: str, protocol: str, role: str = "core_service") -> None:
        self.services[name] = {"provider": provider, "protocol": protocol, "role": role}

    def get(self, name: str) -> dict:
        return self.services.get(name, {})

    def describe(self) -> dict:
        return {
            "service_count": len(self.services),
            "service_names": sorted(self.services.keys()),
        }
