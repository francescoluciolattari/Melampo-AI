from dataclasses import dataclass, field


@dataclass
class ServiceRegistry:
    """In-memory registry for provider-neutral service declarations."""

    services: dict = field(default_factory=dict)

    def register(self, name: str, provider: str, protocol: str) -> None:
        self.services[name] = {"provider": provider, "protocol": protocol}

    def get(self, name: str) -> dict:
        return self.services.get(name, {})
