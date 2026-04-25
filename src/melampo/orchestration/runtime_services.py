from dataclasses import dataclass

from .bootstrap import RegistryBootstrap
from .model_router import ModelRouter


@dataclass
class RuntimeServices:
    """Assemble routing and service registry for runtime use."""

    router: ModelRouter
    registry: object

    @classmethod
    def build(cls, config: object, logger: object):
        registry = RegistryBootstrap().build()
        router = ModelRouter(config=config, logger=logger)
        return cls(router=router, registry=registry)

    def resolve(self, task_name: str) -> dict:
        route = self.router.pick(task_name)
        service = self.registry.get(task_name)
        available = bool(service)
        return {
            "task": task_name,
            "route": route,
            "service": service,
            "available": available,
            "protocol": service.get("protocol", route.get("protocol_hint", "service")) if isinstance(service, dict) else route.get("protocol_hint", "service"),
            "resolution_mode": "direct_registry_match" if available else "router_only",
        }
