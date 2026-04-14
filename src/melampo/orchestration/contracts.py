from dataclasses import dataclass


@dataclass
class ServiceContract:
    name: str
    provider: str
    protocol: str

    def describe(self) -> dict:
        return {"name": self.name, "provider": self.provider, "protocol": self.protocol}
