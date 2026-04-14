from dataclasses import dataclass, field


@dataclass
class PrototypeHead:
    prototypes: list = field(default_factory=list)

    def classify(self, embedding: object) -> dict:
        return {"status": "prototype_placeholder", "prototype_count": len(self.prototypes)}
