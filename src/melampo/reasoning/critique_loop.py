from dataclasses import dataclass


@dataclass
class CritiqueLoop:
    config: object
    logger: object

    def review(self, draft: dict) -> dict:
        return {"status": "reviewed", "draft": draft}
