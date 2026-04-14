from dataclasses import dataclass


@dataclass
class EvidenceRanker:
    """Placeholder evidence ranker for grounded clinical support."""

    def rank(self, items: list) -> list:
        return list(items)
