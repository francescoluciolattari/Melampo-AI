from dataclasses import dataclass


@dataclass
class EvidenceRanker:
    """Baseline evidence ranker producing simple grounded support ordering."""

    def rank(self, items: list) -> list:
        ranked = []
        for index, item in enumerate(items):
            ranked.append({"rank": index + 1, "item": item, "weight": max(len(items) - index, 1)})
        return ranked
