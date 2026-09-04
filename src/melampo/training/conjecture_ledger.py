"""Conjectures: intuitive leaps that become knowledge when they prove true.

The dream branch makes leaps — connections between two concepts that no edge
states directly, reached through a shared mechanism or a longer chain. Expertise
research describes this as what clinical intuition actually is: not a random
association but pattern recognition over encapsulated knowledge, a link whose
halves are each documented while the whole is not. Unorthodox, not false.

Until now such a leap lived and died within one case. This module lets a leap
that proves true become knowledge the graph did not have: a new edge, carrying
the cases that confirmed it. The graph learns intuitions.

It does so under the same asymmetry that governs every other addition to the
knowledge base. A conjecture is **recorded freely** — every leap the branch
makes is a candidate. It is **promoted only when independently confirmed**,
enough times, by evidence produced outside the system's own reasoning. And the
promoted edge carries an interval computed from those confirmations, so a
connection confirmed three times is visibly less certain than one confirmed
thirty times, and a connection that keeps failing is never promoted at all.

A conjecture is never traversable before promotion. If it were, the branch could
reach the next leap through the previous one, and the chain of unverified
connections would grow with no evidence entering anywhere.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..memory.concept_paths import ConceptEdge
from ..memory.ontology_import import wilson_interval

RELATION_CONJECTURED = "conjectured_association"
MIN_CONFIRMATIONS = 3


@dataclass(frozen=True)
class Conjecture:
    """One leap: a connection the branch reached but no edge states."""

    source: str
    target: str
    via: tuple[str, ...]
    hops: int
    strength_upper: float
    origin_case: str

    @property
    def key(self) -> tuple[str, str]:
        return (_normalise(self.source), _normalise(self.target))

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "via": list(self.via),
            "hops": self.hops,
            "strength_upper": round(self.strength_upper, 3),
            "origin_case": self.origin_case,
        }


@dataclass
class ConjectureRecord:
    """A conjecture across the cases it has appeared in and the ones that tested it."""

    conjecture: Conjecture
    raised_in: list[str] = field(default_factory=list)
    confirmed_in: list[str] = field(default_factory=list)
    refuted_in: list[str] = field(default_factory=list)

    @property
    def tested(self) -> int:
        return len(self.confirmed_in) + len(self.refuted_in)

    def interval(self) -> tuple[float, float]:
        """How often the leap held when tested. Full interval until tested."""
        if not self.tested:
            return (0.0, 1.0)
        return wilson_interval(len(self.confirmed_in), self.tested)

    def is_promotable(self, min_confirmations: int) -> bool:
        lower, _ = self.interval()
        return len(self.confirmed_in) >= min_confirmations and lower > 0.0

    def to_edge(self) -> ConceptEdge:
        """The promoted edge, carrying the confirming cases as provenance."""
        lower, upper = self.interval()
        return ConceptEdge(
            source=self.conjecture.source,
            relation=RELATION_CONJECTURED,
            target=self.conjecture.target,
            weight=(lower + upper) / 2.0,
            provenance=(
                f"conjecture:via={'>'.join(self.conjecture.via) or 'direct'}"
                f";confirmed={','.join(self.confirmed_in)}"
                f";refuted={','.join(self.refuted_in) or 'none'}"
            ),
            lower=lower,
            upper=upper,
        )

    def as_dict(self) -> dict[str, Any]:
        lower, upper = self.interval()
        return {
            **self.conjecture.as_dict(),
            "raised_in": list(self.raised_in),
            "confirmed_in": list(self.confirmed_in),
            "refuted_in": list(self.refuted_in),
            "lower": round(lower, 3),
            "upper": round(upper, 3),
        }


@dataclass
class ConjectureLedger:
    """Record leaps, test them against independent confirmations, promote survivors."""

    records: dict[tuple[str, str], ConjectureRecord] = field(default_factory=dict)
    min_confirmations: int = MIN_CONFIRMATIONS

    def record(self, conjecture: Conjecture) -> ConjectureRecord:
        """Record a leap. Free: every leap the branch makes is a candidate."""
        entry = self.records.get(conjecture.key)
        if entry is None:
            entry = ConjectureRecord(conjecture=conjecture)
            self.records[conjecture.key] = entry
        if conjecture.origin_case not in entry.raised_in:
            entry.raised_in.append(conjecture.origin_case)
        return entry

    def record_from_hypothesis(self, hypothesis: Any, case_id: str) -> list[ConjectureRecord]:
        """Record the leaps a mechanism hypothesis embodies, one per linked finding."""
        entries: list[ConjectureRecord] = []
        condition = str(getattr(hypothesis, "condition", "")).strip()
        paths = list(getattr(hypothesis, "paths", ()) or ())
        for path in paths:
            edges = getattr(path, "edges", ())
            if not edges or not condition:
                continue
            source = str(edges[0].source)
            if _normalise(source) == _normalise(condition):
                continue
            entries.append(
                self.record(
                    Conjecture(
                        source=source,
                        target=condition,
                        via=tuple(getattr(path, "intermediates", ()) or ()),
                        hops=int(getattr(path, "hops", len(edges))),
                        strength_upper=float(getattr(path, "strength_upper", 0.0) or 0.0),
                        origin_case=case_id,
                    )
                )
            )
        return entries

    def test(self, source: str, target: str, case_id: str, confirmation: Any) -> bool | None:
        """Test a conjecture against a confirmation. Returns the verdict, or None if not applicable.

        Only an independent confirmation counts. An accepted suggestion is not
        evidence for the leap that produced it, for the same reason it is not
        evidence for anything else: it was produced by the system being tested.
        """
        entry = self.records.get((_normalise(source), _normalise(target)))
        if entry is None or not getattr(confirmation, "is_independent", False):
            return None
        if case_id in entry.confirmed_in or case_id in entry.refuted_in:
            return None
        held = _normalise(getattr(confirmation, "diagnosis", "")) == _normalise(target)
        (entry.confirmed_in if held else entry.refuted_in).append(case_id)
        return held

    def promotable(self) -> list[ConceptEdge]:
        """Conjectures that have earned an edge, with their confirming cases as provenance."""
        return [
            entry.to_edge()
            for entry in self.records.values()
            if entry.is_promotable(self.min_confirmations)
        ]

    def pending(self) -> list[ConjectureRecord]:
        return [entry for entry in self.records.values() if not entry.is_promotable(self.min_confirmations)]

    def report(self) -> dict[str, Any]:
        promotable = self.promotable()
        return {
            "conjectures": len(self.records),
            "tested": sum(1 for entry in self.records.values() if entry.tested),
            "promotable": len(promotable),
            "min_confirmations": self.min_confirmations,
            "records": [entry.as_dict() for entry in self.records.values()],
        }


def leaps_in(hypotheses: Iterable[Any], case_id: str, ledger: ConjectureLedger) -> int:
    """Record every leap in a batch of hypotheses. Returns how many were recorded."""
    return sum(len(ledger.record_from_hypothesis(item, case_id)) for item in hypotheses)


def _normalise(value: str) -> str:
    return " ".join(str(value).lower().split())
