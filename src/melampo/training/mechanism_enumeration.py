"""Hypothesis generation by path enumeration over the concept graph.

``DreamTrainer._alternative_hypotheses`` currently builds scenarios by string
concatenation — ``f"{base_label}_alt_1"`` — and ``CounterfactualSampler``
computes novelty as ``0.2 * len(perturbation_plan)``, which is arithmetic on a
list length rather than a measure of anything. Neither produces a scenario.

This module replaces the generation step. A hypothesis is not written, it is
found: a path through the concept graph connecting the observed findings to a
condition that the case has not raised. The distinction from a generative model
is categorical rather than one of quality — a path either exists in the graph or
it does not, so a hypothesis cannot be fluent and baseless at the same time.

Novelty likewise stops being invented. A short path over well-attested edges
describes a connection any clinician would raise, so its novelty is low. A long
path over weak edges describes something rarely considered: high novelty, low
confidence, and traceable either way. The two properties are reported
separately because they are not the same question, and collapsing them into one
score is what makes a speculative hypothesis look like a strong one.

Output is delivered through ``hypothesis_channel``, which admits it as an
exclusion hypothesis only, gated on diagnostic indeterminacy.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..memory.concept_paths import (
    ConceptGraphView,
    ConceptPath,
    DensityReport,
    find_paths,
    local_density,
    normalise_concept,
)
from .hypothesis_channel import HypothesisEnvelope


@dataclass(frozen=True)
class MechanismHypothesis:
    """A candidate condition reached from the case findings by a traversable path."""

    condition: str
    paths: tuple[ConceptPath, ...]
    findings_linked: tuple[str, ...]

    @property
    def support(self) -> float:
        """Point strength of the best supporting path."""
        return max((path.strength for path in self.paths), default=0.0)

    @property
    def plausibility(self) -> float:
        """Upper bound over the supporting paths: what could be true.

        Hypothesis generation reads this, while the diagnostic path reads the
        lower bound. Same graph, opposite projections — exploration is optimistic
        by construction and decision is prudent by construction.
        """
        return max((path.strength_upper for path in self.paths), default=0.0)

    @property
    def guaranteed(self) -> float:
        return max((path.strength_lower for path in self.paths), default=0.0)

    @property
    def epistemic_width(self) -> float:
        return self.plausibility - self.guaranteed

    @property
    def gap_count(self) -> int:
        return min((path.gap_count for path in self.paths), default=0)

    @property
    def shortest_hops(self) -> int:
        return min((path.hops for path in self.paths), default=0)

    @property
    def novelty(self) -> float:
        """How far outside routine consideration this hypothesis sits.

        Rises with path length and falls with edge strength. Reported alongside
        ``support`` rather than folded into it: a hypothesis can be novel and
        weakly supported at once, and a single number would hide that.
        """
        if not self.paths:
            return 0.0
        distance = min(1.0, (self.shortest_hops - 1) / 3.0)
        return round(min(1.0, distance * 0.6 + (1.0 - self.support) * 0.4), 3)

    @property
    def corroboration(self) -> int:
        """How many distinct case findings reach this condition."""
        return len(self.findings_linked)

    def rationale(self) -> str:
        best = min(self.paths, key=lambda path: (path.hops, -path.strength))
        linked = ", ".join(self.findings_linked)
        return (
            f"Reached from {linked} via {best.describe()} "
            f"({best.hops} hop(s), support {best.strength:.2f}). Not raised by the case."
        )

    def as_envelope(self) -> HypothesisEnvelope:
        return HypothesisEnvelope(
            label=self.condition,
            rationale=self.rationale(),
            origin="concept_path_enumeration",
            novelty_score=self.novelty,
            supporting_patterns=[path.describe() for path in self.paths],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "support": round(self.support, 3),
            "plausibility": round(self.plausibility, 3),
            "guaranteed": round(self.guaranteed, 3),
            "epistemic_width": round(self.epistemic_width, 3),
            "gap_count": self.gap_count,
            "novelty": self.novelty,
            "shortest_hops": self.shortest_hops,
            "corroboration": self.corroboration,
            "findings_linked": list(self.findings_linked),
            "paths": [path.as_provenance() for path in self.paths],
        }


@dataclass
class MechanismEnumerator:
    """Enumerate candidate conditions reachable from the observed findings."""

    graph: ConceptGraphView
    max_hops: int = 3
    min_edge_weight: float = 0.0
    min_support: float = 0.05
    min_plausibility: float = 0.05
    max_candidates: int = 10
    max_gap_edges: int = 1
    min_density: float = 0.5
    min_corroboration_when_sparse: int = 2

    def enumerate(
        self,
        findings: Sequence[str],
        candidate_conditions: Sequence[str],
        *,
        already_considered: Sequence[str] = (),
    ) -> list[MechanismHypothesis]:
        """Return conditions the case has not raised but the graph connects to it.

        ``already_considered`` is excluded because a hypothesis the differential
        already contains is not a hypothesis, and re-proposing it consumes review
        attention that the channel is rationed to protect.

        Candidates are filtered and ranked by the **upper** bound rather than by
        point support. Filtering on support discarded exactly the uncertain
        paths, so the enumerator surfaced conditions the graph already knew well
        — the hypotheses a clinician had already formed. It also contradicted
        the novelty measure, which rewards low support: one mechanism sought
        uncertainty while the filter removed it before novelty was ever computed.
        """
        excluded = {normalise_concept(item) for item in already_considered}
        finding_keys = [item for item in findings if normalise_concept(item)]

        collected: dict[str, tuple[list[ConceptPath], list[str]]] = {}
        for condition in candidate_conditions:
            key = normalise_concept(condition)
            if not key or key in excluded:
                continue
            paths: list[ConceptPath] = []
            linked: list[str] = []
            for finding in finding_keys:
                if normalise_concept(finding) == key:
                    continue
                found = find_paths(
                    self.graph,
                    finding,
                    condition,
                    max_hops=self.max_hops,
                    min_edge_weight=self.min_edge_weight,
                    max_gap_edges=self.max_gap_edges,
                )
                usable = [path for path in found if path.strength_upper >= self.min_plausibility]
                if usable:
                    paths.extend(usable)
                    linked.append(finding)
            if paths:
                collected[condition] = (paths, linked)

        hypotheses = [
            MechanismHypothesis(
                condition=condition,
                paths=tuple(sorted(paths, key=lambda path: (path.hops, -path.strength))),
                findings_linked=tuple(linked),
            )
            for condition, (paths, linked) in collected.items()
        ]

        hypotheses.sort(key=lambda item: (-item.corroboration, -item.plausibility, item.shortest_hops))
        return hypotheses[: self.max_candidates]

    def envelopes(self, hypotheses: Sequence[MechanismHypothesis]) -> list[HypothesisEnvelope]:
        return [hypothesis.as_envelope() for hypothesis in hypotheses]

    def run(
        self,
        findings: Sequence[str],
        candidate_conditions: Sequence[str],
        *,
        already_considered: Sequence[str] = (),
    ) -> "EnumerationOutcome":
        """Enumerate in whichever register the local graph coverage supports.

        On a sparse neighbourhood every unknown path reaches an upper bound of
        1.0, so ranking by the ceiling degenerates: a three-hop chain of unknowns
        outranks a two-hop chain of well-attested edges, and the order is decided
        by tie-breaks. Emitting the top three of an arbitrary order is worse than
        emitting none.

        So the register changes rather than the branch falling silent. Sparse
        coverage yields questions aimed at the knowledge base, which is useful
        from the first day and breaks the deadlock where the graph waits for use
        and use waits for the graph.

        Corroboration is the exception that keeps the branch usable early: a
        spurious path from one finding is easy on a sparse graph, but two
        independent findings converging on the same condition is not, so a
        corroborated hypothesis survives a sparse neighbourhood.
        """
        density = local_density(self.graph, findings)
        hypotheses = self.enumerate(findings, candidate_conditions, already_considered=already_considered)

        if density.density >= self.min_density:
            return EnumerationOutcome(mode=MODE_HYPOTHESES, density=density, hypotheses=hypotheses)

        corroborated = [item for item in hypotheses if item.corroboration >= self.min_corroboration_when_sparse]
        if corroborated:
            return EnumerationOutcome(mode=MODE_HYPOTHESES, density=density, hypotheses=corroborated)

        return EnumerationOutcome(
            mode=MODE_KNOWLEDGE_GAP,
            density=density,
            open_questions=self._open_questions(findings, candidate_conditions, hypotheses),
        )

    def _open_questions(
        self,
        findings: Sequence[str],
        candidate_conditions: Sequence[str],
        hypotheses: Sequence[MechanismHypothesis],
    ) -> list["OpenQuestion"]:
        reached = {normalise_concept(item.condition) for item in hypotheses}
        questions: list[OpenQuestion] = []
        for finding in findings:
            for condition in candidate_conditions:
                if normalise_concept(condition) in reached:
                    continue
                if normalise_concept(condition) == normalise_concept(finding):
                    continue
                questions.append(
                    OpenQuestion(
                        finding=finding,
                        condition=condition,
                        reason="no traversable path; absence reflects graph coverage, not clinical evidence",
                    )
                )
        return questions[: self.max_candidates]


MODE_HYPOTHESES = "clinical_hypotheses"
MODE_KNOWLEDGE_GAP = "knowledge_gap_questions"


@dataclass(frozen=True)
class OpenQuestion:
    """A question directed at the knowledge base rather than at the patient."""

    finding: str
    condition: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "finding": self.finding,
            "condition": self.condition,
            "question": (
                f"Does {self.finding} relate to {self.condition}? The graph does not say."
            ),
            "reason": self.reason,
            "target": "graph_completion_queue",
            "clinical_use": False,
        }


@dataclass
class EnumerationOutcome:
    """Result of one run, in whichever register the graph coverage supports."""

    mode: str
    density: DensityReport
    hypotheses: list[MechanismHypothesis] = field(default_factory=list)
    open_questions: list[OpenQuestion] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "density": self.density.as_dict(),
            "hypotheses": [item.as_dict() for item in self.hypotheses],
            "open_questions": [item.as_dict() for item in self.open_questions],
            "interpretation": (
                "graph coverage around this case is too sparse for hypothesis ranking to "
                "carry meaning; the output names open questions for the knowledge base"
                if self.mode == MODE_KNOWLEDGE_GAP
                else "graph coverage supports ranked hypotheses"
            ),
        }


@dataclass
class EnumerationReport:
    """Summary of one enumeration run, for the audit trail."""

    findings: list[str] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    rejected_without_path: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "findings": list(self.findings),
            "hypotheses": list(self.hypotheses),
            "rejected_without_path": list(self.rejected_without_path),
            "hypothesis_count": len(self.hypotheses),
        }
