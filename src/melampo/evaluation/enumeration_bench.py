"""Measure the differential itself: a spread of hypotheses, ranked, and knowing when not to rank.

The vetting bench measures one claim at a time -- a factor, a target, a
mechanism, grounded or not. That is a real brick, but it is not the
behaviour a differential actually needs, which was named directly in
discussion: given a case with several findings, produce a spread of
competing hypotheses with their relative likelihoods in a sensible order,
and when there is not enough to conclude, say so and ask for what would
settle it rather than ranking noise.

`MechanismEnumerator` already produces exactly that shape --
`EnumerationOutcome` carries either `hypotheses` (each with `support` and
`plausibility` from real graph paths) or `open_questions`, and picks between
the two registers itself based on local graph density. Nothing measured
whether it picks well. This bench does.

**Four properties, measured separately, because they fail independently.**

*Recall*: is the condition that was actually confirmed present in the spread
at all? A differential that omits the right answer has failed regardless of
how well it ordered the rest.

*Ranking*: is it near the top? Present-but-eleventh is a different and
lesser failure than absent, and collapsing them into one number would hide
which happened.

*Restraint*: on a case the graph genuinely cannot support, does the system
emit `knowledge_gap_questions` rather than ranked hypotheses? This is the
property hardest to get right and easiest to score badly on by being
confidently wrong -- a system that always ranks something scores well on
recall and catastrophically here.

*Question quality*: when it does abstain, do the open questions name the
findings and conditions actually at issue, or generic filler?

**No model is involved in scoring.** Each case declares which condition was
confirmed and whether the graph should be able to support a conclusion; the
graph and the enumerator do the rest. The same refusal to appoint an
adjudicator that governs `root_model_cross_check` and `vetting_bench`.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..memory.concept_paths import ConceptEdge, ConceptGraphView, InMemoryConceptGraph
from ..training.mechanism_enumeration import (
    MODE_HYPOTHESES,
    MODE_KNOWLEDGE_GAP,
    EnumerationOutcome,
    MechanismEnumerator,
)


@dataclass(frozen=True)
class DifferentialCase:
    """One case: findings, candidates to weigh, and what turned out to be true."""

    case_id: str
    findings: tuple[str, ...]
    candidate_conditions: tuple[str, ...]
    # The condition independently confirmed for this case. None means the case
    # exists specifically to test restraint -- there is no right answer the
    # graph could reach, and ranking anything confidently is the failure.
    confirmed_condition: str | None
    # Whether the graph should be able to support ranked hypotheses at all.
    # Declared per case rather than inferred, so a case testing restraint is
    # explicitly marked as such rather than silently looking like a miss.
    graph_should_support_conclusion: bool = True


@dataclass
class EnumerationResult:
    """How the enumerator did across a set of differential cases."""

    runs: int = 0
    # Cases where a conclusion was expected
    conclusive_expected: int = 0
    confirmed_in_spread: int = 0
    confirmed_ranked_first: int = 0
    rank_positions: list[int] = field(default_factory=list)
    # Cases where restraint was expected
    restraint_expected: int = 0
    correctly_abstained: int = 0
    questions_named_real_terms: int = 0
    per_case: list[dict[str, Any]] = field(default_factory=list)

    @property
    def recall(self) -> float:
        """Of cases where a conclusion was expected, how often the confirmed
        condition appeared in the spread at all."""
        return self.confirmed_in_spread / self.conclusive_expected if self.conclusive_expected else 0.0

    @property
    def top_rank_rate(self) -> float:
        return self.confirmed_ranked_first / self.conclusive_expected if self.conclusive_expected else 0.0

    @property
    def mean_rank(self) -> float:
        """Average position of the confirmed condition, over cases where it appeared.

        Over cases where it appeared, not all cases: averaging in a sentinel
        for absent ones would blend two different failures into one number,
        which `recall` already reports separately and more honestly.
        """
        return sum(self.rank_positions) / len(self.rank_positions) if self.rank_positions else 0.0

    @property
    def restraint_rate(self) -> float:
        """Of cases the graph cannot support, how often the system declined to rank.

        The property most easily scored well on by accident and most
        dangerous to fail: a system that always ranks something looks strong
        on recall and is wrong exactly where a clinician most needs it to
        say so.
        """
        return self.correctly_abstained / self.restraint_expected if self.restraint_expected else 0.0

    @property
    def question_quality(self) -> float:
        return self.questions_named_real_terms / self.correctly_abstained if self.correctly_abstained else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "recall": round(self.recall, 4),
            "top_rank_rate": round(self.top_rank_rate, 4),
            "mean_rank": round(self.mean_rank, 2),
            "restraint_rate": round(self.restraint_rate, 4),
            "question_quality": round(self.question_quality, 4),
            "conclusive_expected": self.conclusive_expected,
            "restraint_expected": self.restraint_expected,
            "per_case": list(self.per_case),
        }


def _rank_of(outcome: EnumerationOutcome, condition: str) -> int | None:
    """1-based position of a condition among ranked hypotheses, or None if absent."""
    for index, hypothesis in enumerate(outcome.hypotheses, start=1):
        if hypothesis.condition == condition:
            return index
    return None


def bench_enumeration(
    enumerator: MechanismEnumerator, cases: Sequence[DifferentialCase]
) -> EnumerationResult:
    """Run the enumerator over differential cases, scoring the four properties."""
    result = EnumerationResult()

    for case in cases:
        result.runs += 1
        outcome = enumerator.run(list(case.findings), list(case.candidate_conditions))
        record: dict[str, Any] = {"case_id": case.case_id, "mode": outcome.mode}

        if case.graph_should_support_conclusion and case.confirmed_condition:
            result.conclusive_expected += 1
            rank = _rank_of(outcome, case.confirmed_condition) if outcome.mode == MODE_HYPOTHESES else None
            if rank is not None:
                result.confirmed_in_spread += 1
                result.rank_positions.append(rank)
                if rank == 1:
                    result.confirmed_ranked_first += 1
                record["rank_of_confirmed"] = rank
            else:
                record["rank_of_confirmed"] = None
            record["spread_size"] = len(outcome.hypotheses)
        else:
            result.restraint_expected += 1
            abstained = outcome.mode == MODE_KNOWLEDGE_GAP
            record["abstained"] = abstained
            if abstained:
                result.correctly_abstained += 1
                # A question naming a finding and a condition actually from
                # this case is substantive; one naming neither is filler that
                # would look identical in a count-only metric.
                named = any(
                    question.finding in case.findings and question.condition in case.candidate_conditions
                    for question in outcome.open_questions
                )
                if named:
                    result.questions_named_real_terms += 1
                record["questions_named_real_terms"] = named
                record["question_count"] = len(outcome.open_questions)

        result.per_case.append(record)

    return result


# --------------------------------------------------------------------------
# A graph and case set with deliberately uneven coverage
# --------------------------------------------------------------------------

# Dense around sarcoidosis and coeliac disease, deliberately sparse around
# amyloidosis -- so a bench run exercises both registers rather than only the
# comfortable one. A fixture where every neighbourhood is well covered would
# never test restraint at all, which is the property most worth measuring.
DIFFERENTIAL_GRAPH_EDGES = (
    ConceptEdge("sarcoidosis", "manifests_as", "hypercalcaemia", 0.7),
    ConceptEdge("sarcoidosis", "manifests_as", "bilateral hilar lymphadenopathy", 0.85),
    ConceptEdge("sarcoidosis", "manifests_as", "erythema nodosum", 0.6),
    ConceptEdge("tuberculosis", "manifests_as", "bilateral hilar lymphadenopathy", 0.5),
    ConceptEdge("tuberculosis", "manifests_as", "night sweats", 0.7),
    ConceptEdge("lymphoma", "manifests_as", "bilateral hilar lymphadenopathy", 0.55),
    ConceptEdge("lymphoma", "manifests_as", "night sweats", 0.6),
    ConceptEdge("coeliac disease", "manifests_as", "iron malabsorption", 0.8),
    ConceptEdge("coeliac disease", "manifests_as", "villous atrophy", 0.88),
    ConceptEdge("coeliac disease", "manifests_as", "chronic diarrhoea", 0.65),
    ConceptEdge("crohn disease", "manifests_as", "chronic diarrhoea", 0.7),
    # Amyloidosis exists as a candidate but its characteristic findings
    # (periorbital purpura, macroglossia) are deliberately absent from the
    # graph entirely -- that absence is what the restraint case tests.
    ConceptEdge("amyloidosis", "manifests_as", "proteinuria", 0.4),
)

DIFFERENTIAL_CASES = (
    DifferentialCase(
        "sarcoid_spread",
        findings=("bilateral hilar lymphadenopathy", "hypercalcaemia", "erythema nodosum"),
        candidate_conditions=("sarcoidosis", "tuberculosis", "lymphoma"),
        confirmed_condition="sarcoidosis",
    ),
    DifferentialCase(
        "coeliac_spread",
        findings=("iron malabsorption", "villous atrophy", "chronic diarrhoea"),
        candidate_conditions=("coeliac disease", "crohn disease"),
        confirmed_condition="coeliac disease",
    ),
    DifferentialCase(
        "lymphadenopathy_alone",
        findings=("bilateral hilar lymphadenopathy",),
        candidate_conditions=("sarcoidosis", "tuberculosis", "lymphoma"),
        confirmed_condition="sarcoidosis",
    ),
    DifferentialCase(
        "sparse_corner",
        # Findings the graph has no edges for at all. This is what local
        # density actually measures -- whether the findings themselves are
        # mapped -- and two earlier attempts at this fixture got it wrong:
        # a weakly-attested edge still gives density 1.0 (the graph knows
        # this corner, it just knows little), and so does an unknown-strength
        # edge. Restraint is for findings the graph has never heard of, which
        # is the honest form of "not enough coverage to rank anything".
        findings=("periorbital purpura", "macroglossia"),
        candidate_conditions=("amyloidosis", "lymphoma"),
        confirmed_condition=None,
        graph_should_support_conclusion=False,
    ),
)


def differential_graph() -> InMemoryConceptGraph:
    return InMemoryConceptGraph.from_edges(list(DIFFERENTIAL_GRAPH_EDGES))


def default_enumerator(graph: ConceptGraphView | None = None) -> MechanismEnumerator:
    return MechanismEnumerator(graph=graph or differential_graph())
