"""Measure a candidate on vetting, not navigation -- a different task, a different bench.

The format-adherence bench measures whether a candidate can find a fact in a
set of documents and report it in a strict grammar. That is navigation, and
nemotron-3-super and gemma-3-27b won it. This bench measures something the
first one never touches: given a hypothesis that a factor bears on a target
through some mechanism, can the candidate state that claim in a form the
concept graph can check, and does the graph then support what it claimed?

**Why a separate bench rather than more cases in the existing one.** The two
tasks reward different things. Navigation rewards finding what is in the
documents and stopping. Vetting rewards proposing a mechanism that is not in
the documents at all -- it is in the graph, or nowhere -- and being right
about it. A candidate that excels at one may be mediocre at the other, and
the project has already been wrong once by assuming a bench result transfers
to a task it did not measure.

**What makes this measurable at all: the graph is the judge, not another
model.** Each case has a factor and a target that a curated graph genuinely
connects through a known mechanism. A candidate proposes the mechanism;
`mechanism_verification.verify_mechanism` checks it against the graph, and
the grounding it returns is the score. No model grades another model's
answer, so the measurement stays deterministic and inspectable -- the same
discipline that made `root_model_cross_check` refuse to appoint an
adjudicator.

**What this bench does not measure.** Whether a mechanism is *clinically*
correct in a way the graph does not encode. A candidate proposing a
mechanism that is true but absent from the graph scores as ungrounded here,
which is correct behaviour for this bench (the graph is the stated judge)
and a limitation to state plainly: the bench measures agreement with the
graph's knowledge, not with medicine.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..memory.concept_paths import ConceptEdge, ConceptGraphView, InMemoryConceptGraph
from ..memory.information_content import InformationContentTable
from ..reasoning.frame_answer import (
    FRAME_RELEVANCE,
    frame_prompt_instruction,
    parse_frame_answer,
)
from ..reasoning.mechanism_verification import (
    GROUNDING_CONNECTION_WITHOUT_THIS_MECHANISM,
    GROUNDING_NO_CONNECTION,
    verify_mechanism,
)


@dataclass(frozen=True)
class VettingCase:
    """One hypothesis to vet: a factor, a target, and what the graph knows."""

    case_id: str
    factor: str
    target: str
    # The mechanism the graph actually connects them through. Not shown to the
    # candidate -- it is what the graph will confirm or refuse, and naming it
    # here is for the case's own documentation and for asserting the fixture
    # is well-formed, never for scoring by string comparison against the
    # candidate's answer.
    expected_mechanism: str
    question: str

    def prompt(self) -> str:
        """What the candidate is asked, including the required answer format."""
        return (
            f"{self.question}\n\n{frame_prompt_instruction(FRAME_RELEVANCE)}\n\n"
            f"Emit only the answer line, nothing else."
        )


@dataclass
class VettingResult:
    """How one candidate did across the vetting cases."""

    model_name: str
    runs: int = 0
    well_formed: int = 0
    grounded: int = 0
    connection_but_wrong_mechanism: int = 0
    no_connection_claimed: int = 0
    malformed_answers: list[str] = field(default_factory=list)
    per_case: list[dict[str, Any]] = field(default_factory=list)

    @property
    def format_rate(self) -> float:
        """How often the answer parsed into the relevance frame's slots at all.

        Kept separate from grounding: a candidate that cannot produce the
        format is failing for a different reason than one that produces it
        and proposes an unsupported mechanism, and the two need different
        responses -- prompt work versus candidate replacement.
        """
        return self.well_formed / self.runs if self.runs else 0.0

    @property
    def grounding_rate(self) -> float:
        """Of all cases run, how often the graph supported the claim.

        Over runs rather than over well-formed answers: a candidate that
        produces three perfect answers and fails to format seventeen has not
        earned a 100% score, and dividing by well-formed alone would give it
        one.
        """
        return self.grounded / self.runs if self.runs else 0.0

    @property
    def useful_rate(self) -> float:
        """Grounded, or at least identifying a connection the graph knows.

        A candidate naming the wrong mechanism between genuinely connected
        concepts is wrong in a recoverable way -- the graph offers what it
        does know. One claiming a connection where the graph sees none is
        wrong in a different way. Both are failures; only the second is a
        failure of the kind that would surface an unfounded claim.
        """
        return (self.grounded + self.connection_but_wrong_mechanism) / self.runs if self.runs else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "runs": self.runs,
            "format_rate": round(self.format_rate, 4),
            "grounding_rate": round(self.grounding_rate, 4),
            "useful_rate": round(self.useful_rate, 4),
            "grounded": self.grounded,
            "connection_but_wrong_mechanism": self.connection_but_wrong_mechanism,
            "no_connection_claimed": self.no_connection_claimed,
            "malformed_answers": list(self.malformed_answers[:5]),
            "per_case": list(self.per_case),
        }


def bench_vetting(
    model_name: str,
    model: Callable[[str], str],
    cases: Sequence[VettingCase],
    graph: ConceptGraphView,
    *,
    table: InformationContentTable | None = None,
) -> VettingResult:
    """Run one candidate over the vetting cases, scored by the graph."""
    result = VettingResult(model_name=model_name)

    for case in cases:
        result.runs += 1
        raw = model(case.prompt())
        parsed = parse_frame_answer(FRAME_RELEVANCE, raw)
        factor, target = parsed.value("factor"), parsed.value("target")
        mechanism = parsed.value("mechanism")

        if not factor or not target or not mechanism:
            result.malformed_answers.append((raw or "")[:120])
            result.per_case.append({"case_id": case.case_id, "outcome": "malformed"})
            continue

        result.well_formed += 1
        verification = verify_mechanism(graph, factor, target, mechanism, table=table)

        if verification.is_grounded:
            result.grounded += 1
            outcome = "grounded"
        elif verification.grounding == GROUNDING_CONNECTION_WITHOUT_THIS_MECHANISM:
            result.connection_but_wrong_mechanism += 1
            outcome = "connection_but_wrong_mechanism"
        else:
            result.no_connection_claimed += 1
            outcome = GROUNDING_NO_CONNECTION

        result.per_case.append(
            {
                "case_id": case.case_id,
                "outcome": outcome,
                "claimed_mechanism": mechanism,
                "matched_concept": verification.matched_concept,
            }
        )

    return result


def rank_vetting_results(results: Sequence[VettingResult]) -> list[VettingResult]:
    """Grounding first, then usefulness, then format.

    Grounding leads because it is the property the role exists for. Format
    comes last, not because it does not matter, but because a candidate that
    grounds well and formats poorly is a prompt problem, while one that
    formats perfectly and grounds badly is a candidate problem -- and only
    the second is a reason to prefer someone else.
    """
    return sorted(results, key=lambda item: (-item.grounding_rate, -item.useful_rate, -item.format_rate))


# --------------------------------------------------------------------------
# A small curated graph and case set, so the bench is runnable as shipped
# --------------------------------------------------------------------------

VETTING_GRAPH_EDGES = (
    ConceptEdge("marfan syndrome", "causes", "connective tissue weakness", 0.9),
    ConceptEdge("connective tissue weakness", "causes", "aortic root dilation", 0.85),
    ConceptEdge("chronic kidney disease", "causes", "secondary hyperparathyroidism", 0.8),
    ConceptEdge("secondary hyperparathyroidism", "causes", "renal osteodystrophy", 0.75),
    ConceptEdge("coeliac disease", "causes", "villous atrophy", 0.88),
    ConceptEdge("villous atrophy", "causes", "iron malabsorption", 0.8),
    ConceptEdge("iron malabsorption", "causes", "microcytic anaemia", 0.85),
    ConceptEdge("sarcoidosis", "causes", "granuloma formation", 0.85),
    ConceptEdge("granuloma formation", "causes", "hypercalcaemia", 0.7),
)

VETTING_GRAPH_FREQUENCIES = {
    "marfan syndrome": 40, "connective tissue weakness": 120, "aortic root dilation": 300,
    "chronic kidney disease": 900, "secondary hyperparathyroidism": 150, "renal osteodystrophy": 60,
    "coeliac disease": 200, "villous atrophy": 90, "iron malabsorption": 110, "microcytic anaemia": 700,
    "sarcoidosis": 130, "granuloma formation": 95, "hypercalcaemia": 500,
}

VETTING_CASES = (
    VettingCase(
        "marfan_aortic", "marfan syndrome", "aortic root dilation", "connective tissue weakness",
        "A patient's sister has confirmed Marfan syndrome. Today's echocardiogram shows aortic root "
        "dilation. Does the family history bear on this measurement, and by what mechanism?",
    ),
    VettingCase(
        "ckd_bone", "chronic kidney disease", "renal osteodystrophy", "secondary hyperparathyroidism",
        "A patient with long-standing chronic kidney disease has new bone pain and radiographic changes "
        "consistent with renal osteodystrophy. Does the kidney disease bear on the bone findings, and how?",
    ),
    VettingCase(
        "coeliac_anaemia", "coeliac disease", "iron malabsorption", "villous atrophy",
        "A patient with newly diagnosed coeliac disease has iron malabsorption on testing. Does the coeliac "
        "disease bear on the malabsorption, and through what intermediate?",
    ),
    VettingCase(
        "sarcoid_calcium", "sarcoidosis", "hypercalcaemia", "granuloma formation",
        "A patient with sarcoidosis presents with hypercalcaemia. Does the sarcoidosis bear on the calcium "
        "level, and by what mechanism?",
    ),
)


def vetting_graph() -> InMemoryConceptGraph:
    return InMemoryConceptGraph.from_edges(list(VETTING_GRAPH_EDGES))


def vetting_table() -> InformationContentTable:
    return InformationContentTable.from_frequencies(VETTING_GRAPH_FREQUENCIES)
