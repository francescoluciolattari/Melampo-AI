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

**Version 2: broadened after two live runs exposed a fixture that was too
small to decide anything.** Two things changed, not one. First, the graph
itself was too coarse: real Claude and GPT-OSS answers cited genuine
intermediate biochemistry (1-alpha-hydroxylase activity, calcitriol, a
fibrillin-1 mutation) that the original nine-edge graph collapsed into a
single link, so a correct, more detailed answer scored as an unfounded one
-- not a candidate failing, a fixture too coarse to recognise a right
answer. Second, four cases cannot separate two candidates with confidence;
sixteen might, and only with an honest accounting of how much confidence
that sample size actually buys, which is why `grounding_rate` alone is no
longer the whole story -- see `grounding_wilson_lower` below.

**Restraint is now measured, not only grounding.** A candidate that invents
a plausible-sounding connection where the graph has none is a more dangerous
failure than one that gets a real connection's mechanism wrong, and a bench
that never asks a candidate to *decline* a connection cannot tell a
disciplined candidate from a confidently wrong one. Restraint cases pair a
factor and target the graph does not connect, and expect ``bears_on`` to
come back ``no``, or -- if a mechanism is offered anyway -- expect the graph
to correctly find no connection rather than being fooled by a plausible one.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..memory.concept_paths import ConceptEdge, ConceptGraphView, InMemoryConceptGraph
from ..memory.information_content import InformationContentTable
from ..memory.ontology_import import wilson_interval
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
    """One hypothesis to vet: a factor, a target, and what the graph knows.

    ``expected_mechanism`` of ``None`` marks a restraint case: the graph
    genuinely does not connect factor and target, and the correct behaviour
    is declining the connection (``bears_on: no``), not proposing a
    plausible-sounding mechanism the graph cannot support. Restraint cases
    are asserted, not merely asked, to have no path in the shipped graph --
    see ``test_every_restraint_case_genuinely_has_no_path`` -- so the
    fixture cannot silently stop testing what it claims to.
    """

    case_id: str
    factor: str
    target: str
    expected_mechanism: str | None
    question: str

    @property
    def is_restraint_case(self) -> bool:
        return self.expected_mechanism is None

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
    # Restraint bookkeeping, kept apart from the grounding counters above:
    # a restraint case has no "grounded" outcome to aim for, and folding it
    # into the same counters would make grounding_rate depend on how many
    # restraint cases happened to be in the set, rather than on how well the
    # candidate grounds a genuine connection.
    restraint_expected: int = 0
    restraint_correct: int = 0
    latencies_seconds: list[float] = field(default_factory=list)
    malformed_answers: list[str] = field(default_factory=list)
    per_case: list[dict[str, Any]] = field(default_factory=list)

    @property
    def conclusive_runs(self) -> int:
        """Runs where a real connection was expected -- the denominator grounding_rate uses."""
        return self.runs - self.restraint_expected

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
        """Of cases where a real connection was expected, how often the graph supported the claim.

        Divides by conclusive runs, not all runs: a restraint case has no
        connection to ground, and counting it in this denominator would
        dilute the rate by cases it was never meant to measure. Restraint
        itself is `restraint_rate`, reported separately.
        """
        return self.grounded / self.conclusive_runs if self.conclusive_runs else 0.0

    @property
    def grounding_wilson_interval(self) -> tuple[float, float]:
        """A confidence interval on grounding_rate, not just the point estimate.

        The reason this bench was widened in the first place: four cases
        cannot separate two candidates, and a raw point estimate hides how
        little a small sample actually establishes. 3/4 and 12/16 are both
        75% grounded, but the first could plausibly be anywhere from roughly
        30% to 95% while the second is pinned much tighter -- ranking by the
        point estimate alone would treat them as identical when they are not
        equally trustworthy. The same interval this project already uses for
        HPO frequency parsing, HypothesisYield, and ConjectureLedger
        promotion, applied here for the same reason: a proportion from a
        small sample should say so.
        """
        return wilson_interval(self.grounded, self.conclusive_runs) if self.conclusive_runs else (0.0, 0.0)

    @property
    def grounding_wilson_lower(self) -> float:
        """The conservative bound `rank_vetting_results` actually sorts on."""
        return self.grounding_wilson_interval[0]

    @property
    def useful_rate(self) -> float:
        """Grounded, or at least identifying a connection the graph knows.

        A candidate naming the wrong mechanism between genuinely connected
        concepts is wrong in a recoverable way -- the graph offers what it
        does know. One claiming a connection where the graph sees none is
        wrong in a different way. Both are failures; only the second is a
        failure of the kind that would surface an unfounded claim.

        Divides by conclusive_runs for the same reason grounding_rate does.
        """
        return (
            (self.grounded + self.connection_but_wrong_mechanism) / self.conclusive_runs
            if self.conclusive_runs
            else 0.0
        )

    @property
    def restraint_rate(self) -> float:
        """Of restraint cases, how often the candidate correctly declined the connection.

        The property this version of the bench exists to add: a candidate
        that invents a plausible connection where the graph has none is a
        more dangerous failure than one that gets a real mechanism wrong,
        and grounding_rate alone cannot see it -- a candidate could ground
        every real connection perfectly while also confidently inventing
        connections that do not exist, and grounding_rate would never
        reflect the second half of that picture.
        """
        return self.restraint_correct / self.restraint_expected if self.restraint_expected else 0.0

    @property
    def mean_latency_seconds(self) -> float:
        """Mean response time, the efficiency tie-break once correctness ties.

        Empty when no timings were recorded (measure_latency=False) rather
        than raising -- timing is an enhancement this bench can use when
        available, not a requirement to run at all.
        """
        return sum(self.latencies_seconds) / len(self.latencies_seconds) if self.latencies_seconds else 0.0

    def as_dict(self) -> dict[str, Any]:
        lower, upper = self.grounding_wilson_interval
        return {
            "model_name": self.model_name,
            "runs": self.runs,
            "conclusive_runs": self.conclusive_runs,
            "format_rate": round(self.format_rate, 4),
            "grounding_rate": round(self.grounding_rate, 4),
            "grounding_wilson_interval": [round(lower, 4), round(upper, 4)],
            "useful_rate": round(self.useful_rate, 4),
            "restraint_rate": round(self.restraint_rate, 4),
            "restraint_expected": self.restraint_expected,
            "mean_latency_seconds": round(self.mean_latency_seconds, 3),
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
    trials_per_case: int = 1,
    measure_latency: bool = True,
) -> VettingResult:
    """Run one candidate over the vetting cases, scored by the graph.

    ``trials_per_case`` repeats every case this many times and folds all
    trials into one result. Real models vary run to run -- a single trial
    per case reports what happened once, not what the candidate reliably
    does, and a decision as consequential as which engine powers vetting
    deserves the second. Kept as a parameter rather than hard-coded, since
    the right number trades directly against API cost: repeating a growing
    case set through a live provider is not free, and a caller pressed for
    budget can still run a single trial across more cases.
    """
    result = VettingResult(model_name=model_name)

    for case in cases:
        for _ in range(trials_per_case):
            result.runs += 1
            started = time.monotonic() if measure_latency else None
            raw = model(case.prompt())
            if started is not None:
                result.latencies_seconds.append(time.monotonic() - started)

            parsed = parse_frame_answer(FRAME_RELEVANCE, raw)
            factor, target = parsed.value("factor"), parsed.value("target")
            bears_on, mechanism = parsed.value("bears_on"), parsed.value("mechanism")

            if not factor or not target or not bears_on:
                result.malformed_answers.append((raw or "")[:120])
                result.per_case.append({"case_id": case.case_id, "outcome": "malformed"})
                continue

            result.well_formed += 1

            if case.is_restraint_case:
                _score_restraint_case(result, case, graph, table, bears_on, mechanism)
                continue

            if not mechanism:
                result.well_formed -= 1
                result.malformed_answers.append((raw or "")[:120])
                result.per_case.append({"case_id": case.case_id, "outcome": "malformed"})
                continue

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


def _score_restraint_case(
    result: VettingResult,
    case: VettingCase,
    graph: ConceptGraphView,
    table: InformationContentTable | None,
    bears_on: str,
    mechanism: str,
) -> None:
    """Score a case where the graph genuinely connects nothing.

    Two ways to pass: say ``no`` outright, or propose a mechanism the graph
    itself then finds unsupported. Only a mechanism the graph mistakenly
    accepts as grounded counts as a failure -- the graph being fooled is
    exactly the dangerous case this scoring exists to catch.
    """
    result.restraint_expected += 1

    if bears_on == "no":
        result.restraint_correct += 1
        result.per_case.append({"case_id": case.case_id, "outcome": "restraint_correct_declined"})
        return

    if not mechanism:
        result.per_case.append({"case_id": case.case_id, "outcome": "restraint_ambiguous_no_mechanism"})
        return

    verification = verify_mechanism(graph, case.factor, case.target, mechanism, table=table)
    if verification.is_grounded:
        # The graph found a real connection the case fixture did not expect.
        # Not a candidate failure -- a fixture defect worth knowing about,
        # since it means this "restraint" case is not actually one.
        result.restraint_correct += 1
        result.per_case.append(
            {"case_id": case.case_id, "outcome": "restraint_case_actually_grounded", "claimed_mechanism": mechanism}
        )
    else:
        result.per_case.append(
            {
                "case_id": case.case_id,
                "outcome": "restraint_failed_invented_connection",
                "claimed_mechanism": mechanism,
            }
        )


def rank_vetting_results(results: Sequence[VettingResult]) -> list[VettingResult]:
    """Restraint first, then the conservative grounding bound, then usefulness, then format, then speed.

    Restraint leads because inventing a connection is the more dangerous
    failure -- a candidate that grounds real connections well but also
    fabricates ones that do not exist is not safe merely for scoring well on
    the first measure. `grounding_wilson_lower`, not the raw point estimate,
    is the tie-break beneath it: ranking on a point estimate would let a
    candidate's lucky small sample outrank another's more reliable larger
    one at the same observed rate. Format and latency come last, not because
    they do not matter, but because a candidate that grounds well and
    formats poorly is a prompt problem, while one that formats perfectly and
    grounds badly is a candidate problem -- and only the second is a reason
    to prefer someone else.
    """
    return sorted(
        results,
        key=lambda item: (
            -item.restraint_rate,
            -item.grounding_wilson_lower,
            -item.useful_rate,
            -item.format_rate,
            item.mean_latency_seconds,
        ),
    )


# --------------------------------------------------------------------------
# A broader curated graph and case set, so the bench is runnable as shipped
# --------------------------------------------------------------------------

VETTING_GRAPH_EDGES = (
    # Marfan / aortic -- the original chain, plus the upstream genetic cause
    # a real GPT-OSS answer cited that the v1 graph had no node for.
    ConceptEdge("fibrillin-1 mutation", "causes", "connective tissue weakness", 0.9),
    ConceptEdge("marfan syndrome", "causes", "fibrillin-1 mutation", 0.95),
    ConceptEdge("marfan syndrome", "causes", "connective tissue weakness", 0.9),
    ConceptEdge("connective tissue weakness", "causes", "aortic root dilation", 0.85),
    # CKD / bone -- unchanged, already grounded correctly in the first fix.
    ConceptEdge("chronic kidney disease", "causes", "secondary hyperparathyroidism", 0.8),
    ConceptEdge("secondary hyperparathyroidism", "causes", "renal osteodystrophy", 0.75),
    # Coeliac / anaemia -- unchanged.
    ConceptEdge("coeliac disease", "causes", "villous atrophy", 0.88),
    ConceptEdge("villous atrophy", "causes", "iron malabsorption", 0.8),
    ConceptEdge("iron malabsorption", "causes", "microcytic anaemia", 0.85),
    # Sarcoidosis / calcium -- the coarse link kept, plus the real biochemical
    # chain (granulomatous macrophages -> 1-alpha-hydroxylase -> calcitriol)
    # that both live candidates actually cited and the v1 graph could not
    # recognise.
    ConceptEdge("sarcoidosis", "causes", "granuloma formation", 0.85),
    ConceptEdge("granuloma formation", "causes", "hypercalcaemia", 0.7),
    ConceptEdge("sarcoidosis", "causes", "granulomatous macrophage activity", 0.85),
    ConceptEdge("granulomatous macrophage activity", "causes", "1-alpha-hydroxylase activity", 0.8),
    ConceptEdge("1-alpha-hydroxylase activity", "causes", "calcitriol excess", 0.85),
    ConceptEdge("calcitriol excess", "causes", "hypercalcaemia", 0.85),
    # Haemochromatosis / cirrhosis -- new organ system: iron overload.
    ConceptEdge("hereditary haemochromatosis", "causes", "hepatic iron deposition", 0.85),
    ConceptEdge("hepatic iron deposition", "causes", "cirrhosis", 0.75),
    # Cushing syndrome / hyperglycaemia -- new organ system: endocrine.
    ConceptEdge("cushing syndrome", "causes", "cortisol excess", 0.9),
    ConceptEdge("cortisol excess", "causes", "hyperglycaemia", 0.7),
    # SIADH / hyponatraemia -- new organ system: electrolyte.
    ConceptEdge("siadh", "causes", "water retention", 0.85),
    ConceptEdge("water retention", "causes", "hyponatraemia", 0.85),
    # Multiple myeloma / renal impairment -- new organ system: haematologic.
    ConceptEdge("multiple myeloma", "causes", "bence jones proteinuria", 0.85),
    ConceptEdge("bence jones proteinuria", "causes", "renal tubular injury", 0.75),
    ConceptEdge("renal tubular injury", "causes", "renal impairment", 0.8),
    # Primary hyperaldosteronism / hypertension -- new organ system: cardiovascular.
    ConceptEdge("primary hyperaldosteronism", "causes", "sodium retention", 0.85),
    ConceptEdge("sodium retention", "causes", "hypertension", 0.75),
    ConceptEdge("primary hyperaldosteronism", "causes", "hypokalaemia", 0.8),
    # Vitamin B12 deficiency / neurological -- new organ system: neurologic.
    ConceptEdge("vitamin b12 deficiency", "causes", "impaired myelin synthesis", 0.8),
    ConceptEdge("impaired myelin synthesis", "causes", "subacute combined degeneration", 0.75),
    # Hypothyroidism / hyperlipidaemia -- new organ system: metabolic.
    ConceptEdge("hypothyroidism", "causes", "reduced ldl receptor activity", 0.75),
    ConceptEdge("reduced ldl receptor activity", "causes", "hyperlipidaemia", 0.75),
    # SLE / nephritis -- new organ system: rheumatologic.
    ConceptEdge("systemic lupus erythematosus", "causes", "immune complex deposition", 0.85),
    ConceptEdge("immune complex deposition", "causes", "lupus nephritis", 0.8),
)

VETTING_GRAPH_FREQUENCIES = {
    "fibrillin-1 mutation": 30, "marfan syndrome": 40, "connective tissue weakness": 120,
    "aortic root dilation": 300,
    "chronic kidney disease": 900, "secondary hyperparathyroidism": 150, "renal osteodystrophy": 60,
    "coeliac disease": 200, "villous atrophy": 90, "iron malabsorption": 110, "microcytic anaemia": 700,
    "sarcoidosis": 130, "granuloma formation": 95, "hypercalcaemia": 500,
    "granulomatous macrophage activity": 40, "1-alpha-hydroxylase activity": 25, "calcitriol excess": 35,
    "hereditary haemochromatosis": 60, "hepatic iron deposition": 45, "cirrhosis": 400,
    "cushing syndrome": 70, "cortisol excess": 55, "hyperglycaemia": 600,
    "siadh": 80, "water retention": 300, "hyponatraemia": 450,
    "multiple myeloma": 90, "bence jones proteinuria": 40, "renal tubular injury": 65, "renal impairment": 500,
    "primary hyperaldosteronism": 50, "sodium retention": 200, "hypertension": 900, "hypokalaemia": 350,
    "vitamin b12 deficiency": 150, "impaired myelin synthesis": 30, "subacute combined degeneration": 20,
    "hypothyroidism": 400, "reduced ldl receptor activity": 25, "hyperlipidaemia": 550,
    "systemic lupus erythematosus": 100, "immune complex deposition": 45, "lupus nephritis": 70,
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
    VettingCase(
        "haemochromatosis_cirrhosis", "hereditary haemochromatosis", "cirrhosis", "hepatic iron deposition",
        "A patient with hereditary haemochromatosis develops cirrhosis on biopsy. Does the haemochromatosis "
        "bear on the liver finding, and by what mechanism?",
    ),
    VettingCase(
        "cushing_hyperglycaemia", "cushing syndrome", "hyperglycaemia", "cortisol excess",
        "A patient with Cushing syndrome has new hyperglycaemia on routine bloods. Does the Cushing syndrome "
        "bear on the glucose finding, and by what mechanism?",
    ),
    VettingCase(
        "siadh_hyponatraemia", "siadh", "hyponatraemia", "water retention",
        "A patient with SIADH has hyponatraemia on electrolyte panel. Does the SIADH bear on the sodium "
        "finding, and by what mechanism?",
    ),
    VettingCase(
        "myeloma_renal", "multiple myeloma", "renal impairment", "bence jones proteinuria",
        "A patient with multiple myeloma develops renal impairment. Does the myeloma bear on the renal "
        "finding, and through what intermediate?",
    ),
    VettingCase(
        "aldosteronism_hypertension", "primary hyperaldosteronism", "hypertension", "sodium retention",
        "A patient with primary hyperaldosteronism has resistant hypertension. Does the hyperaldosteronism "
        "bear on the blood pressure finding, and by what mechanism?",
    ),
    VettingCase(
        "b12_scd", "vitamin b12 deficiency", "subacute combined degeneration", "impaired myelin synthesis",
        "A patient with untreated vitamin B12 deficiency develops subacute combined degeneration of the "
        "spinal cord. Does the B12 deficiency bear on this finding, and by what mechanism?",
    ),
    VettingCase(
        "hypothyroid_lipids", "hypothyroidism", "hyperlipidaemia", "reduced ldl receptor activity",
        "A patient with untreated hypothyroidism has new hyperlipidaemia. Does the hypothyroidism bear on "
        "the lipid finding, and by what mechanism?",
    ),
    VettingCase(
        "sle_nephritis", "systemic lupus erythematosus", "lupus nephritis", "immune complex deposition",
        "A patient with systemic lupus erythematosus develops nephritis. Does the lupus bear on the renal "
        "finding, and by what mechanism?",
    ),
    VettingCase(
        "sarcoid_calcium_detailed", "granulomatous macrophage activity", "hypercalcaemia", "1-alpha-hydroxylase activity",
        "In a sarcoidosis case, granulomatous macrophage activity is documented on biopsy, and the patient "
        "has hypercalcaemia. Does the macrophage activity bear on the calcium finding, and by what "
        "intermediate step?",
    ),
    # Restraint cases: the graph genuinely connects nothing between factor and
    # target. Correct behaviour is bears_on: no, or a mechanism the graph
    # itself then finds unsupported -- never a plausible-sounding invention
    # the graph mistakenly accepts.
    VettingCase(
        "restraint_sarcoid_villous", "sarcoidosis", "villous atrophy", None,
        "A patient with sarcoidosis is found to have villous atrophy on duodenal biopsy at an unrelated "
        "workup. Does the sarcoidosis bear on the villous atrophy, and if so, by what mechanism?",
    ),
    VettingCase(
        "restraint_myeloma_thyroid", "multiple myeloma", "hypothyroidism", None,
        "A patient with multiple myeloma is incidentally found to have hypothyroidism on screening labs. "
        "Does the myeloma bear on the thyroid finding, and if so, by what mechanism?",
    ),
    VettingCase(
        "restraint_b12_aldosterone", "vitamin b12 deficiency", "primary hyperaldosteronism", None,
        "A patient with vitamin B12 deficiency is separately found to have primary hyperaldosteronism on "
        "endocrine workup. Does the B12 deficiency bear on the aldosteronism, and if so, by what mechanism?",
    ),
)


def vetting_graph() -> InMemoryConceptGraph:
    return InMemoryConceptGraph.from_edges(list(VETTING_GRAPH_EDGES))


def vetting_table() -> InformationContentTable:
    return InformationContentTable.from_frequencies(VETTING_GRAPH_FREQUENCIES)
