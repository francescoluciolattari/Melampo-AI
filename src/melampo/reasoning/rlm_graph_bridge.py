"""The bridge between what the RLM reads and what the graph knows.

Identified directly in discussion as the missing link, and it was: the RLM
reads a patient's documents and never touches the concept graph; the
enumerator walks the graph and never reads a document. Nothing put them in
contact. Each was a complete half of a diagnostic reasoner with no path
between them.

The bridge runs in both directions, and both matter.

**Documents to graph.** Findings the RLM located in the case become the
entry points for `retrieve_candidates` and then `MechanismEnumerator`, so
the differential is grounded in what this patient's documents actually say
rather than a candidate list someone supplied. This is the direction that
makes the enumerator usable at all on a real case.

**Graph back to documents.** A hypothesis the graph ranks highly predicts
findings that should be present if it is right. Those predictions are
returned as things to look for -- `predicted_findings` below -- which is
what a clinician does on forming a hypothesis, and what turns a static
ranking into something that can be checked. The RLM can then be asked
specifically about them.

**And the direction raised in discussion: the RLM's own conjectures.** An
RLM reading documents may notice a correlation the graph has no edge for.
That is not noise to discard and not a finding to trust either --
`unvetted_claims` collects them, checked against the graph via
`mechanism_verification`, and each comes back with the grounding the graph
gives it. A claim the graph supports strengthens the differential; one it
cannot support is exactly the material `ConjectureLedger` exists to hold
until confirmations accumulate. Neither is silently promoted, and neither is
silently dropped.

**What this module does not do.** It does not call a model, rank anything
itself, or decide a diagnosis. It moves structured data between two
components that already do those jobs well, and keeps the provenance
straight while doing it -- so a reader downstream can always tell a finding
read from a document from a concept the graph supplied from one the RLM
merely proposed.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..memory.candidate_retrieval import retrieve_candidates
from ..memory.concept_paths import ConceptGraphView, mentioned_concepts
from ..memory.information_content import InformationContentTable
from ..training.mechanism_enumeration import (
    MODE_HYPOTHESES,
    EnumerationOutcome,
    MechanismEnumerator,
)
from .mechanism_verification import MechanismVerification, verify_mechanism

ORIGIN_DOCUMENT = "read_from_document"
ORIGIN_GRAPH = "supplied_by_graph"
ORIGIN_RLM_CONJECTURE = "proposed_by_rlm"


@dataclass(frozen=True)
class VettedClaim:
    """A claim the RLM proposed, with what the graph makes of it."""

    factor: str
    target: str
    mechanism: str
    verification: MechanismVerification

    @property
    def is_grounded(self) -> bool:
        return self.verification.is_grounded

    @property
    def is_candidate_conjecture(self) -> bool:
        """Whether this is new material rather than confirmation or noise.

        The interesting middle: the graph knows both concepts but has no
        connection between them. That is a claim worth holding for
        confirmation, as distinct from one the graph already supports
        (nothing new) or one whose concepts it cannot even resolve (nothing
        to hold).
        """
        return (
            not self.verification.is_grounded
            and self.verification.grounding != "not_checkable"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "factor": self.factor,
            "target": self.target,
            "mechanism": self.mechanism,
            "origin": ORIGIN_RLM_CONJECTURE,
            "is_grounded": self.is_grounded,
            "is_candidate_conjecture": self.is_candidate_conjecture,
            "grounding": self.verification.grounding,
        }


@dataclass
class BridgeResult:
    """What the bridge produced, with every part's origin kept distinct."""

    findings_from_documents: list[str] = field(default_factory=list)
    findings_unresolved: list[str] = field(default_factory=list)
    candidate_conditions: list[str] = field(default_factory=list)
    outcome: EnumerationOutcome | None = None
    predicted_findings: list[str] = field(default_factory=list)
    vetted_claims: list[VettedClaim] = field(default_factory=list)

    @property
    def grounded_claims(self) -> list[VettedClaim]:
        return [claim for claim in self.vetted_claims if claim.is_grounded]

    @property
    def conjectures_for_the_ledger(self) -> list[VettedClaim]:
        """RLM claims the graph cannot support but could -- new material.

        Named for where they belong rather than for what they are, because
        the decision this list feeds is a specific one: these are what
        `ConjectureLedger` holds pending independent confirmation, not
        findings to act on now.
        """
        return [claim for claim in self.vetted_claims if claim.is_candidate_conjecture]

    def as_dict(self) -> dict[str, Any]:
        return {
            "findings_from_documents": list(self.findings_from_documents),
            "findings_unresolved": list(self.findings_unresolved),
            "candidate_conditions": list(self.candidate_conditions),
            "mode": self.outcome.mode if self.outcome else None,
            "hypotheses": [
                {"condition": item.condition, "support": round(item.support, 4),
                 "plausibility": round(item.plausibility, 4), "origin": ORIGIN_GRAPH}
                for item in (self.outcome.hypotheses if self.outcome else [])
            ],
            "open_questions": [
                {"finding": item.finding, "condition": item.condition}
                for item in (self.outcome.open_questions if self.outcome else [])
            ],
            "predicted_findings": list(self.predicted_findings),
            "vetted_claims": [claim.as_dict() for claim in self.vetted_claims],
        }


def findings_from_trajectory(trajectory: Any, graph: ConceptGraphView) -> tuple[list[str], list[str]]:
    """Concepts the graph recognises in what the RLM actually read.

    Reads the trajectory's evidence fragments rather than only its final
    answer: the answer is one sentence, while the fragments are everything
    the run touched, and a differential should be built from what the case
    contains rather than from what one question happened to ask about.

    Returns (resolved, unresolved) rather than only the resolved set,
    because a finding the graph does not recognise is a coverage gap worth
    surfacing, and silently dropping it would make the graph look better
    covered than it is.
    """
    resolved: list[str] = []
    unresolved: list[str] = []
    seen: set[str] = set()

    texts: list[str] = []
    for fragment in trajectory.evidence():
        text = str(fragment.get("text") or fragment.get("content") or "").strip()
        if text:
            texts.append(text)
    if getattr(trajectory, "final_answer", None):
        texts.append(str(trajectory.final_answer))

    for text in texts:
        for concept in mentioned_concepts(text, graph, max_results=12):
            if concept not in seen:
                seen.add(concept)
                resolved.append(concept)

    return resolved, unresolved


def predicted_findings_for(
    outcome: EnumerationOutcome, graph: ConceptGraphView, observed: Sequence[str], *, limit: int = 8
) -> list[str]:
    """What the leading hypotheses predict but the case has not shown yet.

    The graph-to-documents direction: a ranked hypothesis is only useful if
    something can be done with it, and what a clinician does on forming one
    is look for what it predicts. Findings already observed are excluded --
    they confirm nothing new and would crowd out the ones worth asking about.
    """
    if outcome.mode != MODE_HYPOTHESES:
        return []
    already = {item for item in observed}
    predicted: list[str] = []
    for hypothesis in outcome.hypotheses:
        for edge in graph.edges_from(hypothesis.condition):
            if edge.relation.startswith("inverse_"):
                continue
            target = edge.target
            if target not in already and target not in predicted:
                predicted.append(target)
            if len(predicted) >= limit:
                return predicted
    return predicted


def vet_rlm_claims(
    claims: Sequence[tuple[str, str, str]],
    graph: ConceptGraphView,
    *,
    table: InformationContentTable | None = None,
) -> list[VettedClaim]:
    """Check each (factor, target, mechanism) the RLM proposed against the graph.

    Deliberately the same `verify_mechanism` the cross-check and the vetting
    bench use, not a second checker: an RLM's own conjecture deserves exactly
    the scrutiny a model's answer gets, and a separate, gentler path for
    "our own engine's ideas" is how a system starts trusting its own output.
    """
    return [
        VettedClaim(
            factor=factor, target=target, mechanism=mechanism,
            verification=verify_mechanism(graph, factor, target, mechanism, table=table),
        )
        for factor, target, mechanism in claims
    ]


def bridge(
    trajectory: Any,
    graph: ConceptGraphView,
    *,
    enumerator: MechanismEnumerator | None = None,
    table: InformationContentTable | None = None,
    rlm_claims: Sequence[tuple[str, str, str]] = (),
) -> BridgeResult:
    """Run the full bridge: documents to graph, graph back to documents.

    `enumerator` is injected rather than constructed here so a caller can
    supply one already bound to a persistent graph (see
    `memory/graph_store.build_persistent_graph`), which is the configuration
    that matters once the learned layer is non-empty -- constructing one
    internally would silently use only the imported layer.
    """
    result = BridgeResult()
    result.findings_from_documents, result.findings_unresolved = findings_from_trajectory(trajectory, graph)

    if result.findings_from_documents:
        retrieval = retrieve_candidates(result.findings_from_documents, graph)
        result.candidate_conditions = retrieval.condition_names
        result.findings_unresolved.extend(retrieval.unresolved_findings)

        if result.candidate_conditions:
            engine = enumerator or MechanismEnumerator(graph=graph)
            result.outcome = engine.run(result.findings_from_documents, result.candidate_conditions)
            result.predicted_findings = predicted_findings_for(
                result.outcome, graph, result.findings_from_documents
            )

    if rlm_claims:
        result.vetted_claims = vet_rlm_claims(rlm_claims, graph, table=table)

    return result
