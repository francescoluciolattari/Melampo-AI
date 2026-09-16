"""Assemble the pieces into one working chain, with nothing left unconnected.

This project repeatedly built a correct, tested module and then never called
it -- seven times over, by the end of the investigation that produced this
file. `MechanismEnumerator`, `ConjectureLedger`, `HypothesisYield`, the
guided expansion fallback, and the RLM-graph bridge all existed in that
state. This module is the assembly point, and its existence is the answer:
one place where the wiring lives, so "is it connected?" has a single file to
check rather than a search across the codebase.

**What assembly means here, and what it does not.** Every component keeps
its own behaviour unchanged; none is modified to fit. This composes them:
builds the graph from its persistent layers, binds an enumerator to that
graph, supplies the candidate retrieval the enumerator was missing, and
routes what the RLM reads through the bridge. The judgement calls -- which
model, whether the cross-check is always on -- stay with the caller, because
they are decisions about the system rather than about how its parts fit.

**The graph is built once, not per request.** 285,598 edges take about two
and a half seconds to parse from the HPO release; doing that on every case
would dominate the latency of the thing it serves. `DiagnosticAssembly` is
built at startup and holds the graph for the process's lifetime, which is
also what makes the learned layer meaningful -- an edge promoted mid-session
is visible to the next case without a reload.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..memory.candidate_retrieval import retrieve_candidates
from ..memory.concept_paths import ConceptEdge, ConceptGraphView
from ..memory.graph_store import (
    LearnedEdgeStore,
    build_persistent_graph,
    learned_provenance,
)
from ..memory.information_content import InformationContentTable
from ..training.conjecture_ledger import ConjectureLedger
from ..training.hypothesis_yield import HypothesisYieldModel
from ..training.mechanism_enumeration import EnumerationOutcome, MechanismEnumerator
from .rlm_graph_bridge import BridgeResult, bridge


@dataclass
class DiagnosticAssembly:
    """Every component, bound to one graph, ready to run a case.

    Built once at startup. Holding the graph for the process's lifetime is
    what makes the learned layer worth having: an edge promoted after this
    case is visible to the next one without a reload.
    """

    graph: ConceptGraphView
    store: LearnedEdgeStore
    enumerator: MechanismEnumerator
    ledger: ConjectureLedger
    yield_model: HypothesisYieldModel
    table: InformationContentTable
    learned_edge_count: int = 0
    # Optional: the vector-evolution space (training/vector_evolution_engine.py)
    # for surfacing non-obvious correlations between cases. None by default,
    # matching every other optional component's graceful-degradation contract
    # -- a deployment without it loses the correlation-discovery capability,
    # not the rest of the assembly.
    vector_space: Any = None

    def run_case(
        self,
        trajectory: Any,
        *,
        rlm_claims: Sequence[tuple[str, str, str]] = (),
        record_conjectures: bool = True,
    ) -> "CaseResult":
        """Take what the RLM read and produce a differential, recording what it learned.

        `record_conjectures` defaults to True because a hypothesis's leaps are
        worth recording even on a case that resolves cleanly -- the ledger's
        value comes from accumulating across many cases, and recording only
        the interesting ones would bias what eventually gets promoted toward
        the cases someone thought were interesting at the time.
        """
        bridge_result = bridge(
            trajectory, self.graph, enumerator=self.enumerator, table=self.table, rlm_claims=rlm_claims
        )

        recorded = 0
        if record_conjectures and bridge_result.outcome:
            for hypothesis in bridge_result.outcome.hypotheses:
                recorded += len(
                    self.ledger.record_from_hypothesis(hypothesis, getattr(trajectory, "case_id", "unknown"))
                )

        return CaseResult(bridge_result=bridge_result, conjectures_recorded=recorded)

    def record_hypothesis_vector(self, case_id: str, hypothesis: str, embedder: Any, *, now: float | None = None) -> Any:
        """Embed and record a hypothesis in the vector-evolution space, if one is configured.

        Does nothing and returns None without a configured `vector_space` --
        this capability is additive, and a caller running `record_hypothesis_vector`
        unconditionally after every case must not need to check first whether
        the space exists.
        """
        if self.vector_space is None:
            return None
        vector = tuple(embedder.embed(hypothesis))
        return self.vector_space.update(case_id, hypothesis, vector, now=now)

    def cross_case_correlations(self, *, min_overlap: float = 0.85) -> list[Any]:
        """Hypotheses from different cases whose vectors now overlap above threshold.

        An empty list, not an error, when no vector space is configured --
        the same posture every optional tier in this project takes toward
        its own absence.
        """
        if self.vector_space is None:
            return []
        return self.vector_space.find_cross_case_correlations(min_overlap=min_overlap)

    def promote_confirmed(
        self,
        *,
        min_confirmations: int = 3,
        description_store: Any = None,
        description_extractor: Any = None,
        description_store_path: Any = None,
    ) -> list[ConceptEdge]:
        """Write conjectures that have earned promotion into the persistent store.

        Deliberately a separate call rather than something `run_case` does:
        promotion is a change to the shared knowledge base, and it should
        happen when someone decides to run it, not as a side effect of
        answering one case. Returns the edges written so a caller can see
        what changed rather than only that something did.

        ``description_store``/``description_extractor`` are optional and
        both must be supplied for this to do anything: every newly promoted
        edge -- whether the conjecture came from the Dream Engine's offline
        exploration or from a confirmed RLM hypothesis -- gets a tier-3
        description built and persisted for its target concept, if one does
        not already exist. Without both, promotion behaves exactly as
        before; this is additive, not a new requirement on every caller.
        """
        promoted: list[ConceptEdge] = []
        for record in self.ledger.records.values():
            if not record.is_promotable(min_confirmations):
                continue
            edge = record.to_edge()
            edge = ConceptEdge(
                source=edge.source,
                relation=edge.relation,
                target=edge.target,
                weight=edge.weight,
                provenance=learned_provenance(
                    record.conjecture.origin_case, len(record.confirmed_in)
                ),
                lower=edge.lower,
                upper=edge.upper,
            )
            promoted.append(edge)

        if promoted:
            self.store.append_many(promoted)
            self.learned_edge_count += len(promoted)

        if promoted and description_store is not None and description_extractor is not None:
            self._describe_promoted_concepts(promoted, description_store, description_extractor, description_store_path)

        return promoted

    def _describe_promoted_concepts(
        self, promoted: list[ConceptEdge], description_store: Any, description_extractor: Any, store_path: Any
    ) -> None:
        """Build and persist a description for every promoted edge's concepts that lacks one.

        Extracted from the edge's own justification -- what was confirmed,
        by how many cases -- rather than left unexplained. A concept that
        entered the graph through promotion has exactly as much right to a
        tier-3 description as one that entered through the curated
        literature; the only difference is where the source text comes from.
        """
        for edge in promoted:
            for concept in (edge.source, edge.target):
                if description_store.get(concept) is not None:
                    continue
                justification = (
                    f"{edge.source} {edge.relation.replace('_', ' ')} {edge.target}, "
                    f"confirmed by independent cases (provenance: {edge.provenance})."
                )
                structure = description_extractor(justification)
                if structure.is_empty:
                    continue
                description_store.add(concept, structure)
                if store_path is not None:
                    description_store.append_to(store_path, concept)


@dataclass
class CaseResult:
    """One case's output, and what it contributed back."""

    bridge_result: BridgeResult
    conjectures_recorded: int = 0

    @property
    def outcome(self) -> EnumerationOutcome | None:
        return self.bridge_result.outcome

    def as_dict(self) -> dict[str, Any]:
        return {**self.bridge_result.as_dict(), "conjectures_recorded": self.conjectures_recorded}


def assemble(
    imported_edges: Sequence[ConceptEdge],
    learned_store_path: Path | str,
    *,
    frequencies: dict[str, float] | None = None,
) -> DiagnosticAssembly:
    """Build the whole chain from its persistent layers.

    `frequencies` supplies Information Content where real counts exist;
    without them, IC falls back to the graph's own structure, which is the
    documented intrinsic-IC path rather than a degraded one -- so a caller
    with no frequency data still gets specificity weighting rather than
    none.
    """
    store = LearnedEdgeStore(learned_store_path)
    graph, load_report = build_persistent_graph(imported_edges, store)

    table = (
        InformationContentTable.from_frequencies(frequencies)
        if frequencies
        else InformationContentTable.from_graph_structure(graph)
    )

    return DiagnosticAssembly(
        graph=graph,
        store=store,
        enumerator=MechanismEnumerator(graph=graph),
        ledger=ConjectureLedger(),
        yield_model=HypothesisYieldModel(),
        table=table,
        learned_edge_count=len(load_report.edges),
    )


def candidate_conditions_for(findings: Sequence[str], graph: ConceptGraphView) -> list[str]:
    """The candidate list `DreamTrainer` needs but has no way to produce.

    `DreamTrainer._enumerated` returns None unless the case context already
    carries `candidate_conditions`, and nothing in the pipeline put them
    there -- which is why the trainer fell through to rehearsal labels on
    every real case. Exposed here so a caller populating that context has
    one obvious function to call rather than reimplementing the retrieval.
    """
    return retrieve_candidates(findings, graph).condition_names


def dream_context_for(findings: Sequence[str], graph: ConceptGraphView, **extra: Any) -> dict[str, Any]:
    """A case context with the candidates filled in, ready for DreamTrainer.

    The concrete fix for the hook that was never valorised: pass this as the
    trainer's `case_context` and `_enumerated` finds what it needs instead of
    falling through to placeholder labels.
    """
    return {
        "findings": list(findings),
        "candidate_conditions": candidate_conditions_for(findings, graph),
        **extra,
    }
