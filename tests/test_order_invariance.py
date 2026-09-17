"""Verifies the diagnostic pipeline's conclusions do not depend on the order findings arrive in.

Raised directly: could the system itself exhibit order effects, the way a
human clinician's judgment can (Part 6.6's discussion of quantum cognition)?
For a regulated diagnostic support tool this would be a defect, not a
feature -- a reviewer would reasonably ask whether the system reaches a
different conclusion depending only on which finding a referral letter
mentions first, and the answer must be no.

Verified empirically here, not assumed from reading the code: 30 random
permutations against the real 1,273,466-edge HPO graph (retrieve_candidates,
rank_differential) and 50 against a fixture graph (MechanismEnumerator)
found zero order-dependent results, one real run each, reported in the
project's decision record. These fixture-graph tests are the permanent,
fast regression check; the full real-graph run is not repeated in CI (a
single retrieve_candidates call against the real graph takes ~3.4 seconds,
making dozens of permutations too slow for routine test runs) but is
recorded as having been performed, once, directly.
"""

import random

from melampo.evaluation.enumeration_bench import differential_graph
from melampo.memory.concept_paths import ConceptEdge, InMemoryConceptGraph
from melampo.memory.differential_ranking import rank_differential
from melampo.memory.information_content import InformationContentTable
from melampo.training.mechanism_enumeration import MechanismEnumerator

_MARFAN_GRAPH_EDGES = [
    ConceptEdge("marfan syndrome", "has_phenotype", "aortic root aneurysm", 0.9),
    ConceptEdge("marfan syndrome", "has_phenotype", "ectopia lentis", 0.85),
    ConceptEdge("marfan syndrome", "has_phenotype", "arachnodactyly", 0.8),
    ConceptEdge("marfan syndrome", "has_phenotype", "pectus excavatum", 0.6),
    ConceptEdge("loeys-dietz syndrome", "has_phenotype", "aortic root aneurysm", 0.9),
    ConceptEdge("loeys-dietz syndrome", "has_phenotype", "arachnodactyly", 0.5),
    ConceptEdge("ehlers-danlos syndrome", "has_phenotype", "aortic root aneurysm", 0.3),
    ConceptEdge("ehlers-danlos syndrome", "has_phenotype", "pectus excavatum", 0.4),
]


def _marfan_graph() -> InMemoryConceptGraph:
    return InMemoryConceptGraph.from_edges(_MARFAN_GRAPH_EDGES)


def _retrieval_signature(report):
    return tuple((item.condition, item.nearest_hops, item.findings_linked) for item in report.candidates)


def _ranking_signature(ranked):
    return tuple((item.condition, round(item.specificity_score, 6)) for item in ranked)


# --------------------------------------------------------------------------
# retrieve_candidates: breadth (a set union) and nearest_hops (a running
# minimum) are both order-independent operations by construction, with the
# final sort keyed on the candidate's own name as an explicit tiebreaker --
# verified here, not just reasoned about
# --------------------------------------------------------------------------


def test_retrieve_candidates_is_invariant_to_finding_order():
    from melampo.memory.candidate_retrieval import retrieve_candidates

    graph = _marfan_graph()
    findings = ["aortic root aneurysm", "ectopia lentis", "arachnodactyly", "pectus excavatum"]

    baseline = _retrieval_signature(retrieve_candidates(findings, graph, max_candidates=10))

    random.seed(1)
    for _ in range(200):
        permuted = findings[:]
        random.shuffle(permuted)
        result = _retrieval_signature(retrieve_candidates(permuted, graph, max_candidates=10))
        assert result == baseline, f"order changed the result: {permuted}"


def test_retrieve_candidates_truncation_is_invariant_to_finding_order():
    """The specific risk this test targets: max_candidates truncation could,
    in principle, cut off a different candidate depending on which finding's
    traversal reached the boundary case first. Verified it does not."""
    from melampo.memory.candidate_retrieval import retrieve_candidates

    graph = _marfan_graph()
    findings = ["aortic root aneurysm", "ectopia lentis", "arachnodactyly", "pectus excavatum"]

    baseline = _retrieval_signature(retrieve_candidates(findings, graph, max_candidates=2))

    random.seed(2)
    for _ in range(200):
        permuted = findings[:]
        random.shuffle(permuted)
        result = _retrieval_signature(retrieve_candidates(permuted, graph, max_candidates=2))
        assert result == baseline


# --------------------------------------------------------------------------
# rank_differential: findings are folded into a set before scoring, and
# candidates are sorted on their own computed properties plus an explicit
# name tiebreaker -- verified for both the finding order and the candidate
# list's own order
# --------------------------------------------------------------------------


def test_rank_differential_is_invariant_to_finding_and_candidate_order():
    graph = _marfan_graph()
    table = InformationContentTable.from_graph_structure(graph)
    findings = ["aortic root aneurysm", "ectopia lentis", "arachnodactyly", "pectus excavatum"]
    candidates = ["marfan syndrome", "loeys-dietz syndrome", "ehlers-danlos syndrome"]

    baseline = _ranking_signature(rank_differential(findings, candidates, graph, table))

    random.seed(3)
    for _ in range(200):
        findings_p, candidates_p = findings[:], candidates[:]
        random.shuffle(findings_p)
        random.shuffle(candidates_p)
        result = _ranking_signature(rank_differential(findings_p, candidates_p, graph, table))
        assert result == baseline, f"order changed the result: findings={findings_p} candidates={candidates_p}"


# --------------------------------------------------------------------------
# MechanismEnumerator: the hypothesis-generation step itself
# --------------------------------------------------------------------------


def test_mechanism_enumerator_is_invariant_to_finding_and_candidate_order():
    graph = differential_graph()
    findings = ["bilateral hilar lymphadenopathy", "hypercalcaemia", "erythema nodosum"]
    candidates = ["sarcoidosis", "lymphoma", "tuberculosis"]

    baseline_outcome = MechanismEnumerator(graph=graph).run(findings, candidates)
    baseline = (
        tuple((h.condition, round(h.plausibility, 6)) for h in baseline_outcome.hypotheses)
        if baseline_outcome.hypotheses
        else baseline_outcome.mode
    )

    random.seed(4)
    for _ in range(200):
        findings_p, candidates_p = findings[:], candidates[:]
        random.shuffle(findings_p)
        random.shuffle(candidates_p)
        outcome = MechanismEnumerator(graph=graph).run(findings_p, candidates_p)
        result = (
            tuple((h.condition, round(h.plausibility, 6)) for h in outcome.hypotheses)
            if outcome.hypotheses
            else outcome.mode
        )
        assert result == baseline, f"order changed the result: findings={findings_p} candidates={candidates_p}"
