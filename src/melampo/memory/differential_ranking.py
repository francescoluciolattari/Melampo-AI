"""Rank candidate diseases by how specifically their known phenotypes match the case.

Answers a direct question: instead of a separate "diagnostic discrimination"
data source, can candidates be discriminated by observing findings and
comparing them against what HPO already associates with each suspected
disease? Yes -- this is not a workaround, it is the established method in
clinical bioinformatics (information-content-weighted phenotype semantic
similarity, the approach behind tools such as Phenomizer), and it is exactly
what `information_content.py`'s IC table and the has_phenotype graph were
already built to support. There is no separate "diagnostic differential"
dataset to go looking for: HPO's disease-phenotype associations *are* that
source, used correctly.

**Why specificity, not raw overlap.** `candidate_retrieval.py` already
ranks candidates by how many observed findings they share -- breadth, chosen
there because retrieval only needs to gather plausible candidates cheaply.
Discrimination needs more: two candidates sharing five findings are not
equally well supported if one of those five is "fatigue" (present in
thousands of diseases) and the other's five are all rare, specific findings.
An observed finding present in only a handful of diseases is far more
discriminating than one present in most of them, and IC -- already computed
from exactly this graph -- is precisely that weight.

**What this still does not replace.** Phenotype similarity discriminates
among diseases HPO already associates with observable findings; it says
nothing about *mechanism* ("why does this happen"), which is what the
literature connectors (`europe_pmc.py`, `clinical_trials.py`) and the
concept graph's causal edges exist for. The two are complementary layers,
not competing answers to the same question -- similarity narrows the
differential, mechanism explains the winner.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .concept_paths import ConceptGraphView, normalise_concept, resolve_concept
from .information_content import InformationContentTable

RELATION_HAS_PHENOTYPE = "has_phenotype"


@dataclass(frozen=True)
class DifferentialCandidate:
    """One candidate disease, with how specifically its known profile matches the case."""

    condition: str
    matched_findings: tuple[str, ...]
    unmatched_findings: tuple[str, ...]
    specificity_score: float
    profile_size: int

    @property
    def coverage(self) -> float:
        """Fraction of the case's own findings this candidate accounts for.

        Kept separate from `specificity_score`: a candidate explaining every
        observed finding with common ones and a candidate explaining half of
        them with rare ones can score similarly on specificity alone, and a
        reader needs both numbers to tell which situation they are looking
        at.
        """
        total = len(self.matched_findings) + len(self.unmatched_findings)
        return len(self.matched_findings) / total if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "matched_findings": list(self.matched_findings),
            "unmatched_findings": list(self.unmatched_findings),
            "specificity_score": round(self.specificity_score, 4),
            "coverage": round(self.coverage, 4),
            "profile_size": self.profile_size,
        }


def rank_differential(
    findings: Sequence[str],
    candidates: Sequence[str],
    graph: ConceptGraphView,
    table: InformationContentTable,
    *,
    relation: str = RELATION_HAS_PHENOTYPE,
) -> list[DifferentialCandidate]:
    """Rank candidates by the summed specificity of the findings they actually share.

    ``relation`` defaults to HPO's own `has_phenotype`, the relation this
    module was built around -- but is a parameter, not a hard-coded
    constant, since other graphs in this project use their own vocabulary
    for the same kind of edge (the enumeration bench's fixture, for
    instance, uses `manifests_as`). A caller comparing against a non-HPO
    graph passes the relation that graph actually uses.

    Deliberately not normalised by the candidate's own profile size (dividing
    by how many phenotypes a disease has in total). That would reward a
    sparsely-annotated rare disease purely for having few recorded
    phenotypes, which is a curation artefact, not evidence the disease fits
    the case better -- `profile_size` is reported instead, so a reader can
    see and judge that for themselves rather than having it silently folded
    into the ranking.
    """
    resolved_findings = [item for item in (resolve_concept(f, graph) for f in findings) if item is not None]
    finding_set = {normalise_concept(item) for item in resolved_findings}

    ranked: list[DifferentialCandidate] = []
    for candidate in candidates:
        profile = {
            normalise_concept(edge.target)
            for edge in graph.edges_from(candidate)
            if edge.relation == relation
        }
        matched = finding_set & profile
        unmatched = finding_set - profile
        score = sum(table.value(item) for item in matched)
        ranked.append(
            DifferentialCandidate(
                condition=candidate,
                matched_findings=tuple(sorted(matched)),
                unmatched_findings=tuple(sorted(unmatched)),
                specificity_score=score,
                profile_size=len(profile),
            )
        )

    ranked.sort(key=lambda item: (-item.specificity_score, -item.coverage, item.condition))
    return ranked
