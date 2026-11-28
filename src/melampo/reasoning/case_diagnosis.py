"""The diagnostic path on the case graph (20 November 2026): the profiles, the states, the candidates, the ranking.

The graph of the case (``clinical.case_graph``) holds the patient's facts and the diseases already named; the
profiles (``memory.profile_library``) hold what each disease shows, from the sources. This module joins them, in
the order of "Il grafo del caso":

1. **Profiles to the named diseases.** Each present or hypothesized disease takes its profile's manifestations
   (knowledge, with their citations), found by concept or by name; a disease without a profile stays in the graph
   and says so (``no_profile``): it is not ranked on nothing.
2. **Candidates, from the diagnostic path only.** The candidate branch is empty when the case enters; it is filled
   here, through ``CaseGraph.add_candidate``, each candidate with what proposed it:
   * the rapid part of the intuition engine: the differential the pipeline computes from the findings
     (``proposals``: the concept graph's candidates, ``memory.differential_ranking``), when the library has their
     profile;
   * the lateral part (``reasoning.lateral_intuition``, 28 November 2026): every profile that accounts for an
     observation, and pairs when the best single leaves salient observations unexplained, weighed against the
     whole case; only what the guard lets through, as a hypothesis to exclude. It replaces the profiles that had
     at least two manifestations present and one unexplained (``MIN_PRESENT``, ``MAX_CANDIDATES``: numbers chosen
     by hand, removed).
3. **The states.** Every manifestation of every disease is present, absent (stated, measured, silent in a covering
   exam), doubtful or not assessed in this patient (``clinical.manifestation_state``).
4. **The ranking.** Hypothesized and candidate diseases are ranked together by likelihood ratios, absences
   included (``reasoning.likelihood_ranking``), the background taken from the whole library; since 28 November 2026
   each observation of the case a disease does not account for weighs as LIRICAL's no match. Present diseases are
   not ranked: they are already established; what they do not explain is what the candidates are for.

What comes back is read manifestation by manifestation: each disease with its contributions; the doubts (principle
of doubt: kept, never resolved here); the exams to propose (the not-assessed manifestations of the first diseases,
with what would show them); the findings nothing explains; the version of the knowledge used. Since 28 November
2026 it is the diagnosis of the pipeline (``diagnostic_orchestrator.from_case_diagnosis``); the October path
(``DifferentialEngine``) stays in the result for comparison only, until the evaluation of December 2026.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable, Iterable
from typing import Any

from ..clinical import case_graph as cg
from ..clinical import manifestation_state as ms
from ..memory.profile_library import ProfileLibrary
from . import lateral_intuition as li
from . import likelihood_ranking as lr

EXAMS_FOR = 3                # the first diseases whose not-assessed manifestations become the exams to propose


def _manifestations(profile: cg.Disease) -> list[cg.Manifestation]:
    return [copy.deepcopy(m) for m in profile.manifestations]


def _concept_of(profile: cg.Disease) -> cg.Concept | None:
    if profile.concept is not None:
        return profile.concept
    if profile.disease_id.startswith("MONDO:"):
        return cg.Concept("MONDO", profile.disease_id, profile.words)
    return None


def _ids_in_graph(graph: cg.CaseGraph) -> set[str]:
    return {d.concept.curie for d in graph.diseases if d.concept is not None and d.concept.code}


def attach_profiles(graph: cg.CaseGraph, library: ProfileLibrary) -> dict[str, str]:
    """Each present or hypothesized disease without manifestations takes its profile's; returns how each was
    found (``concept``, ``name``) or why not (``no_profile``, ``ambiguous_name``)."""
    how: dict[str, str] = {}
    for disease in graph.diseases:
        if disease.branch == cg.CANDIDATE_DISEASES:
            continue
        profile, found = library.find(concept=disease.concept, words=disease.words)
        how[disease.disease_id] = found
        if profile is None:
            continue
        if not disease.manifestations:
            disease.manifestations = _manifestations(profile)
        if disease.concept is None:
            disease.concept = _concept_of(profile)
    return how


def _candidate(profile: cg.Disease, citation: dict[str, Any]) -> cg.Disease:
    digest = hashlib.sha256(profile.disease_id.encode()).hexdigest()[:12]
    return cg.Disease(disease_id=f"cand:{digest}", branch=cg.CANDIDATE_DISEASES, words=profile.words,
                      origin=cg.DIAGNOSTIC_PATH, concept=_concept_of(profile),
                      manifestations=_manifestations(profile), citations=[citation])


def add_proposed(graph: cg.CaseGraph, library: ProfileLibrary, proposals: Iterable[dict[str, Any]]) -> list[str]:
    """The differential's candidates that have a profile and are not in the graph yet; returns the notes of those
    left out."""
    notes = []
    for rank_, p in enumerate(proposals, 1):
        name = str(p.get("condition") or p.get("name") or p.get("disease") or "").strip()
        if not name:
            continue
        profile, found = library.find(words=name)
        if profile is None:
            notes.append(f"proposal {name!r}: {found}")
            continue
        concept = _concept_of(profile)
        if concept is not None and concept.curie in _ids_in_graph(graph):
            continue
        graph.add_candidate(_candidate(profile, {"proposed_by": "differential_from_findings", "name": name,
                                                 "rank": rank_,
                                                 "score": p.get("score", p.get("specificity_score"))}))
    return notes


def add_lateral(graph: cg.CaseGraph, library: ProfileLibrary, lateral: dict[str, Any]) -> None:
    """The profiles of the hypotheses the lateral part lets through, as candidates: each with what proposed it, what
    it would explain, and that it is a hypothesis to exclude."""
    in_graph = _ids_in_graph(graph)
    for item in lateral.get("accepted", []):
        for pid in item["diseases"]:
            profile = library.profiles.get(pid)
            if profile is None:
                continue
            concept = _concept_of(profile)
            if (concept is not None and concept.curie in in_graph) or pid in in_graph:
                continue
            graph.add_candidate(_candidate(profile, {
                "proposed_by": li.LATERAL, "kind": item["kind"], "from": item["from"],
                "together_with": [d for d in item["diseases"] if d != pid],
                "explains_salient": item["explains_salient"], "energy": item["energy"],
                "delta_energy": item["delta_energy"], "insight": item["insight"], "status": item["status"]}))
            in_graph = _ids_in_graph(graph) | {pid}


def _exams(disease: cg.Disease, ranked: lr.Ranked) -> list[dict[str, Any]]:
    by_id = {m.manifestation_id: m for m in disease.manifestations}
    out = []
    for mid in ranked.not_assessed:
        m = by_id.get(mid)
        if m is None:
            continue
        out.append({"disease": disease.words, "manifestation": m.words,
                    "concept": m.concept.curie if m.concept else "", "typicality": m.typicality,
                    "frequency": m.frequency, "verified_by": m.verified_by, "region": m.region})
    weight = {"hallmark": 0, "excluding": 0, "common": 1, "": 2, "rare": 3}
    return sorted(out, key=lambda e: (weight.get(e["typicality"], 2), -(e["frequency"] or 0)))


def diagnose(graph: cg.CaseGraph, library: ProfileLibrary, *, proposals: Iterable[dict[str, Any]] = (),
             below: Callable[[str, str], bool] | None = None) -> dict[str, Any]:
    """The diagnostic path on one case's graph; the graph is completed in place (profiles, candidates, states)."""
    graph.snapshot.sources["disease_profiles"] = library.version
    found = attach_profiles(graph, library)
    notes = add_proposed(graph, library, proposals)
    lateral = li.lateral_cycle(graph, library, below=below)
    add_lateral(graph, library, lateral)
    counts = ms.assess_case(graph, below=below)
    ranked_branches = (cg.HYPOTHESIZED_DISEASES, cg.CANDIDATE_DISEASES)
    to_rank = [d for d in graph.diseases if d.branch in ranked_branches and d.manifestations]
    observed, masked = lateral.pop("observed"), lateral.pop("masked_keys")
    ranked = lr.rank(to_rank, profiles=library.profiles.values(), observed=observed, masked=masked)
    by_id = {d.disease_id: d for d in graph.diseases}
    exams = [e for r in ranked[:EXAMS_FOR] for e in _exams(by_id[r.disease_id], r)]
    still_unexplained = ms.unexplained_findings(graph, below=below)
    return {
        "knowledge": {"disease_profiles": library.version, "files": library.files,
                      "profiles": len(library.profiles)},
        "named": [{"disease": d.disease_id, "words": d.words, "branch": d.branch, "profile": found.get(d.disease_id),
                   "states": _counts(d)} for d in graph.diseases if d.branch != cg.CANDIDATE_DISEASES],
        "candidates": [{"disease": d.disease_id, "words": d.words, "proposed_by": d.citations}
                       for d in graph.branch(cg.CANDIDATE_DISEASES)],
        "ranking": [{**r.as_dict(), "origin": by_id[r.disease_id].origin} for r in ranked],
        "doubts": [{"disease": r.words, "manifestations": r.doubts} for r in ranked if r.doubts],
        "exams_to_propose": exams,
        "unexplained_findings": [{"finding": f.finding_id, "words": f.words, "document": f.document}
                                 for f in still_unexplained],
        "lateral": lateral,
        "states": counts,
        "notes": notes + [f"{d}: {how}" for d, how in found.items() if how not in ("concept", "name")],
        "graph_problems": graph.validate(),
    }


def _counts(d: cg.Disease) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in d.states:
        out[s.state] = out.get(s.state, 0) + 1
    return out


def hpo_below(path: str) -> Callable[[str, str], bool] | None:
    """``below(child, parent)`` from an hp.obo file (is_a), or None when there is no file."""
    if not path:
        return None
    from functools import lru_cache
    from pathlib import Path

    from ..memory import disease_profiles as dp

    terms = dp.Terms.from_obo(Path(path).read_text("utf-8", errors="replace").splitlines())
    above = lru_cache(maxsize=None)(lambda term: frozenset(terms.ancestors(term)))
    return lambda child, parent: parent in above(child)


__all__ = ["EXAMS_FOR", "add_lateral", "add_proposed",
           "attach_profiles", "diagnose", "hpo_below"]
