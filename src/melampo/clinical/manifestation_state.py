"""The state of each manifestation of a disease in this patient (14 November 2026): step 3 of "Il grafo del caso".

Four states, and how each is known:

- **present**: a finding of the case with the manifestation's concept (or a concept below it), affirmed and
  certain; or a laboratory value beyond its range in the manifestation's direction;
- **absent**, with its basis: ``stated`` (a finding with the concept, negated), ``measured`` (the value within its
  range, or beyond it the other way), ``silent_in_covering_exam`` (an exam that studies the manifestation's region
  with a modality that shows it, and says nothing of it);
- **doubtful**: the case says it with uncertainty, or two facts disagree (principle of doubt: both are kept as
  evidence, nothing is chosen here);
- **not assessed**: nothing in the case could show it. It weighs nothing in the ranking and is the exam to propose.

The difference between absent and not assessed is the point: a chest CT that does not mention the liver says
nothing about the liver. Silence counts only inside the region the exam studies (``memory.exam_area``: a structure
of the area, or the whole body), and it weighs less than a stated absence (``reasoning.likelihood_ranking``).

Matching is by concept, never by a guess: the same code, or ``below(child, parent)`` when the caller gives the
ontology's is_a (HPO through ``memory.phenotypes`` or any ontology). Without concepts, only the same words
(normalised) match. What the words mean is the readers' work (``FindingOwner`` chooses the concept in the
reading); this module only compares what is already coded.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from . import case_graph as cg

UNCERTAIN = {"uncertain", "possible", "probable", "hedged", "doubt", "doubtful", "uncertain_low", "uncertain_high"}
HYPOTHETICAL = {"hypothetical", "future", "conditional"}
_ABOVE, _BELOW, _WITHIN = "above", "below", "within"


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.lower()).split())


def _same_concept(fact: cg.Concept | None, wanted: cg.Concept | None, below: Callable[[str, str], bool] | None,
                  fact_words: str, wanted_words: str) -> bool:
    if fact is not None and wanted is not None:
        if fact.curie == wanted.curie:
            return True
        return bool(below and below(fact.curie, wanted.curie))
    if fact is None and wanted is None:
        return bool(fact_words.strip()) and _norm(fact_words) == _norm(wanted_words)
    return False


def _covering(graph: cg.CaseGraph, manifestation: cg.Manifestation) -> list[cg.Document]:
    """The imaging documents that study the manifestation's region with a modality that shows it."""
    imaging = [v for v in manifestation.verified_by if v.get("kind") == "imaging"]
    if not imaging:
        return []
    out = []
    for doc in graph.documents:
        if doc.kind not in ("imaging_report", "imaging_study") or not doc.areas:
            continue
        for v in imaging:
            region = v.get("region") or manifestation.region
            modality = str(v.get("modality") or "").upper()
            if not region:
                continue
            if region not in doc.areas and "whole_body" not in doc.areas:
                continue
            if modality and doc.modality and modality != doc.modality.upper():
                continue
            out.append(doc)
            break
    return out


def _deviation(measurement: cg.Measurement, manifestation: cg.Manifestation,
               below: Callable[[str, str], bool] | None) -> str:
    """Which side of a reference row the manifestation is (20 November 2026): ``above`` when it is the row's
    deviation above (``above_hpo``, or its words when neither has a concept), ``below`` likewise, else ""."""
    ref = measurement.reference or {}
    for side, code_key, words_key in ((_ABOVE, "above_hpo", "above_means"), (_BELOW, "below_hpo", "below_means")):
        code, words = str(ref.get(code_key) or ""), str(ref.get(words_key) or "")
        concept = cg.Concept("HPO", code) if code else None
        if (code or words) and _same_concept(concept, manifestation.concept, below, words, manifestation.words):
            return side
    return ""


def _laboratory(graph: cg.CaseGraph, manifestation: cg.Manifestation,
                below: Callable[[str, str], bool] | None = None) -> list[tuple[cg.Measurement, str]]:
    """The values that measure the manifestation, each with what it says: present, absent or doubtful. A laboratory
    value by the exam that verifies it; a measure of a report by the deviation of its reference row (20 November
    2026: "spleen 15 cm" against a verified row whose deviation above is Splenomegaly)."""
    out = []
    for m in graph.measurements:
        side = _deviation(m, manifestation, below)
        if not side:
            continue
        if m.interpretation == side:
            out.append((m, cg.PRESENT))
        elif m.interpretation in (_WITHIN, _ABOVE, _BELOW):
            out.append((m, cg.ABSENT))
        else:
            out.append((m, cg.DOUBTFUL))
    for v in manifestation.verified_by:
        if v.get("kind") != "laboratory":
            continue
        direction = v.get("direction", "")
        code, analyte = v.get("code", ""), _norm(str(v.get("analyte", "")))
        for m in graph.measurements:
            same = (code and m.code is not None and m.code.curie == code) or (analyte and _norm(m.analyte) == analyte)
            if not same:
                continue
            if m.interpretation == direction:
                out.append((m, cg.PRESENT))
            elif m.interpretation in (_WITHIN, _ABOVE, _BELOW):
                out.append((m, cg.ABSENT))
            else:
                out.append((m, cg.DOUBTFUL))
    return out


def assess(manifestation: cg.Manifestation, graph: cg.CaseGraph, *,
           below: Callable[[str, str], bool] | None = None) -> cg.ManifestationState:
    """The state of one manifestation in the case, with the facts it rests on."""
    present, absent, doubtful = [], [], []
    for f in graph.findings:
        if not _same_concept(f.concept, manifestation.concept, below, f.words, manifestation.words):
            continue
        if str(f.temporality).lower() in HYPOTHETICAL:
            continue
        if f.polarity == cg.NEGATED:
            absent.append(f.finding_id)
        elif str(f.certainty).lower() in UNCERTAIN:
            doubtful.append(f.finding_id)
        else:
            present.append(f.finding_id)
    measured = _laboratory(graph, manifestation, below)
    present += [m.measurement_id for m, s in measured if s == cg.PRESENT]
    absent_measured = [m.measurement_id for m, s in measured if s == cg.ABSENT]
    doubtful += [m.measurement_id for m, s in measured if s == cg.DOUBTFUL]
    mid = manifestation.manifestation_id
    if present and (absent or absent_measured):
        return cg.ManifestationState(mid, cg.DOUBTFUL, "facts_disagree", present + absent + absent_measured,
                                     "the case says it is there and says it is not: both kept")
    if present:
        stated = any(p in {f.finding_id for f in graph.findings} for p in present)
        return cg.ManifestationState(mid, cg.PRESENT, cg.STATED if stated else cg.MEASURED, present + doubtful)
    if doubtful:
        return cg.ManifestationState(mid, cg.DOUBTFUL, "stated_with_uncertainty", doubtful + absent + absent_measured)
    if absent:
        return cg.ManifestationState(mid, cg.ABSENT, cg.STATED, absent + absent_measured)
    if absent_measured:
        return cg.ManifestationState(mid, cg.ABSENT, cg.MEASURED, absent_measured)
    covering = _covering(graph, manifestation)
    if covering:
        return cg.ManifestationState(mid, cg.ABSENT, cg.SILENT, [],
                                     "not mentioned in " + ", ".join(d.document_id for d in covering))
    return cg.ManifestationState(mid, cg.NOT_ASSESSED, "", [], "nothing in the case could show it")


def assess_disease(disease: cg.Disease, graph: cg.CaseGraph, *,
                   below: Callable[[str, str], bool] | None = None) -> list[cg.ManifestationState]:
    disease.states = [assess(m, graph, below=below) for m in disease.manifestations]
    return disease.states


def assess_case(graph: cg.CaseGraph, *, below: Callable[[str, str], bool] | None = None) -> dict[str, Any]:
    """Every disease of every branch assessed; returns the counts of the states."""
    counts = {s: 0 for s in cg.STATES}
    for d in graph.diseases:
        for s in assess_disease(d, graph, below=below):
            counts[s.state] += 1
    return counts


def unexplained_findings(graph: cg.CaseGraph, *, below: Callable[[str, str], bool] | None = None) -> list[cg.Finding]:
    """The affirmed findings no manifestation of any disease in the graph accounts for: what the diagnostic path
    takes to the three channels to look for candidates."""
    out = []
    for f in graph.findings:
        if f.polarity == cg.NEGATED:
            continue
        explained = any(_same_concept(f.concept, m.concept, below, f.words, m.words)
                        for d in graph.diseases for m in d.manifestations)
        if not explained:
            out.append(f)
    return out


__all__ = ["assess", "assess_case", "assess_disease", "unexplained_findings"]
