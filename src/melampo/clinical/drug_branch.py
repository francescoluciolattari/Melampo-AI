"""The drug branch of the graph of a case (15 November 2026): each drug the patient takes, what its label says,
which of the patient's other drugs its label names, which of the patient's findings it names among its effects.

Everything is read from the local cache (``connectors.drug_sources``), never from a service while a case is
read: the patient's list of drugs is a patient's datum and does not leave the machine.

- ``DrugKnowledge.resolve``: the drug as written -> the AIFA ingredient(s) (by AIC code, commercial name or
  ingredient name in the words) -> the international name (``ingredients.json``, resolved through the ATC code by
  Wikidata and EMA). A name reached by one source only says so; two sources that disagree leave it in doubt; a
  drug no list holds stays unresolved, with its words.
- ``enrich``: one ``DrugProfile`` per resolved ingredient (knowledge), the ATC and INN codes beside the words of
  the medication, the label's version in the case's snapshot.
- ``find_interactions``: a sentence of a label section (interactions, contraindications, warnings, boxed warning)
  that names another drug of the patient, by its international name, as a word. Interactions a label states by
  class ("strong CYP3A4 inhibitors") are not found this way: that needs the class of each drug, a later step.
- ``find_effects``: a sentence of the adverse reactions, warnings or boxed warning that names a finding of the
  patient, by the English label of its concept (the label is in English), with the timing against the drug's
  start: after the start, before it, or unknown. A drug that could explain a finding is a candidate cause for the
  diagnostic path, never a conclusion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..connectors.drug_sources import AifaProduct, DrugCache, ingredient_core, norm
from . import case_graph as cg

INTERACTION_SECTIONS = ("drug_interactions", "contraindications", "boxed_warning", "warnings_and_cautions",
                        "warnings")
EFFECT_SECTIONS = ("adverse_reactions", "boxed_warning", "warnings_and_cautions", "warnings")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
_AIC = re.compile(r"\b(?:AIC\s*:?\s*)?(0\d{8})\b")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(" ".join(str(text).split())) if s.strip()]


def _names_word(sentence: str, name: str) -> bool:
    name = name.strip()
    return bool(name) and re.search(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", sentence, re.IGNORECASE) is not None


@dataclass
class Resolution:
    ingredient: str
    inn: list[str] = field(default_factory=list)
    atc: list[str] = field(default_factory=list)
    basis: str = "unresolved"
    by: str = ""                              # aic, name, ingredient


@dataclass
class DrugKnowledge:
    products: list[AifaProduct]
    ingredients: dict[str, dict[str, Any]]
    cache: DrugCache | None = None
    manifest: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, root: str | Path) -> DrugKnowledge:
        cache = DrugCache(Path(root))
        return cls(products=[AifaProduct(**p) for p in cache.read_json("products.json", [])],
                   ingredients=cache.read_json("ingredients.json", {}), cache=cache,
                   manifest=cache.read_json("manifest.json", {}))

    def _by_ingredient(self, ingredient: str, by: str) -> Resolution:
        entry = self.ingredients.get(ingredient, {})
        return Resolution(ingredient=ingredient, inn=list(entry.get("inn", [])), atc=list(entry.get("atc", [])),
                          basis=entry.get("basis", "unresolved"), by=by)

    def resolve(self, medication: cg.Medication) -> list[Resolution]:
        """The ingredients of a drug as written, each with how it was found."""
        words = f"{medication.name} {medication.words}"
        text = norm(words)
        aics = set(_AIC.findall(words))
        if aics:
            hits = {p.ingredient for p in self.products if p.aic and p.aic.zfill(9) in aics}
            if hits:
                return [self._by_ingredient(i, "aic") for i in sorted(hits)]
        tokens = text.split()
        # a commercial name: the longest product name that the drug's words begin with, or that its name equals
        best, best_len = set(), 0
        for p in self.products:
            name = norm(p.name)
            if not name:
                continue
            size = len(name.split())
            if (tokens[:size] == name.split() or norm(medication.name) == name) and size >= best_len:
                if size > best_len:
                    best, best_len = set(), size
                best.add(p.ingredient)
        if best:
            return [self._by_ingredient(i, "name") for i in sorted(best)]
        # an ingredient named in the words (its core, without salt and hydration words)
        found = set()
        for ingredient in self.ingredients:
            core = ingredient_core(ingredient)
            if core and re.search(rf"(?<![a-z]){re.escape(core)}(?![a-z])", text):
                found.add(ingredient)
        # keep the longest cores only ("acido acetilsalicilico" over "acido")
        cores = {i: ingredient_core(i) for i in found}
        kept = {i for i in found if not any(c != cores[i] and cores[i] in c for c in cores.values())}
        return [self._by_ingredient(i, "ingredient") for i in sorted(kept)]


def enrich(graph: cg.CaseGraph, knowledge: DrugKnowledge) -> dict[str, Any]:
    """Drug profiles for the medications of the case; returns what was resolved and what was not."""
    report = {"resolved": 0, "unresolved": [], "in_doubt": [], "without_label": []}
    have = {(p.medication_id, p.ingredient) for p in graph.drug_profiles}
    for med in graph.medications:
        found = knowledge.resolve(med)
        if not found:
            report["unresolved"].append(med.medication_id)
            continue
        for r in found:
            if (med.medication_id, r.ingredient) in have:
                continue
            if r.basis == "doubt":
                report["in_doubt"].append(med.medication_id)
            inn = r.inn[0] if len(r.inn) == 1 and r.basis != "doubt" else ""
            label = knowledge.cache.label(inn) if (inn and knowledge.cache) else None
            if inn and label is None:
                report["without_label"].append(med.medication_id)
            citations = [{"source": "AIFA liste dei farmaci", "licence": "CC BY 4.0",
                          "version": knowledge.manifest.get("aifa", {}).get("taken", ""), "by": r.by}]
            if label is not None:
                citations.append(label.citation)
                graph.snapshot.sources[f"openFDA label {inn}"] = f"{label.set_id} v{label.version}"
            graph.drug_profiles.append(cg.DrugProfile(
                medication_id=med.medication_id, ingredient=r.ingredient, inn=inn, atc=",".join(r.atc),
                resolution=r.basis, sections=dict(label.sections) if label else {}, citations=citations))
            for code in r.atc:
                if not any(c.code == f"ATC:{code}" for c in med.codes):
                    med.codes.append(cg.Concept("ATC", f"ATC:{code}", r.ingredient))
            if inn and not any(c.system == "INN" and c.label == inn for c in med.codes):
                med.codes.append(cg.Concept("INN", f"INN:{norm(inn).replace(' ', '_')}", inn))
            report["resolved"] += 1
    if knowledge.manifest.get("aifa", {}).get("taken"):
        graph.snapshot.sources["AIFA liste dei farmaci"] = knowledge.manifest["aifa"]["taken"]
    return report


def find_interactions(graph: cg.CaseGraph) -> list[cg.DrugInteraction]:
    """Each sentence of a label that names another drug of the patient, once per pair, section and sentence."""
    out, seen = [], set()
    profiles = [p for p in graph.drug_profiles if p.sections]
    names = [(p.medication_id, p.inn) for p in graph.drug_profiles if p.inn]
    for p in profiles:
        citation = next((c for c in p.citations if c.get("source", "").startswith("openFDA")), {})
        for section in INTERACTION_SECTIONS:
            for text in p.sections.get(section, []):
                for sentence in sentences(text):
                    for other_id, other_inn in names:
                        if other_id == p.medication_id or not _names_word(sentence, other_inn):
                            continue
                        key = (p.medication_id, other_id, section, sentence)
                        if key not in seen:
                            seen.add(key)
                            out.append(cg.DrugInteraction(p.medication_id, other_id, section, sentence, citation))
    graph.drug_interactions = out
    return out


def _timing(med: cg.Medication, finding: cg.Finding) -> str:
    if not med.start or not finding.date:
        return "unknown"
    return "after_start" if finding.date >= med.start else "before_start"


def find_effects(graph: cg.CaseGraph) -> list[cg.DrugEffect]:
    """Each sentence of a label's effects that names an affirmed finding of the patient by its concept's English
    label (at least four letters, as a word)."""
    out, seen = [], set()
    meds = {m.medication_id: m for m in graph.medications}
    findings = [f for f in graph.findings if f.polarity == cg.AFFIRMED and f.concept is not None
                and len(f.concept.label) >= 4]
    for p in graph.drug_profiles:
        if not p.sections:
            continue
        citation = next((c for c in p.citations if c.get("source", "").startswith("openFDA")), {})
        for section in EFFECT_SECTIONS:
            for text in p.sections.get(section, []):
                for sentence in sentences(text):
                    for f in findings:
                        if not _names_word(sentence, f.concept.label):
                            continue
                        key = (p.medication_id, f.finding_id, section)
                        if key not in seen:
                            seen.add(key)
                            out.append(cg.DrugEffect(p.medication_id, f.finding_id, section, sentence,
                                                     _timing(meds[p.medication_id], f), citation))
    graph.drug_effects = out
    return out


def drug_branch(graph: cg.CaseGraph, knowledge: DrugKnowledge) -> dict[str, Any]:
    report = enrich(graph, knowledge)
    report["interactions"] = len(find_interactions(graph))
    report["effects"] = len(find_effects(graph))
    return report


__all__ = ["DrugKnowledge", "Resolution", "drug_branch", "enrich", "find_effects", "find_interactions",
           "sentences"]
