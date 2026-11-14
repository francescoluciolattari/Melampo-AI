"""The profile of a disease from the sources (15 November 2026): step 5 of "Il grafo del caso", the pilot.

A profile is a ``clinical.case_graph.Disease`` whose manifestations each carry the words of the source, the
concept when an ontology has it, the frequency or typicality when the source says it, and the citation (source,
identifier, licence, the sentence). It is built cheapest first:

1. **Structured sources, no model.** The HPO annotations (``phenotype.hpoa``: frequency classes, n/m, percentages,
   NOT annotations as "excluding"), through MONDO's cross-references; the Disease Ontology's ``has_symptom``
   axioms (``doid.owl``), named through the Symptom Ontology.
2. **Text, only the sections that describe the disease.** A review under CC BY or CC0 from Europe PMC (its
   clinical presentation, diagnosis and imaging sections, at most ``SECTION_CHARACTERS``). StatPearls is CC
   BY-NC-ND: only on request, for research, and its licence travels with every citation taken from it.
3. **Two readers, then the words, then two judges.** The reading pair (Nemotron 3 Ultra, Gemma 4 31B) lists the
   manifestations with the sentence each comes from. A sentence that is not in the text is refused: an invented
   quote. A sentence that names the manifestation (its words, or the name or a synonym of its HPO term) is
   verified without a model. Only the rest goes to the two judges (Muse Spark Open with reasoning, Claude
   Sonnet 5), who must agree; when they do not, the link stays out of the profile, listed as a doubt.

Typicality comes from the words of the sentence ("pathognomonic", "most common", "rare", ...) or from the two
readers when they agree; otherwise it is left empty and the manifestation is not weighed
(``reasoning.likelihood_ranking``).
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..clinical import case_graph as cg

SECTION_CHARACTERS = 24000
SECTION_WORDS = re.compile(r"clinical|present|symptom|sign|manifest|feature|diagnos|imaging|radiolog|evaluat|"
                           r"examination|history|laborator|finding|course", re.IGNORECASE)
# HPO frequency classes (HP:0040279 subtree): the middle of each band
HPO_FREQUENCY = {"HP:0040280": 1.0, "HP:0040281": 0.895, "HP:0040282": 0.545, "HP:0040283": 0.17,
                 "HP:0040284": 0.025, "HP:0040285": 0.0}
TYPICALITY_WORDS = (
    ("hallmark", re.compile(r"pathognomonic|hallmark|characteristic|cardinal|classic(al)?|typical(ly)?|"
                            r"almost (all|always)|nearly (all|always)|virtually all|most (common|frequent)|"
                            r"majority of", re.IGNORECASE)),
    ("rare", re.compile(r"\brare(ly)?\b|uncommon(ly)?|infrequent(ly)?|occasional(ly)?|seldom|minority of", re.IGNORECASE)),
    ("common", re.compile(r"\bcommon(ly)?\b|frequent(ly)?|\boften\b|usually|majority", re.IGNORECASE)),
)
KINDS = ("symptom", "sign", "laboratory", "imaging")
REGIONS = ("head", "neck", "chest", "abdomen", "pelvis", "spine", "upper_limb", "lower_limb", "whole_body")
MODALITIES = ("CT", "MR", "US", "XR", "NM", "")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"with", "from", "that", "this", "which", "their", "there", "patients", "patient", "disease", "may",
         "have", "been", "also", "into", "most", "more", "than", "such", "other", "after", "before", "acute",
         "chronic", "severe", "mild", "signs", "sign", "symptoms", "symptom", "finding", "findings"}


def norm(text: str) -> str:
    return " ".join(_WORD.findall(str(text).lower()))


def content_stems(text: str) -> set[str]:
    return {w[:5] for w in _WORD.findall(str(text).lower()) if len(w) >= 4 and w not in _STOP}


# -- ontologies --------------------------------------------------------------------------------------------------

@dataclass
class Terms:
    """id -> name, normalised name or exact synonym -> ids, id -> exact synonyms (an OBO file)."""

    names: dict[str, str] = field(default_factory=dict)
    by_name: dict[str, set[str]] = field(default_factory=dict)
    synonyms: dict[str, list[str]] = field(default_factory=dict)
    xrefs: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def from_obo(cls, lines: Iterable[str]) -> Terms:
        t, current, obsolete = cls(), "", False
        pending: list[tuple[str, str, str]] = []
        for line in lines:
            line = line.rstrip("\n")
            if line == "[Term]" or line.startswith("[Typedef]"):
                current, obsolete = "", False
            elif line.startswith("id: "):
                current = line[4:].strip()
            elif line.startswith("is_obsolete: true"):
                obsolete = True
                t.names.pop(current, None)
            elif current and not obsolete:
                if line.startswith("name: "):
                    t.names[current] = line[6:].strip()
                    pending.append((current, line[6:].strip(), "name"))
                elif line.startswith("synonym: ") and " EXACT" in line:
                    m = re.match(r'synonym: "(.*?)"', line)
                    if m:
                        t.synonyms.setdefault(current, []).append(m.group(1))
                        pending.append((current, m.group(1), "synonym"))
                elif line.startswith("xref: "):
                    t.xrefs.setdefault(current, set()).add(line[6:].split()[0])
        for term, words, _ in pending:
            if term in t.names:
                t.by_name.setdefault(norm(words), set()).add(term)
        return t

    def find(self, words: str) -> str:
        """The one term with this exact name or exact synonym, or ''."""
        hits = self.by_name.get(norm(words), set())
        return next(iter(hits)) if len(hits) == 1 else ""

    def words_of(self, term: str) -> list[str]:
        return [self.names.get(term, "")] + self.synonyms.get(term, [])


def hpoa_frequency(value: str) -> float | None:
    value = (value or "").strip()
    if value in HPO_FREQUENCY:
        return HPO_FREQUENCY[value]
    m = re.fullmatch(r"(\d+)\s*/\s*(\d+)", value)
    if m and int(m.group(2)):
        return int(m.group(1)) / int(m.group(2))
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%", value)
    if m:
        return float(m.group(1)) / 100
    return None


def from_hpoa(lines: Iterable[str], disease_ids: set[str], hpo: Terms) -> list[cg.Manifestation]:
    """The phenotypes the HPO annotations give these diseases (OMIM:, ORPHA:, DECIPHER:), aspect P only."""
    out: dict[str, cg.Manifestation] = {}
    for line in lines:
        if line.startswith(("#", "database_id")):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 11 or parts[0] not in disease_ids or parts[10] != "P":
            continue
        term = parts[3]
        excluded = parts[2].strip().upper() == "NOT"
        frequency = 0.0 if excluded else hpoa_frequency(parts[7])
        citation = {"source": "HPO annotations", "id": parts[0], "reference": parts[4], "evidence": parts[5],
                    "licence": "HPO licence (open, attribution)"}
        m = out.get(term)
        if m is None:
            out[term] = cg.Manifestation(
                manifestation_id=term, words=hpo.names.get(term, term),
                concept=cg.Concept("HP", term, hpo.names.get(term, "")),
                typicality="excluding" if excluded else "", frequency=frequency, citations=[citation])
        else:
            m.citations.append(citation)
            if frequency is not None and (m.frequency is None or frequency > m.frequency):
                m.frequency = frequency
    return list(out.values())


def from_doid(doid: str, symptom_ids: Iterable[str], symp: Terms, hpo: Terms) -> list[cg.Manifestation]:
    """The Disease Ontology's has_symptom axioms of one disease, named by the Symptom Ontology; the HPO term when
    the symptom's name is exactly an HPO name or synonym."""
    out = []
    for sid in sorted(set(symptom_ids)):
        name = symp.names.get(sid, "")
        if not name:
            continue
        term = hpo.find(name)
        concept = cg.Concept("HP", term, hpo.names[term]) if term else cg.Concept("SYMP", sid, name)
        out.append(cg.Manifestation(
            manifestation_id=term or sid, words=name, concept=concept,
            citations=[{"source": "Disease Ontology has_symptom", "id": doid, "symptom": sid,
                        "licence": "CC0 (Disease Ontology)"}]))
    return out


# -- text --------------------------------------------------------------------------------------------------------

def _text(element: ET.Element) -> str:
    return " ".join("".join(element.itertext()).split())


def sections_from_jats(xml: str, limit: int = SECTION_CHARACTERS) -> list[tuple[str, str]]:
    """(title, text) of the sections of a JATS article whose title speaks of the disease's presentation,
    diagnosis or imaging, in the order of the article, until ``limit`` characters."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    out, used = [], 0
    for sec in root.iter("sec"):
        title_el = sec.find("title")
        title = _text(title_el) if title_el is not None else ""
        if not SECTION_WORDS.search(title):
            continue
        paragraphs = [_text(p) for p in sec.findall("p")]
        text = " ".join(p for p in paragraphs if p)
        if not text:
            continue
        text = text[: max(0, limit - used)]
        if not text:
            break
        out.append((title, text))
        used += len(text)
    return out


def html_sections(page: str, titles: Iterable[str], limit: int = SECTION_CHARACTERS) -> list[tuple[str, str]]:
    """The named sections of a Bookshelf page (StatPearls: "History and Physical", "Evaluation"), tags removed."""
    out, used = [], 0
    for title in titles:
        m = re.search(rf">\s*{re.escape(title)}\s*</h2>(.*?)(?:<h2\b|$)", page, re.DOTALL | re.IGNORECASE)
        if not m:
            continue
        text = " ".join(re.sub(r"<[^>]+>", " ", m.group(1)).split())
        text = text[: max(0, limit - used)]
        if text:
            out.append((title, text))
            used += len(text)
    return out


# -- the readers -------------------------------------------------------------------------------------------------

def extraction_prompt(disease: str, sections: list[tuple[str, str]]) -> str:
    body = "\n\n".join(f"## {t}\n{x}" for t, x in sections)
    return (
        f"Text about the disease: {disease}.\n\n{body}\n\n"
        "List every manifestation of this disease that the text states: symptoms, signs, laboratory findings and "
        "imaging findings. Only what the text says; nothing from your own knowledge.\n"
        "Answer with a JSON array only. Each item:\n"
        '{"manifestation": "<short English name>", "quote": "<one sentence copied exactly from the text>", '
        '"typicality": "hallmark|common|rare|excluding|", "kind": "symptom|sign|laboratory|imaging", '
        '"modality": "CT|MR|US|XR|NM|", "region": "head|neck|chest|abdomen|pelvis|spine|upper_limb|lower_limb|'
        'whole_body|"}\n'
        "typicality only when the quote says how often or how typical; otherwise an empty string. "
        "modality and region only for an imaging finding.")


def read_extraction(answer: str) -> list[dict[str, str]]:
    """The items of a reader's answer; anything that is not a list of objects is an empty answer."""
    text = (answer or "").strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        name, quote = str(item.get("manifestation", "")).strip(), str(item.get("quote", "")).strip()
        if name and quote:
            out.append({"manifestation": name, "quote": quote,
                        "typicality": str(item.get("typicality", "")).strip().lower(),
                        "kind": str(item.get("kind", "")).strip().lower(),
                        "modality": str(item.get("modality", "")).strip().upper(),
                        "region": str(item.get("region", "")).strip().lower()})
    return out


def quote_in(quote: str, sections: list[tuple[str, str]]) -> bool:
    q = " ".join(quote.split()).strip(" .")
    return bool(q) and any(q.lower() in " ".join(x.split()).lower() for _, x in sections)


def names_it(item: dict[str, str], hpo: Terms) -> bool:
    """The quote names the manifestation: all its content words (by their first five letters), or the name or an
    exact synonym of its HPO term."""
    quote = norm(item["quote"])
    stems = content_stems(item["manifestation"])
    quote_stems = content_stems(item["quote"])
    if stems and stems <= quote_stems:
        return True
    term = hpo.find(item["manifestation"])
    return bool(term) and any(norm(w) and f" {norm(w)} " in f" {quote} " for w in hpo.words_of(term))


def typicality_of(quote: str) -> str:
    for name, pattern in TYPICALITY_WORDS:
        if pattern.search(quote):
            return name
    return ""


def judge_prompt(disease: str, item: dict[str, str]) -> str:
    return (f"Disease: {disease}\nSentence from a published text: \"{item['quote']}\"\n"
            f"Claim: the sentence states that {disease} presents with \"{item['manifestation']}\".\n"
            'Does the sentence state it? Answer with JSON only: {"states": "yes" | "no" | "unclear"}')


def read_judgment(answer: str) -> str:
    m = re.search(r'"states"\s*:\s*"(yes|no|unclear)"', answer or "", re.IGNORECASE)
    return m.group(1).lower() if m else ""


@dataclass
class Link:
    manifestation: str
    quote: str
    source: dict[str, Any]
    readers: list[str] = field(default_factory=list)
    typicality: dict[str, str] = field(default_factory=dict)
    kind: str = ""
    modality: str = ""
    region: str = ""
    verdict: str = ""                 # names, judges, refused_quote, judges_no, doubt
    judges: dict[str, str] = field(default_factory=dict)


def merge_readings(readings: dict[str, list[dict[str, str]]], sections: list[tuple[str, str]],
                   source: dict[str, Any], hpo: Terms) -> list[Link]:
    """One link per (manifestation, quote); refused when its quote is not in the text, verified by the words when
    the quote names it, else left for the judges (verdict '')."""
    links: dict[tuple[str, str], Link] = {}
    for reader, items in readings.items():
        for item in items:
            key = (hpo.find(item["manifestation"]) or norm(item["manifestation"]), norm(item["quote"]))
            link = links.get(key)
            if link is None:
                link = links[key] = Link(item["manifestation"], item["quote"], source, kind=item["kind"],
                                         modality=item["modality"], region=item["region"])
            link.readers.append(reader)
            link.typicality[reader] = item["typicality"]
            if not quote_in(item["quote"], sections):
                link.verdict = "refused_quote"
            elif link.verdict != "refused_quote" and names_it(item, hpo):
                link.verdict = "names"
    return list(links.values())


def judged(link: Link, verdicts: dict[str, str]) -> str:
    link.judges = dict(verdicts)
    values = set(verdicts.values())
    if values == {"yes"} and len(verdicts) >= 2:
        return "judges"
    if values == {"no"} and len(verdicts) >= 2:
        return "judges_no"
    return "doubt"


def to_manifestation(link: Link, hpo: Terms, index: int) -> cg.Manifestation:
    term = hpo.find(link.manifestation)
    stated = typicality_of(link.quote)
    readers = {t for t in link.typicality.values() if t}
    typicality = stated or (readers.pop() if len(readers) == 1 and len(link.readers) >= 2 else "")
    if typicality not in cg.TYPICALITIES:
        typicality = ""
    verified = []
    if link.kind == "imaging" and (link.modality or link.region):
        verified.append({"kind": "imaging", "modality": link.modality if link.modality in MODALITIES else "",
                         "region": link.region if link.region in REGIONS else ""})
    return cg.Manifestation(
        manifestation_id=term or f"text:{index}", words=link.manifestation,
        concept=cg.Concept("HP", term, hpo.names[term]) if term else None, typicality=typicality,
        region=link.region if link.region in REGIONS else "", verified_by=verified,
        citations=[{**link.source, "quote": link.quote, "verified_by": link.verdict, "readers": link.readers,
                    "judges": link.judges}])


def merge_profile(disease_id: str, words: str, parts: Iterable[cg.Manifestation]) -> cg.Disease:
    """One manifestation per concept (or per words when there is none), every citation kept; the highest stated
    frequency, and a typicality only when the sources do not contradict each other."""
    merged: dict[str, cg.Manifestation] = {}
    for m in parts:
        key = m.concept.curie if m.concept and m.concept.code else norm(m.words)
        have = merged.get(key)
        if have is None:
            merged[key] = cg.Manifestation(**{**m.__dict__, "citations": list(m.citations),
                                              "verified_by": list(m.verified_by)})
            continue
        have.citations += m.citations
        have.verified_by += [v for v in m.verified_by if v not in have.verified_by]
        if m.frequency is not None and (have.frequency is None or m.frequency > have.frequency):
            have.frequency = m.frequency
        if m.typicality and have.typicality and m.typicality != have.typicality:
            have.typicality = ""
        elif m.typicality and not have.typicality:
            have.typicality = m.typicality
        have.region = have.region or m.region
    return cg.Disease(disease_id=disease_id, branch=cg.HYPOTHESIZED_DISEASES, words=words,
                      origin=cg.PHYSICIAN, manifestations=list(merged.values()))


__all__ = [n for n in dir() if not n.startswith("_")]
