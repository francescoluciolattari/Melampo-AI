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
# sections about how the evidence was gathered or what follows the diagnosis say little of how the disease shows
SECTION_EXCLUDED = re.compile(r"method|statistic|search|selection|result|discussion|limitation|prognos|outcome|"
                              r"treatment|therap|management|surveillance|screening|accuracy|biomarker|score|"
                              r"risk factor|epidemiolog|conclusion|future", re.IGNORECASE)
# a sentence that speaks of a test's value or of what predicts an outcome is not a sentence that the disease shows
# something: it goes to the judges even when it names the manifestation
NOT_A_MANIFESTATION = re.compile(r"predict|prognos|mortality|outcome|risk factor|little value|no (significant )?"
                                 r"(difference|association)|not associated|sensitivity|specificity|accuracy|"
                                 r"diagnostic value|defined|score|surveillance|screening|associated with|accura|coinciden|"
                                 r"correlat|usefulness|criteria|\b(rats?|mice|mouse|murine|rodents?|pigs?|dogs?|"
                                 r"animals?)\b|animal model|experimental|in vitro|placebo|randomi[sz]ed|\btrial|"
                                 r"mg/day|treat(ed|ment)", re.IGNORECASE)
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
    by_own_name: dict[str, set[str]] = field(default_factory=dict)
    synonyms: dict[str, list[str]] = field(default_factory=dict)
    xrefs: dict[str, set[str]] = field(default_factory=dict)
    parents: dict[str, set[str]] = field(default_factory=dict)

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
                elif line.startswith("is_a: "):
                    t.parents.setdefault(current, set()).add(line[6:].split()[0])
        for term, words, kind in pending:
            if term in t.names:
                t.by_name.setdefault(norm(words), set()).add(term)
                if kind == "name":
                    t.by_own_name.setdefault(norm(words), set()).add(term)
        return t

    def find(self, words: str, prefix: str = "") -> str:
        """The one term whose name is these words; else the one term with them as an exact synonym; else ''.
        (Corrected 15 November 2026: "pneumonia" is MONDO's name of one term and an exact synonym of others, and
        the first pilot, which looked at names and synonyms together, found no term for 7 of its 10 diagnoses.)"""
        # (15 November 2026, second run: mondo.obo carries HPO classes too, and "pneumonia" is the name of a MONDO
        # term and of an HP term; the caller says which ontology it means)
        own = {t for t in self.by_own_name.get(norm(words), set()) if t.startswith(prefix)}
        if len(own) == 1:
            return next(iter(own))
        if own:
            return ""
        hits = {t for t in self.by_name.get(norm(words), set()) if t.startswith(prefix)}
        return next(iter(hits)) if len(hits) == 1 else ""

    def ancestors(self, term: str) -> set[str]:
        """Every term above this one by is_a."""
        out, todo = set(), list(self.parents.get(term, ()))
        while todo:
            t = todo.pop()
            if t not in out:
                out.add(t)
                todo += self.parents.get(t, ())
        return out

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
        if not SECTION_WORDS.search(title) or SECTION_EXCLUDED.search(title):
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


_HEADING = re.compile(r"<h([23])\b[^>]*>(.*?)</h\1>", re.DOTALL | re.IGNORECASE)
_NOISE = re.compile(r"<(sup|style|table|figure|math)\b.*?</\1>", re.DOTALL | re.IGNORECASE)


def wikipedia_sections(html: str, limit: int = SECTION_CHARACTERS) -> list[tuple[str, str]]:
    """The sections of a Wikipedia article (the HTML of ``action=parse``) whose heading speaks of the presentation,
    the diagnosis or the imaging, with the paragraphs and list items only: references, tables, figures and
    formulas left out."""
    html = _NOISE.sub(" ", html or "")
    marks = list(_HEADING.finditer(html))
    out, used, taken_until = [], 0, -1
    for i, m in enumerate(marks):
        if m.start() < taken_until:                  # a subsection of a section already taken: read with it
            continue
        title = " ".join(re.sub(r"<[^>]+>", " ", m.group(2)).split())
        if not SECTION_WORDS.search(title) or SECTION_EXCLUDED.search(title):
            continue
        end = len(html)
        for later in marks[i + 1:]:
            if later.group(1) <= m.group(1):         # the next heading of the same or a higher level ends it
                end = later.start()
                break
        body = html[m.end():end]
        taken_until = end
        parts = re.findall(r"<(p|li)\b[^>]*>(.*?)</\1>", body, re.DOTALL | re.IGNORECASE)
        text = " ".join(" ".join(re.sub(r"<[^>]+>", " ", x).split()) for _, x in parts)
        text = re.sub(r"\s+([.,;:])", r"\1", text).replace("&amp;", "&").replace("&#160;", " ")
        text = text[: max(0, limit - used)]
        if text and title not in [t for t, _ in out]:
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
        "Not a manifestation: a test studied for its accuracy, a predictor of an outcome or of severity, a "
        "complication of treatment. "
        "typicality only when the quote says how often or how typical; otherwise an empty string. "
        "modality and region only for an imaging finding.")


def _json_list(text: str) -> list:
    """The first JSON array of objects in the text (fenced or not, after prose, "[1]" citations skipped), or the
    list under a key of a JSON object ({"manifestations": [...]})."""
    decoder = json.JSONDecoder()
    for m in re.finditer(r"[\[{]", text):
        try:
            value, _ = decoder.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, list)), None)
        if isinstance(value, list) and any(isinstance(v, dict) for v in value):
            return value
    return []


def read_extraction(answer: str) -> list[dict[str, str]]:
    """The items of a reader's answer; anything that is not a list of objects is an empty answer. (Corrected 15
    November 2026: the first pilot took the text from the first "[" to the last "]", and lost every answer whose
    prose cited "[1]" before the list.)"""
    items = _json_list((answer or "").strip())
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
    if NOT_A_MANIFESTATION.search(item["quote"]):
        return False
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
            f"Claim: the sentence states that patients with {disease} present with \"{item['manifestation']}\" "
            "as a sign, symptom, laboratory or imaging finding of the disease.\n"
            "Answer no when the sentence speaks of another disease, of animals or experiments, of a treatment or a "
            "trial, of a test's accuracy, or of what predicts an outcome or severity.\n"
            'Does the sentence state the claim? Answer with JSON only: {"states": "yes" | "no" | "unclear"}')


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


def disease_words(name: str, synonyms: Iterable[str], sections: list[tuple[str, str]]) -> set[str]:
    """How the text may name the disease: its name and exact synonyms, the initials of its name ("AC" for acute
    cholecystitis) and any abbreviation the text defines for it ("acute pancreatitis (AP)")."""
    words = {n for n in [name, *synonyms] if n}
    initials = "".join(w[0] for w in name.split() if w)
    if len(initials) >= 2:
        words.add(initials.upper())
    text = " ".join(x for _, x in sections)
    for n in list(words):
        for m in re.finditer(rf"{re.escape(n)}\s*\(([A-Z][A-Za-z]{{1,6}})\)", text, re.IGNORECASE):
            words.add(m.group(1))
    return words


def names_the_disease(quote: str, words: Iterable[str]) -> bool:
    for w in words:
        if w.isupper():            # an abbreviation: as written, a word of its own
            if re.search(rf"(?<![A-Za-z]){re.escape(w)}(?![A-Za-z])", quote):
                return True
        elif re.search(rf"(?<![a-z]){re.escape(w.lower())}(?![a-z])", quote.lower()):
            return True
    return False


def merge_readings(readings: dict[str, list[dict[str, str]]], sections: list[tuple[str, str]],
                   source: dict[str, Any], hpo: Terms, disease: Iterable[str] | None = None) -> list[Link]:
    """One link per (manifestation, quote); refused when its quote is not in the text; verified by the words when
    the quote names the manifestation and, when ``disease`` gives its words, the disease too (15 November 2026,
    third run: a sentence of a nephrolithiasis review that named "hypokalemic alkalosis" spoke of Bartter
    syndrome); else left for the judges (verdict '')."""
    disease = set(disease or ())
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
            elif link.verdict != "refused_quote" and names_it(item, hpo) and (
                    not disease or names_the_disease(item["quote"], disease)):
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


def source_key(citation: dict[str, Any]) -> str:
    """What makes two citations independent: the publication (PMID, PMC, DOI) or the curated reference."""
    for k in ("reference", "pmid", "id"):
        value = str(citation.get(k) or "").strip()
        if value:
            return value
    return str(citation.get("source", ""))


def evidence_of(m: cg.Manifestation) -> str:
    """``established`` when two independent sources or more state it, ``reported`` when one does."""
    keys = {source_key(c) for c in m.citations if source_key(c)}
    return "established" if len(keys) >= 2 else ("reported" if keys else "")


def merge_profile(disease_id: str, words: str, parts: Iterable[cg.Manifestation]) -> cg.Disease:
    """One manifestation per concept (or per words when there is none), every citation kept; the highest stated
    frequency, a typicality only when the sources do not contradict each other, and the level of evidence by the
    number of independent sources (``evidence_of``)."""
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
    for m in merged.values():
        m.evidence = evidence_of(m)
    return cg.Disease(disease_id=disease_id, branch=cg.HYPOTHESIZED_DISEASES, words=words,
                      origin=cg.PHYSICIAN, manifestations=list(merged.values()))


# -- the licence of a text ---------------------------------------------------------------------------------------

def open_for_use(licence: str) -> bool:
    """CC0, CC BY and CC BY-SA allow use in a commercial product; NC (non-commercial) and ND do not serve here.
    (Corrected 15 November 2026: the pilot accepted any licence beginning with "cc by", CC BY-NC included.)"""
    text = norm(licence)
    if not text:
        return False
    if "nc" in text.split() or "nd" in text.split() or "noncommercial" in text or "noderiv" in text:
        return False
    return text.startswith(("cc0", "cc by", "cc zero", "public domain")) or text in ("cc by sa", "cc by 4 0")


# -- Wikipedia as an index of its references ---------------------------------------------------------------------

_CITE_NOTE = re.compile(r'href="#(cite_note-[^"]+)"')
_REF_ITEM = re.compile(r'<li id="(cite_note-[^"]+)"[^>]*>(.*?)</li>', re.DOTALL)
_PMID = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)|PMID[^0-9]{0,40}(\d{5,9})")
_PMC = re.compile(r"\b(PMC\d{1,9})\b")
_DOI = re.compile(r'doi\.org/(10\.\d{4,9}/[^\s"<>&]+)')


def wikipedia_references(html: str) -> list[dict[str, str]]:
    """The publications cited in the sections ``wikipedia_sections`` reads (presentation, diagnosis, imaging): each
    with its PMID, PMCID and DOI when the reference gives them. Wikipedia is an index here, never the source: the
    text that enters a profile is the publication's own (15 November 2026)."""
    html = html or ""
    marks = list(_HEADING.finditer(html))
    cited: list[str] = []
    taken_until = -1
    for i, m in enumerate(marks):
        if m.start() < taken_until:
            continue
        title = " ".join(re.sub(r"<[^>]+>", " ", m.group(2)).split())
        if not SECTION_WORDS.search(title) or SECTION_EXCLUDED.search(title):
            continue
        end = next((later.start() for later in marks[i + 1:] if later.group(1) <= m.group(1)), len(html))
        taken_until = end
        for note in _CITE_NOTE.findall(html[m.end():end]):
            if note not in cited:
                cited.append(note)
    items = {k: v for k, v in _REF_ITEM.findall(html)}
    out = []
    for note in cited:
        body = items.get(note, "")
        pmid = next((a or b for a, b in _PMID.findall(body)), "")
        pmc = next(iter(_PMC.findall(body)), "")
        doi = next(iter(_DOI.findall(body)), "").rstrip(".")
        if pmid or pmc or doi:
            ref = {"pmid": pmid, "pmcid": pmc, "doi": doi}
            if ref not in out:
                out.append(ref)
    return out


def body_text(xml: str, limit: int = 80000) -> str:
    """The paragraphs of a JATS article's body (methods, references and tables left out), at most ``limit``."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return ""
    body = root.find(".//body")
    if body is None:
        return ""
    parts, used = [], 0
    for sec in body.iter("sec"):
        title_el = sec.find("title")
        title = _text(title_el) if title_el is not None else ""
        if re.search(r"method|statistic|reference|acknowledg|funding|conflict", title, re.IGNORECASE):
            continue
        for p in sec.findall("p"):
            text = _text(p)
            if text and used < limit:
                parts.append(text)
                used += len(text)
    if not parts:
        parts = [_text(p) for p in body.findall(".//p")]
    return " ".join(parts)[:limit]


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def supporting_sentences(text: str, manifestation: str, hpo: Terms, disease: Iterable[str],
                         title: str = "") -> list[str]:
    """The sentences of a publication that name the manifestation (its words or its HPO term's name or synonym)
    and say something of the disease (they name it, or the publication's title does), and are not about a test's
    value, an outcome, animals or a treatment."""
    disease = list(disease)
    about = names_the_disease(title, disease)
    out = []
    for sentence in _SENTENCE_END.split(" ".join(text.split())):
        if len(sentence) > 600:
            continue
        item = {"manifestation": manifestation, "quote": sentence}
        if names_it(item, hpo) and (about or names_the_disease(sentence, disease)):
            out.append(sentence)
    return out


# -- the reading measured against a curated profile --------------------------------------------------------------

def _within(term: str, below: str, hpo: Terms, steps: int) -> bool:
    """``term`` is at most ``steps`` is_a steps above ``below``."""
    level = {below}
    for _ in range(steps):
        level = {p for t in level for p in hpo.parents.get(t, ())}
        if term in level:
            return True
    return False


def agrees_with_curation(term: str, curated: set[str], hpo: Terms) -> bool:
    """A term read from a publication agrees with a curated profile when the profile has it, a term above it (the
    reading is more specific than the curators), or a term at most two steps below it (the reading is a little more
    general). A term far above ("Abnormality of the abdomen") does not agree with everything below it."""
    if not term:
        return False
    if term in curated or hpo.ancestors(term) & curated:
        return True
    return any(_within(term, c, hpo, 2) for c in curated)


__all__ = [n for n in dir() if not n.startswith("_")]
