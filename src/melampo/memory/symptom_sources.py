"""Disease-to-symptom links from every source with a usable licence, in one shape.

The concept graph's disease-to-phenotype edges come from `phenotype.hpoa`,
which annotates rare diseases only. This module turns each candidate source
for *all* diseases into the same `SymptomLink` record, so that the question
"which source covers which diseases, and how well" is answered by measuring
(`evaluation/symptom_coverage.py`) rather than by reading each source's own
description of itself.

Sources, and why each is here (licences verified at the source, 2026-09-26):

- **HPO annotations** (`phenotype.hpoa`, HPO licence): curated and referenced,
  rare diseases only. The baseline every other source is compared against.
- **NCI Thesaurus** (CC BY 4.0), read through UMLS: the roles
  ``Disease_Has_Finding`` (48,691 pairs in UMLS), ``Disease_May_Have_Finding``
  (18,671) and ``Disease_Excludes_Finding`` (30,402). The last is negative
  evidence -- a finding that argues *against* a disease -- which no other
  source here provides.
- **Disease Ontology** (CC0): ``has_symptom`` (RO:0002452) axioms in
  ``doid.owl``, pointing at Symptom Ontology (SYMP, CC0) terms. Measured on
  the 2026-08-31 release: 873 diseases, 2,268 pairs. The OBO releases carry
  none of these axioms -- only the OWL file does -- and a further set of
  symptoms exists only as "has_symptom <phrase>" inside definition text,
  which is extracted separately and at a lower evidence tier.
- **Wikidata** (CC0): property P780 "symptoms and signs". Community-edited;
  whether a statement carries a reference is recorded, because that is the
  difference between a claim and a sourced claim.

SNOMED CT is deliberately absent. Its licence treats any use in a
non-Member country as use (Affiliate Licence, clause 9.5), whether through
an API or a download, and Italy is not a Member; it also models disorders
by site, morphology and cause rather than by symptoms.

**Evidence tiers, not a single weight.** A source's tier says how a link was
produced, not how strongly the finding predicts the disease:

    1  curated, referenced assertion        HPO; NCIt Disease_Has_Finding / Excludes
    2  curated, possibility or axiom        NCIt Disease_May_Have_Finding; DO axiom
    3  community-edited, with a reference   Wikidata P780 with a reference
    4  community-edited or text-derived     Wikidata without a reference; DO text

Links from different tiers are kept apart all the way to the graph so a
downstream weight can be chosen -- and revised -- in one place, not baked in
here.

**Two identifier spaces meet here.** Diseases are aligned to Mondo, and only
through Mondo xrefs marked ``MONDO:equivalentTo`` -- the other xrefs are
broader, narrower or obsolete, and following them would silently attach a
rare subtype's annotations to a common disease. Symptoms are aligned to HPO,
because HPO is the graph's phenotype vocabulary; every link records *how* its
HPO id was obtained, and a link that could not be mapped keeps an empty
``hpo_id`` rather than a guess.
"""

import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import IO, Any

SOURCE_HPO = "hpo"
SOURCE_NCIT = "ncit"
SOURCE_DO = "disease_ontology"
SOURCE_DO_TEXT = "disease_ontology_text"
SOURCE_WIKIDATA = "wikidata"

RELATION_HAS_FINDING = "has_finding"
RELATION_MAY_HAVE_FINDING = "may_have_finding"
RELATION_EXCLUDES_FINDING = "excludes_finding"

TIER_CURATED = 1
TIER_CURATED_POSSIBLE = 2
TIER_COMMUNITY_REFERENCED = 3
TIER_COMMUNITY_UNREFERENCED = 4

MAPPING_SOURCE = "source"  # the source itself gives an HPO id
MAPPING_XREF = "xref"  # an hp.obo xref points at the source's symptom id
MAPPING_LABEL = "label"  # the symptom's label is an HPO name or safe synonym
MAPPING_NONE = ""


@dataclass(frozen=True)
class SymptomLink:
    """One disease-to-finding assertion, as one source states it."""

    source: str
    disease_id: str
    disease_label: str
    symptom_id: str
    symptom_label: str
    relation: str
    tier: int
    mondo_id: str = ""
    reference: str = ""
    hpo_id: str = ""
    mapping: str = MAPPING_NONE

    @property
    def is_exclusion(self) -> bool:
        return self.relation == RELATION_EXCLUDES_FINDING

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "mondo_id": self.mondo_id,
            "disease_id": self.disease_id,
            "disease_label": self.disease_label,
            "symptom_id": self.symptom_id,
            "symptom_label": self.symptom_label,
            "relation": self.relation,
            "tier": self.tier,
            "reference": self.reference,
            "hpo_id": self.hpo_id,
            "mapping": self.mapping,
        }


# ---------------------------------------------------------------------------
# Mondo: the disease identity layer
# ---------------------------------------------------------------------------

_XREF_PREFIX_ALIASES = {"ORPHANET": "ORPHA", "UMLS_CUI": "UMLS", "NCI": "NCIT", "MIM": "OMIM"}


def normalise_xref(xref: str) -> str:
    """One spelling per vocabulary: ``Orphanet:558`` and ``ORPHA:558`` are the same id."""
    prefix, _, local = xref.strip().partition(":")
    if not local:
        return ""
    prefix = prefix.upper()
    return f"{_XREF_PREFIX_ALIASES.get(prefix, prefix)}:{local.strip()}"


@dataclass(frozen=True)
class MondoDisease:
    mondo_id: str
    label: str
    exact_synonyms: tuple[str, ...] = ()
    equivalent_xrefs: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()
    obsolete: bool = False


def parse_mondo(lines: Iterable[str]) -> Iterator[MondoDisease]:
    """Parse mondo.obo keeping only the xrefs Mondo itself marks as equivalent.

    `concept_resolution.parse_obo` drops xref qualifiers, which is right for
    HPO but wrong here: in Mondo the qualifier is the whole meaning of the
    xref (equivalentTo, obsoleteEquivalent, or a bare provenance pointer).
    """
    term_id = label = ""
    synonyms: list[str] = []
    xrefs: list[str] = []
    parents: list[str] = []
    obsolete = False
    inside = False

    def flush() -> MondoDisease | None:
        if not inside or not term_id.startswith("MONDO:"):
            return None
        return MondoDisease(term_id, label, tuple(synonyms), tuple(xrefs), tuple(parents), obsolete)

    for raw in lines:
        line = raw.rstrip("\n")
        if line.startswith("["):
            finished = flush()
            if finished is not None:
                yield finished
            inside = line.strip() == "[Term]"
            term_id, label, synonyms, xrefs, parents, obsolete = "", "", [], [], [], False
            continue
        if not inside or not line:
            continue
        key, _, value = line.partition(": ")
        if key == "id":
            term_id = value.strip()
        elif key == "name":
            label = value.strip()
        elif key == "is_a":
            parents.append(value.split("!")[0].split("{")[0].strip())
        elif key == "is_obsolete":
            obsolete = value.strip().lower() == "true"
        elif key == "synonym" and value.startswith('"'):
            closing = value.find('"', 1)
            if closing > 0 and value[closing + 1 :].strip().startswith("EXACT"):
                synonyms.append(value[1:closing])
        elif key == "xref" and "MONDO:equivalentTo" in value:
            xref = normalise_xref(value.split("{")[0].split(" ")[0])
            if xref:
                xrefs.append(xref)
    finished = flush()
    if finished is not None:
        yield finished


@dataclass
class DiseaseIndex:
    """Mondo diseases, and every equivalent id another source may use for them."""

    diseases: dict[str, MondoDisease] = field(default_factory=dict)
    by_xref: dict[str, set[str]] = field(default_factory=dict)
    children: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def from_mondo(cls, diseases: Iterable[MondoDisease]) -> "DiseaseIndex":
        index = cls()
        for disease in diseases:
            if disease.obsolete:
                continue
            index.diseases[disease.mondo_id] = disease
            for xref in disease.equivalent_xrefs:
                index.by_xref.setdefault(xref, set()).add(disease.mondo_id)
            for parent in disease.parents:
                index.children.setdefault(parent, set()).add(disease.mondo_id)
        return index

    def mondo_for(self, xref: str) -> set[str]:
        return set(self.by_xref.get(normalise_xref(xref), ()))

    def xrefs_of(self, mondo_id: str, prefix: str) -> list[str]:
        disease = self.diseases.get(mondo_id)
        if disease is None:
            return []
        wanted = prefix.upper() + ":"
        return [xref for xref in disease.equivalent_xrefs if xref.startswith(wanted)]

    def descendants(self, mondo_id: str) -> set[str]:
        seen: set[str] = set()
        stack = [mondo_id]
        while stack:
            for child in self.children.get(stack.pop(), ()):
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        return seen


def assign_mondo(links: Iterable[SymptomLink], index: DiseaseIndex) -> list[SymptomLink]:
    """Attach the Mondo id to links that arrived with only a source-native disease id.

    A source id that maps to more than one Mondo disease is left unassigned:
    choosing one would be a guess, and the coverage report would then count
    the guess as coverage.
    """
    assigned: list[SymptomLink] = []
    for link in links:
        if link.mondo_id:
            assigned.append(link)
            continue
        candidates = index.mondo_for(link.disease_id)
        assigned.append(replace(link, mondo_id=next(iter(candidates))) if len(candidates) == 1 else link)
    return assigned


# ---------------------------------------------------------------------------
# HPO annotations
# ---------------------------------------------------------------------------


def links_from_hpoa(annotations: Iterable[Any]) -> Iterator[SymptomLink]:
    """Phenotypic-abnormality rows of phenotype.hpoa as links (aspect P only).

    Onset, inheritance and clinical-course rows (aspects C, I, M) describe
    the disease, not a finding a patient presents with, and are skipped.
    """
    for annotation in annotations:
        if annotation.aspect != "P":
            continue
        yield SymptomLink(
            source=SOURCE_HPO,
            disease_id=normalise_xref(annotation.disease_id),
            disease_label=annotation.disease_name,
            symptom_id=annotation.phenotype_id,
            symptom_label="",
            relation=RELATION_EXCLUDES_FINDING if annotation.is_excluded else RELATION_HAS_FINDING,
            tier=TIER_CURATED,
            reference=annotation.reference,
            hpo_id=annotation.phenotype_id,
            mapping=MAPPING_SOURCE,
        )


# ---------------------------------------------------------------------------
# Disease Ontology (doid.owl) and Symptom Ontology (symp.obo)
# ---------------------------------------------------------------------------

_OWL = "{http://www.w3.org/2002/07/owl#}"
_RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
_RDFS = "{http://www.w3.org/2000/01/rdf-schema#}"
_OBO_DEFINITION = "{http://purl.obolibrary.org/obo/}IAO_0000115"
_HAS_SYMPTOM = "http://purl.obolibrary.org/obo/RO_0002452"
_OBO_PREFIX = "http://purl.obolibrary.org/obo/"


def _curie(iri: str) -> str:
    local = iri.removeprefix(_OBO_PREFIX)
    prefix, _, number = local.partition("_")
    return f"{prefix}:{number}" if number else local


@dataclass(frozen=True)
class DoidClass:
    doid: str
    label: str
    definition: str
    symptom_ids: tuple[str, ...]
    deprecated: bool = False


def parse_doid_owl(source: str | Path | IO[bytes]) -> Iterator[DoidClass]:
    """Stream doid.owl, yielding each DOID class with its has_symptom targets.

    Streamed with ``iterparse`` and cleared as it goes: the file is ~28 MB
    of RDF/XML, and nothing here needs more than one class at a time.
    """
    for _, element in ET.iterparse(source, events=("end",)):
        if element.tag != f"{_OWL}Class":
            continue
        about = element.get(f"{_RDF}about", "")
        if not about.startswith(f"{_OBO_PREFIX}DOID_"):
            element.clear()
            continue
        label_element = element.find(f"{_RDFS}label")
        definition_element = element.find(_OBO_DEFINITION)
        deprecated_element = element.find(f"{_OWL}deprecated")
        symptoms: list[str] = []
        for restriction in element.iter(f"{_OWL}Restriction"):
            prop = restriction.find(f"{_OWL}onProperty")
            target = restriction.find(f"{_OWL}someValuesFrom")
            if prop is None or target is None:
                continue
            if prop.get(f"{_RDF}resource") == _HAS_SYMPTOM:
                symptom = _curie(target.get(f"{_RDF}resource", ""))
                if symptom and symptom not in symptoms:
                    symptoms.append(symptom)
        yield DoidClass(
            doid=_curie(about),
            label=(label_element.text or "").strip() if label_element is not None else "",
            definition=(definition_element.text or "").strip() if definition_element is not None else "",
            symptom_ids=tuple(symptoms),
            deprecated=deprecated_element is not None and (deprecated_element.text or "").strip() == "true",
        )
        element.clear()


# "has_symptom fever, has_symptom arthralgia, and has_symptom maculopapular rash."
_TEXT_SYMPTOM = re.compile(r"has_symptom\s+(.+?)(?=,|;|\.(?:\s|$)|\s+and\s+has_symptom|$)")


def symptoms_in_definition(definition: str) -> list[str]:
    """Phrases the Disease Ontology writes as "has_symptom <phrase>" in prose."""
    phrases: list[str] = []
    for match in _TEXT_SYMPTOM.finditer(definition):
        phrase = match.group(1).strip()
        phrase = re.sub(r"^(?:and|or)\s+", "", phrase).strip()
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    return phrases


def links_from_doid(classes: Iterable[DoidClass], symptom_labels: Mapping[str, str]) -> Iterator[SymptomLink]:
    """DO axioms (tier 2) plus the definition-text symptoms no axiom already states (tier 4)."""
    for cls in classes:
        if cls.deprecated:
            continue
        axiom_labels = set()
        for symptom_id in cls.symptom_ids:
            label = symptom_labels.get(symptom_id, "")
            axiom_labels.add(label.lower())
            yield SymptomLink(
                source=SOURCE_DO,
                disease_id=cls.doid,
                disease_label=cls.label,
                symptom_id=symptom_id,
                symptom_label=label,
                relation=RELATION_HAS_FINDING,
                tier=TIER_CURATED_POSSIBLE,
            )
        for phrase in symptoms_in_definition(cls.definition):
            if phrase.lower() in axiom_labels:
                continue
            yield SymptomLink(
                source=SOURCE_DO_TEXT,
                disease_id=cls.doid,
                disease_label=cls.label,
                symptom_id="",
                symptom_label=phrase,
                relation=RELATION_HAS_FINDING,
                tier=TIER_COMMUNITY_UNREFERENCED,
            )


# ---------------------------------------------------------------------------
# NCI Thesaurus, through the UMLS source-asserted relations endpoint
# ---------------------------------------------------------------------------

# Each NCIt role appears in UMLS under a label and its inverse. Which of the
# two a query returns depends on which end of the pair was queried and on
# how the API orients the label -- a convention the UTS documentation does
# not state (checked 2026-09-26). `infer_orientation` decides it from the
# data instead of assuming it.
NCIT_FORWARD_LABELS = {
    "disease_has_finding": RELATION_HAS_FINDING,
    "disease_may_have_finding": RELATION_MAY_HAVE_FINDING,
    "disease_excludes_finding": RELATION_EXCLUDES_FINDING,
}
NCIT_INVERSE_LABELS = {
    "is_finding_of_disease": RELATION_HAS_FINDING,
    "may_be_finding_of_disease": RELATION_MAY_HAVE_FINDING,
    "is_not_finding_of_disease": RELATION_EXCLUDES_FINDING,
}
NCIT_FINDING_LABELS = (*NCIT_FORWARD_LABELS, *NCIT_INVERSE_LABELS)

ORIENTATION_QUERIED_IS_SUBJECT = "queried_is_subject"  # label reads queried -> related
ORIENTATION_RELATED_IS_SUBJECT = "related_is_subject"  # label reads related -> queried
ORIENTATION_UNDETERMINED = "undetermined"

_NCIT_TIER = {
    RELATION_HAS_FINDING: TIER_CURATED,
    RELATION_MAY_HAVE_FINDING: TIER_CURATED_POSSIBLE,
    RELATION_EXCLUDES_FINDING: TIER_CURATED,
}


def infer_orientation(label_counts: Mapping[str, int], *, minimum_rows: int = 20) -> str:
    """Decide how the API orients NCIt role labels, from rows returned for disease codes.

    Every code queried by the coverage probe is a disease. A disease is the
    subject of far more finding relations than it is the object of (a
    disease that is itself a finding of another disease -- anaemia in
    chronic kidney disease -- is the minority case). So whichever label
    family dominates across all queried diseases tells which way the labels
    read. Below ``minimum_rows`` the evidence is too thin and the answer is
    "undetermined": the report then shows raw label counts and no links,
    rather than links whose direction might be reversed.
    """
    forward = sum(label_counts.get(label, 0) for label in NCIT_FORWARD_LABELS)
    inverse = sum(label_counts.get(label, 0) for label in NCIT_INVERSE_LABELS)
    if forward + inverse < minimum_rows or forward == inverse:
        return ORIENTATION_UNDETERMINED
    return ORIENTATION_QUERIED_IS_SUBJECT if forward > inverse else ORIENTATION_RELATED_IS_SUBJECT


def code_from_uts_url(url: str) -> str:
    """``https://uts-ws.nlm.nih.gov/rest/content/2026AA/source/NCI/C3038`` -> ``C3038``."""
    return url.rstrip("/").rsplit("/", 1)[-1] if url else ""


def links_from_ncit_relations(
    rows: Iterable[Mapping[str, Any]],
    *,
    queried_code: str,
    disease_label: str,
    orientation: str,
    mondo_id: str = "",
) -> Iterator[SymptomLink]:
    """NCIt finding roles for one queried disease code, oriented as `infer_orientation` decided."""
    if orientation == ORIENTATION_QUERIED_IS_SUBJECT:
        wanted = NCIT_FORWARD_LABELS
    elif orientation == ORIENTATION_RELATED_IS_SUBJECT:
        wanted = NCIT_INVERSE_LABELS
    else:
        return
    for row in rows:
        relation = wanted.get(str(row.get("additionalRelationLabel", "")))
        if relation is None or str(row.get("rootSource", "NCI")) != "NCI":
            continue
        code = code_from_uts_url(str(row.get("relatedId", "")))
        if not code:
            continue
        yield SymptomLink(
            source=SOURCE_NCIT,
            disease_id=f"NCIT:{queried_code}",
            disease_label=disease_label,
            symptom_id=f"NCIT:{code}",
            symptom_label=str(row.get("relatedIdName", "")),
            relation=relation,
            tier=_NCIT_TIER[relation],
            mondo_id=mondo_id,
        )


# ---------------------------------------------------------------------------
# Wikidata P780
# ---------------------------------------------------------------------------


def _binding(row: Mapping[str, Any], name: str) -> str:
    value = row.get(name)
    return str(value.get("value", "")) if isinstance(value, Mapping) else ""


def _mondo_curie(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    return value if value.upper().startswith("MONDO:") else f"MONDO:{value}"


def links_from_wikidata(bindings: Iterable[Mapping[str, Any]], *, mondo_for_doid: Mapping[str, str] | None = None) -> Iterator[SymptomLink]:
    """Rows of `connectors.wikidata.symptom_query` results as links.

    Each row is keyed by either a Mondo id (P5270) or a DOID (P699), and
    carries the number of references on the P780 statement: one or more
    puts the link in tier 3, none in tier 4.
    """
    seen: set[tuple[str, str]] = set()
    for row in bindings:
        disease_iri = _binding(row, "disease")
        symptom_iri = _binding(row, "symptom")
        if not disease_iri or not symptom_iri:
            continue
        mondo_id = _mondo_curie(_binding(row, "mondo"))
        doid = _binding(row, "doid")
        if not mondo_id and doid and mondo_for_doid:
            mondo_id = mondo_for_doid.get(doid, "")
        disease_id = "wikidata:" + disease_iri.rsplit("/", 1)[-1]
        symptom_id = "wikidata:" + symptom_iri.rsplit("/", 1)[-1]
        if (disease_id, symptom_id) in seen:
            continue
        seen.add((disease_id, symptom_id))
        try:
            references = int(float(_binding(row, "refs") or 0))
        except ValueError:
            references = 0
        hpo = _binding(row, "hpo")
        yield SymptomLink(
            source=SOURCE_WIKIDATA,
            disease_id=disease_id,
            disease_label=_binding(row, "diseaseLabel"),
            symptom_id=symptom_id,
            symptom_label=_binding(row, "symptomLabel"),
            relation=RELATION_HAS_FINDING,
            tier=TIER_COMMUNITY_REFERENCED if references > 0 else TIER_COMMUNITY_UNREFERENCED,
            mondo_id=mondo_id,
            reference=f"{references} reference(s) on the statement",
            hpo_id=hpo if hpo.startswith("HP:") else "",
            mapping=MAPPING_SOURCE if hpo.startswith("HP:") else MAPPING_NONE,
        )


# ---------------------------------------------------------------------------
# Symptom normalisation to HPO
# ---------------------------------------------------------------------------


def hpo_xref_map(hpo_terms: Iterable[Any], prefix: str) -> dict[str, str]:
    """``NCIT:C3038`` -> ``HP:0001945``, from the xrefs hp.obo itself states.

    Only xrefs that point at exactly one HPO term are kept; an id two HPO
    terms both claim is ambiguous and is left to label matching.
    """
    owners: dict[str, set[str]] = defaultdict(set)
    wanted = prefix.upper() + ":"
    for term in hpo_terms:
        if getattr(term, "obsolete", False):
            continue
        for xref in getattr(term, "xrefs", ()):
            normalised = normalise_xref(xref)
            if normalised.startswith(wanted):
                owners[normalised].add(term.term_id)
    return {xref: next(iter(ids)) for xref, ids in owners.items() if len(ids) == 1}


def normalise_to_hpo(
    links: Iterable[SymptomLink],
    *,
    term_index: Any = None,
    xref_map: Mapping[str, str] | None = None,
) -> list[SymptomLink]:
    """Give each link an HPO id where one can be obtained without guessing.

    Order: an id the source states itself, then an hp.obo xref, then an
    exact match of the symptom's label on an HPO name or safe synonym
    (`TermIndex`, which already excludes BROAD/NARROW/RELATED synonyms). A
    label that matches more than one HPO term stays unmapped.
    """
    from .concept_resolution import normalise_surface

    xref_map = xref_map or {}
    result: list[SymptomLink] = []
    for link in links:
        if link.hpo_id:
            result.append(link)
            continue
        mapped = xref_map.get(normalise_xref(link.symptom_id)) if link.symptom_id else None
        if mapped:
            result.append(replace(link, hpo_id=mapped, mapping=MAPPING_XREF))
            continue
        if term_index is not None and link.symptom_label:
            holders = term_index.by_surface.get(normalise_surface(link.symptom_label), [])
            if len(holders) == 1:
                result.append(replace(link, hpo_id=holders[0], mapping=MAPPING_LABEL))
                continue
        result.append(link)
    return result


def label_counts(rows: Iterable[Mapping[str, Any]]) -> Counter:
    """How often each NCIt finding label appears -- the evidence `infer_orientation` reads."""
    return Counter(
        str(row.get("additionalRelationLabel", ""))
        for row in rows
        if str(row.get("additionalRelationLabel", "")) in NCIT_FINDING_LABELS
    )


def group_by_mondo(links: Sequence[SymptomLink]) -> dict[str, list[SymptomLink]]:
    grouped: dict[str, list[SymptomLink]] = defaultdict(list)
    for link in links:
        if link.mondo_id:
            grouped[link.mondo_id].append(link)
    return dict(grouped)
