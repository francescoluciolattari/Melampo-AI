"""The graph of a case (14 November 2026): every datum of the case in branches, each with the words it came from.

The structure is the owner's proposal ("Melampo — Il grafo del caso e il percorso diagnostico"), completed by the
improvements written there:

- **Branches.** Drugs (one node per drug or supplement: dose, start, end, route); exams (one node per document:
  each laboratory value, each finding of a report with its words, the imprint of an image); diseases present;
  diseases hypothesized; diseases that are candidates. Each disease carries its manifestations.
- **The candidate branch is empty when the case enters.** Nobody has named those diseases: the diagnostic path
  fills it (``CaseGraph.add_candidate``), and a candidate then follows the same schema as the others. A candidate
  added by anything else is refused (``validate``).
- **Three levels kept apart.** *Facts* of the patient (findings, measurements, imprints, medications, declared
  diseases), each with the words and their place in the document; *knowledge* (the manifestations of a disease,
  from the sources, with their citations, saved with the case as a versioned snapshot); *inference* (the state of
  each manifestation in this patient, with the facts it rests on). A fact is never rewritten by an inference.
- **Four states of a manifestation in the patient.** Present, absent, not assessed, doubtful. Absent says how:
  stated as absent, measured within range, or silent in an exam that would have shown it; these weigh differently
  (``reasoning.likelihood_ranking``). Not assessed weighs nothing and becomes the exam to propose.
- **Time.** Every fact can carry a date; a drug carries its start and end.
- **Words first.** A concept (HPO, LOINC, ATC, ...) is added beside the words, never in their place.

The graph is plain data: ``to_dict`` / ``from_dict`` round-trip it exactly, ``to_phenopacket`` /
``from_phenopacket`` exchange the patient's facts in GA4GH Phenopacket v2 (words in ``description``, document and
offsets in the evidence or in ``metaData.externalReferences``), and ``memory.case_graph_store`` writes it into
FalkorDB. Patients' text never leaves the machine through any of this.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

SCHEMA_VERSION = "case-graph/2"
READABLE_SCHEMAS = ("case-graph/1", "case-graph/2")   # /2 adds the drug knowledge and its inferences

FACT, KNOWLEDGE, INFERENCE = "fact", "knowledge", "inference"

PRESENT, ABSENT, NOT_ASSESSED, DOUBTFUL = "present", "absent", "not_assessed", "doubtful"
STATES = (PRESENT, ABSENT, NOT_ASSESSED, DOUBTFUL)
# how an absence is known: stated in a document, measured within range, or a covering exam that says nothing of it
STATED, MEASURED, SILENT = "stated", "measured", "silent_in_covering_exam"

DRUGS, EXAMS = "drugs", "exams"
PRESENT_DISEASES, HYPOTHESIZED_DISEASES, CANDIDATE_DISEASES = (
    "present_diseases", "hypothesized_diseases", "candidate_diseases")
DISEASE_BRANCHES = (PRESENT_DISEASES, HYPOTHESIZED_DISEASES, CANDIDATE_DISEASES)
BRANCHES = (DRUGS, EXAMS) + DISEASE_BRANCHES

# who put a disease in its branch
NAMED_IN_DOCUMENTS, PHYSICIAN, DIAGNOSTIC_PATH = "named_in_documents", "physician", "diagnostic_path"

TYPICALITIES = ("hallmark", "common", "rare", "excluding")

AFFIRMED, NEGATED = "affirmed", "negated"


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


class _Record:
    """Plain records: ``as_dict`` keeps every field, ``from_dict`` reads them back."""

    def as_dict(self) -> dict[str, Any]:
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = value.as_dict() if isinstance(value, _Record) else _clean(
                [v.as_dict() if isinstance(v, _Record) else v for v in value] if isinstance(value, list) else value)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        kwargs = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data[f.name]
            nested = _NESTED.get((cls.__name__, f.name))
            if nested is not None and value is not None:
                value = [nested.from_dict(v) for v in value] if isinstance(value, list) else nested.from_dict(value)
            kwargs[f.name] = value
        return cls(**kwargs)


@dataclass
class Provenance(_Record):
    """Where the words are: the document, the characters ``start``..``end`` (or the ``line``), and the words."""

    document: str
    words: str
    start: int | None = None
    end: int | None = None
    line: int | None = None


@dataclass
class Concept(_Record):
    """A code beside the words: ``system`` (HPO, LOINC, ATC, MONDO, UBERON, ...), ``code`` and its ``label``."""

    system: str
    code: str
    label: str = ""

    @property
    def curie(self) -> str:
        return self.code if ":" in self.code else f"{self.system}:{self.code}"


@dataclass
class Document(_Record):
    document_id: str
    kind: str                                   # imaging_report, laboratory, pathology, note, imaging_study, ...
    title: str = ""                             # as written ("TC torace-addome con mdc")
    date: str = ""                              # ISO date when known
    areas: list[str] = field(default_factory=list)       # the regions the exam studies (memory.exam_area)
    modality: str = ""


@dataclass
class Finding(_Record):
    """What a document states: its words, the site, whether it is affirmed or negated, how certain, when."""

    finding_id: str
    document: str
    words: str
    provenance: Provenance
    site: str = ""                              # the linked structure (TotalSegmentator class or UBERON id)
    site_words: str = ""
    polarity: str = AFFIRMED
    certainty: str = ""                         # as the reading gives it (ConText): "", certain, uncertain, ...
    temporality: str = ""
    concept: Concept | None = None
    date: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    level: str = FACT


@dataclass
class Measurement(_Record):
    """A laboratory value as printed: analyte, value, unit, the range printed beside it and the status against it."""

    measurement_id: str
    document: str
    analyte: str
    value: Any
    provenance: Provenance
    unit: str = ""
    reference: dict[str, Any] | None = None
    interpretation: str = ""                    # lab_results: within, above, below, ...
    code: Concept | None = None
    date: str = ""
    level: str = FACT


@dataclass
class Imprint(_Record):
    """The imprint of an image (memory.visual_imprint): its digest and the findings read from it."""

    imprint_id: str
    document: str
    study: str = ""
    digest: str = ""
    findings: list[str] = field(default_factory=list)
    date: str = ""
    level: str = FACT


@dataclass
class Medication(_Record):
    """A drug or a supplement the patient takes or took, as written, with its dose and its dates."""

    medication_id: str
    words: str
    provenance: Provenance
    name: str = ""
    kind: str = "drug"                          # drug, supplement
    dose: str = ""
    unit: str = ""
    frequency: str = ""
    route: str = ""
    start: str = ""
    end: str = ""
    codes: list[Concept] = field(default_factory=list)
    level: str = FACT


@dataclass
class Manifestation(_Record):
    """What a disease shows, from the sources (knowledge): symptom, sign, laboratory value or imaging finding.

    ``frequency`` (0..1) when a source gives it, else ``typicality`` (hallmark, common, rare, excluding);
    ``verified_by`` names what can show it or exclude it: ``{"kind": "imaging", "modality": "CT", "region":
    "abdomen"}`` or ``{"kind": "laboratory", "analyte": "creatinina", "code": "LOINC:2160-0", "direction":
    "above"}``; ``region`` is the body region it is seen in (memory.exam_area regions)."""

    manifestation_id: str
    words: str
    concept: Concept | None = None
    typicality: str = ""
    frequency: float | None = None
    onset: str = ""
    region: str = ""
    verified_by: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    level: str = KNOWLEDGE


@dataclass
class ManifestationState(_Record):
    """The state of one manifestation in this patient (inference), with the facts it rests on."""

    manifestation_id: str
    state: str
    basis: str = ""                             # stated, measured, silent_in_covering_exam, ...
    evidence: list[str] = field(default_factory=list)    # fact ids
    note: str = ""
    level: str = INFERENCE


@dataclass
class Disease(_Record):
    disease_id: str
    branch: str
    words: str
    origin: str
    provenance: Provenance | None = None
    concept: Concept | None = None
    status_words: str = ""                      # "pregressa", "sospetta", "in esiti di", as written
    date: str = ""
    manifestations: list[Manifestation] = field(default_factory=list)
    states: list[ManifestationState] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)   # for a candidate: what proposed it

    def state_of(self, manifestation_id: str) -> ManifestationState | None:
        return next((s for s in self.states if s.manifestation_id == manifestation_id), None)


@dataclass
class DrugProfile(_Record):
    """What the sources say of a drug the patient takes (knowledge): the ingredient as AIFA writes it, its
    international name, its ATC code, and the sections of its label with their citation (``connectors.
    drug_sources``). ``resolution`` says how the name was reached and whether it is in doubt."""

    medication_id: str
    ingredient: str
    inn: str = ""
    atc: str = ""
    resolution: str = ""                        # wikidata+ema, wikidata, ema, doubt, unresolved
    sections: dict[str, list[str]] = field(default_factory=dict)
    citations: list[dict[str, Any]] = field(default_factory=list)
    level: str = KNOWLEDGE


@dataclass
class DrugInteraction(_Record):
    """A label that names another drug of the patient (inference): the sentence, its section, its citation."""

    medication: str
    other: str
    section: str
    sentence: str
    citation: dict[str, Any] = field(default_factory=dict)
    level: str = INFERENCE


@dataclass
class DrugEffect(_Record):
    """A label that names a finding of the patient among a drug's effects (inference), with the time between the
    start of the drug and the finding: ``after_start``, ``before_start`` or ``unknown``."""

    medication: str
    finding: str
    section: str
    sentence: str
    timing: str = "unknown"
    citation: dict[str, Any] = field(default_factory=dict)
    level: str = INFERENCE


@dataclass
class KnowledgeSnapshot(_Record):
    """What knowledge the case used: each source with its version, and when it was taken."""

    sources: dict[str, str] = field(default_factory=dict)
    taken: str = ""


_NESTED = {
    ("Finding", "provenance"): Provenance, ("Finding", "concept"): Concept,
    ("Measurement", "provenance"): Provenance, ("Measurement", "code"): Concept,
    ("Medication", "provenance"): Provenance, ("Medication", "codes"): Concept,
    ("Manifestation", "concept"): Concept,
    ("Disease", "provenance"): Provenance, ("Disease", "concept"): Concept,
    ("Disease", "manifestations"): Manifestation, ("Disease", "states"): ManifestationState,
}


class CaseGraphError(ValueError):
    pass


@dataclass
class CaseGraph(_Record):
    case_id: str
    documents: list[Document] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    measurements: list[Measurement] = field(default_factory=list)
    imprints: list[Imprint] = field(default_factory=list)
    medications: list[Medication] = field(default_factory=list)
    diseases: list[Disease] = field(default_factory=list)
    snapshot: KnowledgeSnapshot = field(default_factory=KnowledgeSnapshot)
    schema: str = SCHEMA_VERSION
    drug_profiles: list[DrugProfile] = field(default_factory=list)
    drug_interactions: list[DrugInteraction] = field(default_factory=list)
    drug_effects: list[DrugEffect] = field(default_factory=list)

    # -- the branches ---------------------------------------------------------------------------------------------
    def branch(self, name: str) -> list[Any]:
        if name == DRUGS:
            return list(self.medications)
        if name == EXAMS:
            return list(self.documents)
        if name in DISEASE_BRANCHES:
            return [d for d in self.diseases if d.branch == name]
        raise CaseGraphError(f"unknown branch {name!r}")

    def branches(self) -> dict[str, list[Any]]:
        return {name: self.branch(name) for name in BRANCHES}

    def of_document(self, document_id: str) -> dict[str, list[Any]]:
        """The sub-branch of one document: its findings, values and imprints."""
        return {"findings": [f for f in self.findings if f.document == document_id],
                "measurements": [m for m in self.measurements if m.document == document_id],
                "imprints": [i for i in self.imprints if i.document == document_id]}

    def facts(self) -> dict[str, Any]:
        """Every fact of the patient by its id."""
        out: dict[str, Any] = {}
        for item in self.findings:
            out[item.finding_id] = item
        for item in self.measurements:
            out[item.measurement_id] = item
        for item in self.imprints:
            out[item.imprint_id] = item
        for item in self.medications:
            out[item.medication_id] = item
        return out

    def document(self, document_id: str) -> Document | None:
        return next((d for d in self.documents if d.document_id == document_id), None)

    def disease(self, disease_id: str) -> Disease | None:
        return next((d for d in self.diseases if d.disease_id == disease_id), None)

    # -- what may enter -------------------------------------------------------------------------------------------
    def add_disease(self, disease: Disease) -> Disease:
        if disease.branch == CANDIDATE_DISEASES:
            raise CaseGraphError("a candidate enters only through add_candidate, from the diagnostic path")
        if disease.branch not in DISEASE_BRANCHES:
            raise CaseGraphError(f"unknown disease branch {disease.branch!r}")
        self._unique(disease.disease_id)
        self.diseases.append(disease)
        return disease

    def add_candidate(self, disease: Disease) -> Disease:
        """A disease nobody named, found by the diagnostic path: it enters the candidate branch with what proposed
        it (``citations``), then follows the same schema as the others."""
        if not disease.citations:
            raise CaseGraphError("a candidate carries what proposed it (citations)")
        disease.branch, disease.origin = CANDIDATE_DISEASES, DIAGNOSTIC_PATH
        self._unique(disease.disease_id)
        self.diseases.append(disease)
        return disease

    def _unique(self, disease_id: str) -> None:
        if self.disease(disease_id) is not None:
            raise CaseGraphError(f"disease {disease_id!r} is already in the graph")

    # -- checks ---------------------------------------------------------------------------------------------------
    def validate(self) -> list[str]:
        """What breaks the rules of the graph; empty when it holds."""
        problems: list[str] = []
        documents = {d.document_id for d in self.documents}
        ids: set[str] = set()
        for fact_id, fact in self.facts().items():
            if fact_id in ids:
                problems.append(f"{fact_id}: id used twice")
            ids.add(fact_id)
            provenance = getattr(fact, "provenance", None)
            if isinstance(fact, Imprint):
                if fact.document not in documents:
                    problems.append(f"{fact_id}: document {fact.document!r} is not in the case")
                continue
            if provenance is None or not str(provenance.words).strip():
                problems.append(f"{fact_id}: no words")
            elif provenance.document not in documents and not isinstance(fact, Medication):
                problems.append(f"{fact_id}: document {provenance.document!r} is not in the case")
            if fact.level != FACT:
                problems.append(f"{fact_id}: a fact at level {fact.level!r}")
        for finding in self.findings:
            p = finding.provenance
            if p.start is not None and p.end is not None and p.end - p.start != len(p.words):
                problems.append(f"{finding.finding_id}: offsets {p.start}..{p.end} do not hold the words")
        medications = {m.medication_id for m in self.medications}
        for profile in self.drug_profiles:
            if profile.medication_id not in medications:
                problems.append(f"drug profile of an unknown medication {profile.medication_id}")
        for x in self.drug_interactions:
            if x.medication not in medications or x.other not in medications:
                problems.append(f"interaction between drugs not in the case: {x.medication}, {x.other}")
            if not x.sentence.strip():
                problems.append(f"interaction {x.medication}-{x.other} without its sentence")
        findings = {f.finding_id for f in self.findings}
        for x in self.drug_effects:
            if x.medication not in medications or x.finding not in findings:
                problems.append(f"drug effect on facts not in the case: {x.medication}, {x.finding}")
            if not x.sentence.strip():
                problems.append(f"drug effect {x.medication}-{x.finding} without its sentence")
        for disease in self.diseases:
            if disease.branch == CANDIDATE_DISEASES and disease.origin != DIAGNOSTIC_PATH:
                problems.append(f"{disease.disease_id}: a candidate not found by the diagnostic path")
            if disease.branch != CANDIDATE_DISEASES and disease.origin == DIAGNOSTIC_PATH:
                problems.append(f"{disease.disease_id}: the diagnostic path writes only candidates")
            names = {m.manifestation_id for m in disease.manifestations}
            for m in disease.manifestations:
                if not m.words.strip():
                    problems.append(f"{disease.disease_id}/{m.manifestation_id}: a manifestation with no words")
                if m.typicality and m.typicality not in TYPICALITIES:
                    problems.append(f"{disease.disease_id}/{m.manifestation_id}: typicality {m.typicality!r}")
                if not m.citations:
                    problems.append(f"{disease.disease_id}/{m.manifestation_id}: a manifestation with no source")
            for s in disease.states:
                if s.manifestation_id not in names:
                    problems.append(f"{disease.disease_id}: state of an unknown manifestation {s.manifestation_id}")
                if s.state not in STATES:
                    problems.append(f"{disease.disease_id}/{s.manifestation_id}: state {s.state!r}")
                missing = [e for e in s.evidence if e not in ids]
                if missing:
                    problems.append(f"{disease.disease_id}/{s.manifestation_id}: evidence not in the case {missing}")
        return problems

    # -- plain data -----------------------------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseGraph:
        if data.get("schema", SCHEMA_VERSION) not in READABLE_SCHEMAS:
            raise CaseGraphError(f"schema {data.get('schema')!r}, expected one of {READABLE_SCHEMAS}")
        return cls(
            case_id=data["case_id"],
            documents=[Document.from_dict(d) for d in data.get("documents", [])],
            findings=[Finding.from_dict(d) for d in data.get("findings", [])],
            measurements=[Measurement.from_dict(d) for d in data.get("measurements", [])],
            imprints=[Imprint.from_dict(d) for d in data.get("imprints", [])],
            medications=[Medication.from_dict(d) for d in data.get("medications", [])],
            diseases=[Disease.from_dict(d) for d in data.get("diseases", [])],
            snapshot=KnowledgeSnapshot.from_dict(data.get("snapshot", {})),
            schema=SCHEMA_VERSION,
            drug_profiles=[DrugProfile.from_dict(d) for d in data.get("drug_profiles", [])],
            drug_interactions=[DrugInteraction.from_dict(d) for d in data.get("drug_interactions", [])],
            drug_effects=[DrugEffect.from_dict(d) for d in data.get("drug_effects", [])],
        )

    def summary(self) -> dict[str, Any]:
        """Counts only, no words: what may travel with the case result."""
        return {"schema": self.schema, "documents": len(self.documents), "findings": len(self.findings),
                "measurements": len(self.measurements), "imprints": len(self.imprints),
                "medications": len(self.medications),
                **{name: len(self.branch(name)) for name in DISEASE_BRANCHES},
                "manifestations": sum(len(d.manifestations) for d in self.diseases),
                "states": {s: sum(1 for d in self.diseases for x in d.states if x.state == s) for s in STATES},
                "drug_profiles": len(self.drug_profiles), "drug_interactions": len(self.drug_interactions),
                "drug_effects": len(self.drug_effects),
                "knowledge_sources": dict(self.snapshot.sources)}

    # -- Phenopacket v2 ---------------------------------------------------------------------------------------------
    def to_phenopacket(self) -> dict[str, Any]:
        from .phenopacket import to_phenopacket

        return to_phenopacket(self)

    @classmethod
    def from_phenopacket(cls, packet: dict[str, Any]) -> CaseGraph:
        from .phenopacket import from_phenopacket

        return from_phenopacket(packet)


__all__ = [name for name in dir() if not name.startswith("_")]
