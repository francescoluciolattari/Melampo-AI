"""Bridge clinical text to concept graph nodes.

The graph and the case corpus were speaking different languages. Graph nodes are
HPO identifiers and disease names; the pipeline extracted entities against
sixteen hand-written English terms mapped to invented references
(``"cough": "Symptom:Cough"``). Nothing connected the two, so every traversal
started from a concept the graph had never heard of.

That failure is dangerous because of how it presents. The dream branch produces
nothing, local density reads zero, and the obvious reading is "the graph is too
sparse" — while the graph holds hundreds of thousands of edges, none of them
reachable. A resolution gap and a coverage gap look identical from the outside
and call for opposite work, so this module reports which one it is.

**Matching is exact and deterministic.** Surface forms are normalised and
matched against term names and synonyms; nothing is approximate. A fuzzy match
that silently resolves the wrong concept is worse than no match: an unresolved
finding announces itself, a mis-resolved one propagates into a path, a
hypothesis, and a provenance record that all look well-formed.

Ambiguity is reported rather than broken. When one surface form maps to several
terms the resolver says so instead of choosing, because choosing would be a
clinical judgement made silently by a tie-break rule.

The module is language-agnostic: the index is built from whichever OBO file is
supplied, so an Italian or French HPO release works without code changes.
"""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

SCOPE_EXACT = "EXACT"
SCOPE_NARROW = "NARROW"
SCOPE_BROAD = "BROAD"
SCOPE_RELATED = "RELATED"

SAFE_SCOPES = frozenset({SCOPE_EXACT})

CLINICAL_MODIFIER_ROOT = "HP:0012823"
INHERITANCE_ROOT = "HP:0000005"

ROLE_FINDING = "finding"
ROLE_MODIFIER = "modifier"
ROLE_INHERITANCE = "inheritance"

MATCH_TRANSLATION_OFFICIAL = "translation_official"
MATCH_TRANSLATION_CANDIDATE = "translation_candidate"

TRANSLATION_STATUS_OFFICIAL = "OFFICIAL"

# Match kinds that did not come from a curated English source. A resolution
# obtained through an unreviewed machine translation is not equivalent to one
# obtained from the ontology's own label, and the difference travels with the
# result rather than being averaged away.
UNVERIFIED_MATCH_KINDS = frozenset({MATCH_TRANSLATION_CANDIDATE})

MATCH_NAME = "name"
MATCH_SYNONYM = "synonym"

OUTCOME_RESOLVED = "resolved"
OUTCOME_AMBIGUOUS = "ambiguous"
OUTCOME_UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class OntologyTerm:
    """One term from an OBO release."""

    term_id: str
    name: str
    synonyms: tuple[tuple[str, str], ...] = ()
    alt_ids: tuple[str, ...] = ()
    obsolete: bool = False
    parents: tuple[str, ...] = ()

    def surface_forms(self, scopes: Iterable[str] = SAFE_SCOPES) -> list[tuple[str, str]]:
        """Text forms that may stand for this term, with how each was obtained."""
        allowed = set(scopes)
        forms: list[tuple[str, str]] = []
        if self.name:
            forms.append((self.name, MATCH_NAME))
        forms.extend((text, MATCH_SYNONYM) for text, scope in self.synonyms if scope in allowed)
        return forms


def parse_obo(lines: Iterable[str]) -> Iterator[OntologyTerm]:
    """Parse ``[Term]`` stanzas from an OBO file.

    Obsolete terms are emitted with their flag set rather than dropped, so a
    caller can tell "retired concept" from "never existed" — the same
    distinction the graph draws between a documented exclusion and a gap.
    """
    term_id = ""
    name = ""
    synonyms: list[tuple[str, str]] = []
    alt_ids: list[str] = []
    parents: list[str] = []
    obsolete = False
    inside = False

    def flush() -> OntologyTerm | None:
        if not inside or not term_id:
            return None
        return OntologyTerm(
            term_id=term_id,
            name=name,
            synonyms=tuple(synonyms),
            alt_ids=tuple(alt_ids),
            obsolete=obsolete,
            parents=tuple(parents),
        )

    for raw in lines:
        line = raw.rstrip("\n")
        if line.startswith("["):
            finished = flush()
            if finished is not None:
                yield finished
            inside = line.strip() == "[Term]"
            term_id, name, synonyms, alt_ids, parents, obsolete = "", "", [], [], [], False
            continue
        if not inside or not line:
            continue
        key, _, value = line.partition(": ")
        if key == "id":
            term_id = value.strip()
        elif key == "name":
            name = value.strip()
        elif key == "alt_id":
            alt_ids.append(value.strip())
        elif key == "is_a":
            parents.append(value.split("!")[0].strip())
        elif key == "is_obsolete":
            obsolete = value.strip().lower() == "true"
        elif key == "synonym":
            parsed = _parse_synonym(value)
            if parsed is not None:
                synonyms.append(parsed)

    finished = flush()
    if finished is not None:
        yield finished


def _parse_synonym(value: str) -> tuple[str, str] | None:
    """Read ``"text" SCOPE type []`` into (text, scope)."""
    text = value.strip()
    if not text.startswith('"'):
        return None
    closing = text.find('"', 1)
    if closing < 0:
        return None
    surface = text[1:closing]
    remainder = text[closing + 1 :].strip().split()
    scope = remainder[0] if remainder else SCOPE_RELATED
    return (surface, scope) if surface else None


@dataclass
class TermIndex:
    """Surface form to term identifier, built from an ontology release."""

    by_id: dict[str, OntologyTerm] = field(default_factory=dict)
    by_surface: dict[str, list[str]] = field(default_factory=dict)
    match_kind: dict[str, str] = field(default_factory=dict)
    _ancestors: dict[str, frozenset[str]] = field(default_factory=dict, repr=False)

    @classmethod
    def from_terms(
        cls,
        terms: Iterable[OntologyTerm],
        *,
        scopes: Iterable[str] = SAFE_SCOPES,
        include_obsolete: bool = False,
    ) -> "TermIndex":
        index = cls()
        for term in terms:
            if term.obsolete and not include_obsolete:
                continue
            index.by_id[term.term_id] = term
            for alt in term.alt_ids:
                index.by_id.setdefault(alt, term)
            for surface, kind in term.surface_forms(scopes):
                key = normalise_surface(surface)
                if not key:
                    continue
                holders = index.by_surface.setdefault(key, [])
                if term.term_id not in holders:
                    holders.append(term.term_id)
                index.match_kind.setdefault(key, kind)
        return index

    @classmethod
    def from_obo(cls, lines: Iterable[str], **kwargs: Any) -> "TermIndex":
        return cls.from_terms(parse_obo(lines), **kwargs)

    def add_translations(
        self,
        rows: Iterable[dict[str, str]],
        *,
        include_candidate: bool = True,
    ) -> dict[str, int]:
        """Merge Babelon translation rows into the surface index.

        The identifier is the pivot, so a translated surface form points at the
        same term as its English label and the knowledge layer stays in one
        language. That matters beyond convenience: ontologies, terminologies and
        the literature that supplies likelihoods are English, so translating the
        graph would mean maintaining two knowledge bases instead of two
        vocabularies.

        Translation status is preserved as the match kind. An unreviewed machine
        translation resolves, but the result says so, and a caller can require
        curated matches where a wrong resolution would be costlier than none.
        """
        added = {"official": 0, "candidate": 0, "skipped": 0}
        for row in rows:
            term_id = str(row.get("subject_id", "")).strip()
            surface = str(row.get("translation_value", "")).strip()
            status = str(row.get("translation_status", "")).strip().upper()
            if not term_id or not surface or term_id not in self.by_id:
                added["skipped"] += 1
                continue
            official = status == TRANSLATION_STATUS_OFFICIAL
            if not official and not include_candidate:
                added["skipped"] += 1
                continue
            key = normalise_surface(surface)
            if not key:
                added["skipped"] += 1
                continue
            holders = self.by_surface.setdefault(key, [])
            if term_id not in holders:
                holders.append(term_id)
            self.match_kind.setdefault(
                key, MATCH_TRANSLATION_OFFICIAL if official else MATCH_TRANSLATION_CANDIDATE
            )
            added["official" if official else "candidate"] += 1
        return added

    def label_map(self) -> dict[str, str]:
        """Identifier to preferred label, for substitution during graph import."""
        return {term_id: term.name for term_id, term in self.by_id.items() if term.name}

    def lookup(self, surface: str) -> list[str]:
        return list(self.by_surface.get(normalise_surface(surface), []))

    def ancestors(self, term_id: str) -> frozenset[str]:
        """Transitive ``is_a`` closure, memoised.

        The hierarchy is what makes the modifier rule deterministic. HPO already
        separates clinical modifiers from phenotypic abnormalities in its own
        structure, so deciding whether a term is a finding is a lookup rather
        than a judgement — no model, and complete coverage over the branch.
        """
        cached = self._ancestors.get(term_id)
        if cached is not None:
            return cached
        seen: set[str] = set()
        frontier = [term_id]
        while frontier:
            current = frontier.pop()
            term = self.by_id.get(current)
            if term is None:
                continue
            for parent in term.parents:
                if parent not in seen:
                    seen.add(parent)
                    frontier.append(parent)
        result = frozenset(seen)
        self._ancestors[term_id] = result
        return result

    def descends_from(self, term_id: str, root: str) -> bool:
        return term_id == root or root in self.ancestors(term_id)

    def role_of(self, term_id: str) -> str:
        """Whether a term is a finding, a modifier, or an inheritance statement.

        Modifiers such as *Bilateral* and *Progressive* qualify a finding; they
        are not findings themselves, and admitting them as graph nodes would
        connect them to thousands of conditions, making every path through them
        uninformative — the same degeneracy an unknown edge produces.
        """
        if self.descends_from(term_id, CLINICAL_MODIFIER_ROOT):
            return ROLE_MODIFIER
        if self.descends_from(term_id, INHERITANCE_ROOT):
            return ROLE_INHERITANCE
        return ROLE_FINDING

    @property
    def size(self) -> int:
        return len(self.by_id)

    @property
    def surface_count(self) -> int:
        return len(self.by_surface)


@dataclass(frozen=True)
class ResolvedConcept:
    """A surface form matched to a graph concept, with its position in the source."""

    surface: str
    term_id: str
    label: str
    match_kind: str
    char_start: int | None = None
    char_end: int | None = None
    role: str = ROLE_FINDING

    @property
    def is_verified_match(self) -> bool:
        """Whether the surface form came from a curated source.

        An unreviewed machine translation can resolve correctly and often does,
        but a wrong resolution propagates into a path, a hypothesis and a
        provenance record that all look well-formed, so the distinction is kept
        rather than flattened.
        """
        return self.match_kind not in UNVERIFIED_MATCH_KINDS

    def as_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "term_id": self.term_id,
            "label": self.label,
            "match_kind": self.match_kind,
            "verified_match": self.is_verified_match,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "role": self.role,
        }


@dataclass(frozen=True)
class UnresolvedSurface:
    """A finding the index could not place, or placed ambiguously."""

    surface: str
    outcome: str
    candidates: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"surface": self.surface, "outcome": self.outcome, "candidates": list(self.candidates)}


@dataclass
class ResolutionReport:
    resolved: list[ResolvedConcept] = field(default_factory=list)
    unresolved: list[UnresolvedSurface] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.resolved) + len(self.unresolved)

    @property
    def resolution_rate(self) -> float:
        return len(self.resolved) / self.total if self.total else 0.0

    @property
    def concepts(self) -> list[str]:
        """Labels usable as graph entry points."""
        return [item.label for item in self.resolved]

    def as_dict(self) -> dict[str, Any]:
        return {
            "resolution_rate": round(self.resolution_rate, 4),
            "resolved_count": len(self.resolved),
            "unresolved_count": len(self.unresolved),
            "resolved": [item.as_dict() for item in self.resolved],
            "unresolved": [item.as_dict() for item in self.unresolved],
        }


@dataclass
class ConceptResolver:
    """Deterministic surface-to-concept resolution against a term index."""

    index: TermIndex
    max_text_matches: int = 32

    def resolve_findings(self, findings: Sequence[str]) -> ResolutionReport:
        """Resolve already-extracted findings, one surface form each.

        The denominator is the supplied list, so ``resolution_rate`` answers the
        question that matters before any traversal: how much of this case can
        the graph even be asked about.
        """
        report = ResolutionReport()
        for surface in findings:
            text = (surface or "").strip()
            if not text:
                continue
            matches = self.index.lookup(text)
            if not matches:
                report.unresolved.append(UnresolvedSurface(text, OUTCOME_UNRESOLVED))
                continue
            if len(matches) > 1:
                report.unresolved.append(
                    UnresolvedSurface(text, OUTCOME_AMBIGUOUS, tuple(sorted(matches)))
                )
                continue
            term = self.index.by_id[matches[0]]
            report.resolved.append(
                ResolvedConcept(
                    surface=text,
                    term_id=term.term_id,
                    label=term.name,
                    match_kind=self.index.match_kind.get(normalise_surface(text), MATCH_NAME),
                    role=self.index.role_of(term.term_id),
                )
            )
        return report

    def resolve_text(self, text: str) -> list[ResolvedConcept]:
        """Find concept mentions in free text, longest first, without overlap.

        Character offsets are carried through so a concept mention keeps the same
        provenance discipline as every other piece of evidence: a hypothesis
        built on it can be traced back to the words that produced it.
        """
        if not text:
            return []
        normalised, offsets = _normalise_with_offsets(text)
        haystack = f" {normalised} "
        found: list[ResolvedConcept] = []
        taken: list[tuple[int, int]] = []

        for surface in sorted(self.index.by_surface, key=len, reverse=True):
            if len(found) >= self.max_text_matches:
                break
            matches = self.index.by_surface[surface]
            if len(matches) != 1:
                continue
            start = haystack.find(f" {surface} ")
            if start < 0:
                continue
            begin = start
            end = begin + len(surface)
            if any(begin < stop and start_taken < end for start_taken, stop in taken):
                continue
            taken.append((begin, end))
            term = self.index.by_id[matches[0]]
            found.append(
                ResolvedConcept(
                    surface=surface,
                    term_id=term.term_id,
                    label=term.name,
                    match_kind=self.index.match_kind.get(surface, MATCH_NAME),
                    char_start=offsets[begin] if begin < len(offsets) else None,
                    char_end=offsets[end - 1] + 1 if end - 1 < len(offsets) else None,
                    role=self.index.role_of(term.term_id),
                )
            )
        return sorted(found, key=lambda item: (item.char_start is None, item.char_start or 0))


@dataclass(frozen=True)
class Finding:
    """A clinical finding with its modifiers held as attributes, never as nodes.

    Modifiers qualify a finding; they do not stand on their own. Admitting
    *Bilateral* as a concept in its own right would attach it to thousands of
    conditions, so every path crossing it would carry no information — the same
    degeneracy an unknown edge produces, arrived at from the other direction.

    The finding keeps its own identifier and therefore its traversability. The
    modifier travels with it as an attribute, which also means a fused
    pre-coordinated term is an optional refinement rather than a requirement:
    across the whole ontology only 52 ``Bilateral X`` and 58 ``Progressive X``
    terms exist, so fusion cannot be the primary mechanism.
    """

    concept: ResolvedConcept
    modifiers: tuple[ResolvedConcept, ...] = ()

    @property
    def term_id(self) -> str:
        return self.concept.term_id

    @property
    def label(self) -> str:
        return self.concept.label

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.concept.as_dict(),
            "modifiers": [
                {"term_id": item.term_id, "label": item.label, "surface": item.surface}
                for item in self.modifiers
            ],
        }


@dataclass
class ExtractionResult:
    """Findings, attached modifiers, and what was discarded and why."""

    findings: list[Finding] = field(default_factory=list)
    collapsed_modifiers: list[ResolvedConcept] = field(default_factory=list)
    inheritance_statements: list[ResolvedConcept] = field(default_factory=list)

    @property
    def concepts(self) -> list[str]:
        return [item.label for item in self.findings]

    def as_dict(self) -> dict[str, Any]:
        return {
            "findings": [item.as_dict() for item in self.findings],
            "collapsed_modifiers": [item.as_dict() for item in self.collapsed_modifiers],
            "inheritance_statements": [item.as_dict() for item in self.inheritance_statements],
            "finding_count": len(self.findings),
        }


def attach_modifiers(
    concepts: Sequence[ResolvedConcept], *, window: int = 80
) -> ExtractionResult:
    """Bind modifiers to the nearest finding, discarding those with no anchor.

    A modifier with no finding to qualify has nothing to mean. Rather than
    admitting it as a free-standing concept, it collapses — and is reported, so
    the discard is visible rather than silent. The window bounds attachment to
    the same clause-sized neighbourhood; a modifier far from any finding is more
    likely a coincidence than a qualification.
    """
    result = ExtractionResult()
    findings = [item for item in concepts if item.role == ROLE_FINDING]
    result.inheritance_statements = [item for item in concepts if item.role == ROLE_INHERITANCE]
    modifiers = [item for item in concepts if item.role == ROLE_MODIFIER]

    attached: dict[int, list[ResolvedConcept]] = {index: [] for index in range(len(findings))}
    for modifier in modifiers:
        index = _nearest_finding(modifier, findings, window)
        if index is None:
            result.collapsed_modifiers.append(modifier)
            continue
        attached[index].append(modifier)

    result.findings = [
        Finding(concept=finding, modifiers=tuple(attached[index]))
        for index, finding in enumerate(findings)
    ]
    return result


def _nearest_finding(
    modifier: ResolvedConcept, findings: Sequence[ResolvedConcept], window: int
) -> int | None:
    if not findings:
        return None
    if modifier.char_start is None:
        return 0 if len(findings) == 1 else None
    best_index: int | None = None
    best_distance = window + 1
    for index, finding in enumerate(findings):
        if finding.char_start is None:
            continue
        distance = min(
            abs(finding.char_start - (modifier.char_end or modifier.char_start)),
            abs((finding.char_end or finding.char_start) - modifier.char_start),
        )
        if distance <= window and distance < best_distance:
            best_index, best_distance = index, distance
    return best_index


def parse_babelon(lines: Iterable[str]) -> Iterator[dict[str, str]]:
    """Parse a Babelon translation table, tab separated with a header row."""
    columns: list[str] | None = None
    for raw in lines:
        line = raw.rstrip("\n")
        if not line:
            continue
        fields = line.split("\t")
        if columns is None:
            columns = fields
            continue
        yield dict(zip(columns, fields, strict=False))


@dataclass
class LanguageCoverage:
    """How much of the index a language reaches, and how much of it is curated."""

    language: str
    translated_terms: int
    official: int
    candidate: int
    index_size: int

    @property
    def coverage(self) -> float:
        return self.translated_terms / self.index_size if self.index_size else 0.0

    @property
    def verified_fraction(self) -> float:
        total = self.official + self.candidate
        return self.official / total if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "translated_terms": self.translated_terms,
            "official": self.official,
            "candidate": self.candidate,
            "index_size": self.index_size,
            "coverage": round(self.coverage, 4),
            "verified_fraction": round(self.verified_fraction, 4),
        }


def measure_language_coverage(
    index: TermIndex, rows: Sequence[dict[str, str]], language: str
) -> LanguageCoverage:
    """Report reachable terms per language before relying on that language.

    Coverage differs sharply between languages, and a resolution rate measured
    on one says nothing about the other. Reporting it turns "the system supports
    both" into a statement with a number attached.
    """
    official = candidate = 0
    terms: set[str] = set()
    for row in rows:
        term_id = str(row.get("subject_id", "")).strip()
        if term_id not in index.by_id or not str(row.get("translation_value", "")).strip():
            continue
        terms.add(term_id)
        if str(row.get("translation_status", "")).strip().upper() == TRANSLATION_STATUS_OFFICIAL:
            official += 1
        else:
            candidate += 1
    return LanguageCoverage(
        language=language,
        translated_terms=len(terms),
        official=official,
        candidate=candidate,
        index_size=index.size,
    )


def diagnose_empty_result(
    resolution: ResolutionReport,
    density: float | None,
    *,
    minimum_resolution: float = 0.5,
    minimum_density: float = 0.5,
) -> dict[str, Any]:
    """Say why a traversal produced nothing: resolution, coverage, or neither.

    These call for opposite work and look identical from the outside. Without
    this distinction the usual reading of an empty result is "the graph is too
    sparse", which sends effort into populating a graph that may already be
    populated and simply unreachable.
    """
    if resolution.total == 0:
        cause, action = "no_findings", "no findings were supplied to resolve"
    elif resolution.resolution_rate < minimum_resolution:
        cause = "resolution_gap"
        action = "findings could not be mapped to graph concepts; extend the term index, not the graph"
    elif density is not None and density < minimum_density:
        cause = "coverage_gap"
        action = "findings resolve but the neighbourhood is sparse; extend the graph"
    else:
        cause = "genuinely_absent"
        action = "findings resolve and the neighbourhood is covered; no candidate relation exists"

    return {
        "cause": cause,
        "action": action,
        "resolution_rate": round(resolution.resolution_rate, 4),
        "density": None if density is None else round(density, 4),
        "unresolved": [item.surface for item in resolution.unresolved],
    }


def normalise_surface(value: str) -> str:
    return " ".join(
        "".join(character if character.isalnum() else " " for character in str(value).lower()).split()
    )


def _normalise_with_offsets(text: str) -> tuple[str, list[int]]:
    """Normalise text while keeping a map back to original character positions."""
    characters: list[str] = []
    offsets: list[int] = []
    previous_space = True
    for position, character in enumerate(text):
        replacement = character.lower() if character.isalnum() else " "
        if replacement == " ":
            if previous_space:
                continue
            previous_space = True
        else:
            previous_space = False
        characters.append(replacement)
        offsets.append(position)
    while characters and characters[-1] == " ":
        characters.pop()
        offsets.pop()
    return ("".join(characters), offsets)
