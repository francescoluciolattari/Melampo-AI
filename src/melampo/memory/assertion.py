"""Deterministic assertion detection over clinical text.

Exact concept matching resolves "denies fever" and "presents with fever"
identically, so the negated finding enters as a present one. In this system that
error does not stop at extraction: a false finding becomes a graph entry point,
then a path, then a hypothesis in the differential, with provenance intact to
the character. It is the same shape of error the interval representation was
built to prevent, arriving through the extraction door.

The approach follows the ConText line — cue lists with bounded scope and
termination — because a rule system here is high precision, auditable, and
reproducible, and because it doubles as the baseline any model must beat and as
a generator of training annotations. That bootstrap is the documented strategy
behind NegBERT.

Four axes are detected, matching the i2b2 assertion vocabulary:

- **polarity** — affirmed or negated;
- **certainty** — factual, possible, or hypothetical;
- **experiencer** — the patient, or someone else;
- **temporality** — current or historical.

A fifth, **source**, separates what an examination establishes from what a
patient reports. It matters because objectivity does not override subjectivity
uniformly: a clinician can contradict "I have no arrhythmia" because the ECG is
observable, and cannot contradict "I have no pain", because pain is not. On
observable signs the observation wins; on experienced symptoms the person
experiencing them is the authoritative source.

The output is an interval and an epistemic state, never a scalar. A scalar would
collapse "looked for and absent" into the same number as "nobody looked", which
is the distinction the whole representation exists to carry.

Cue lists are supplied, not hard-coded to one language: English and Italian sets
are provided and either can be replaced, so the corpus language stays an open
decision.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

POLARITY_AFFIRMED = "affirmed"
POLARITY_NEGATED = "negated"

CERTAINTY_FACTUAL = "factual"
CERTAINTY_POSSIBLE = "possible"
CERTAINTY_HYPOTHETICAL = "hypothetical"

EXPERIENCER_PATIENT = "patient"
EXPERIENCER_OTHER = "other"

TEMPORALITY_CURRENT = "current"
TEMPORALITY_HISTORICAL = "historical"

SOURCE_OBJECTIVE = "objective"
SOURCE_SUBJECTIVE = "subjective"
SOURCE_UNSPECIFIED = "unspecified"

STATE_DOCUMENTED = "documented"
STATE_UNCERTAIN_POSITIVE = "uncertain_positive"
STATE_WEAK_NEGATION = "weak_negation"
STATE_DOCUMENTED_EXCLUSION = "documented_exclusion"
STATE_GAP = "gap"


@dataclass(frozen=True)
class CueSet:
    """Trigger phrases for one language, with the direction each applies in."""

    negation_before: tuple[str, ...] = ()
    negation_after: tuple[str, ...] = ()
    hypothetical: tuple[str, ...] = ()
    possible: tuple[str, ...] = ()
    experiencer_other: tuple[str, ...] = ()
    historical: tuple[str, ...] = ()
    objective: tuple[str, ...] = ()
    subjective: tuple[str, ...] = ()
    terminators: tuple[str, ...] = ()


ENGLISH_CUES = CueSet(
    negation_before=(
        "no", "not", "denies", "denied", "without", "absent", "negative for",
        "no evidence of", "free of", "rules out", "ruled out", "resolved",
    ),
    negation_after=(
        "was ruled out", "is ruled out", "not seen", "not present", "absent",
        # Absence carried by the clinical term itself, with no negation marker.
        # Enumerable — a few dozen per language — and the part neither a lexicon
        # of negation words nor a parser reaches, because nothing in the sentence
        # signals negation except what the term means.
        "unremarkable", "within normal limits", "grossly normal", "non contributory",
        "noncontributory", "clear", "patent", "intact", "negative",
    ),
    hypothetical=("rule out", "r/o", "evaluate for", "screen for", "if", "to exclude", "workup for"),
    possible=("possible", "probable", "suspected", "suspicious for", "cannot be excluded", "may", "might"),
    experiencer_other=(
        "family history", "mother", "father", "sister", "brother", "sibling",
        "parent", "grandmother", "grandfather", "aunt", "uncle", "cousin", "in the family",
    ),
    historical=("history of", "previous", "prior", "past", "formerly", "in the past", "resolved", "status post"),
    objective=("examination", "exam shows", "on examination", "imaging", "radiograph", "ecg", "laboratory", "reveals", "demonstrates"),
    subjective=("reports", "complains", "denies", "describes", "states", "refers"),
    terminators=("but", "however", "although", "though", "except", "aside from", "otherwise"),
)

ITALIAN_CUES = CueSet(
    negation_before=(
        "non", "nega", "negato", "senza", "assente", "assenza di", "nessun", "nessuna",
        "negativo per", "non si evidenzia", "esclude", "escluso", "risolto",
    ),
    negation_after=(
        "non riscontrato", "non presente", "assente", "escluso",
        "nella norma", "nei limiti", "nei limiti di norma", "regolare", "libero",
        "indenne", "negativo", "silente",
    ),
    hypothetical=("da escludere", "per escludere", "in caso di", "valutare per", "screening per", "sospetto di"),
    possible=("possibile", "probabile", "sospetto", "non si puo escludere", "potrebbe", "verosimile"),
    experiencer_other=(
        "familiarita", "madre", "padre", "sorella", "fratello", "genitore",
        "nonna", "nonno", "zia", "zio", "cugino", "in famiglia", "anamnesi familiare",
    ),
    historical=("anamnesi", "pregresso", "pregressa", "precedente", "in passato", "risolto", "esiti di"),
    objective=("esame obiettivo", "all esame", "imaging", "radiografia", "ecg", "laboratorio", "si riscontra", "si rileva", "evidenzia"),
    subjective=("riferisce", "lamenta", "nega", "descrive", "sostiene"),
    terminators=("ma", "tuttavia", "sebbene", "benche", "eccetto", "a parte", "peraltro"),
)

CUE_SETS: dict[str, CueSet] = {"en": ENGLISH_CUES, "it": ITALIAN_CUES}


def select_cues(language: str) -> CueSet:
    """Cue set for a document's language, defaulting to English.

    Selection is per document rather than per sentence, and the sets are not
    merged. Merging looks convenient and is not: cues collide across languages —
    Italian "ma" is a terminator while English "ma" is nothing, Italian "non" is
    negation while English "non" appears inside words — so a merged set fires on
    text it was not written for, and the failure is silent.
    """
    return CUE_SETS.get((language or "").strip().lower()[:2], ENGLISH_CUES)


# Category to interval. Deterministic and versioned in
# docs/semantic_extraction_decision_record.md, so the numbers are a documented
# decision rather than a model's opinion.
ASSERTION_INTERVALS: dict[tuple[str, str], tuple[float, float, str]] = {
    (POLARITY_AFFIRMED, SOURCE_OBJECTIVE): (0.90, 1.00, STATE_DOCUMENTED),
    (POLARITY_AFFIRMED, SOURCE_SUBJECTIVE): (0.50, 0.90, STATE_UNCERTAIN_POSITIVE),
    (POLARITY_AFFIRMED, SOURCE_UNSPECIFIED): (0.60, 0.95, STATE_UNCERTAIN_POSITIVE),
    (POLARITY_NEGATED, SOURCE_OBJECTIVE): (0.00, 0.05, STATE_DOCUMENTED_EXCLUSION),
    (POLARITY_NEGATED, SOURCE_SUBJECTIVE): (0.00, 0.30, STATE_WEAK_NEGATION),
    (POLARITY_NEGATED, SOURCE_UNSPECIFIED): (0.00, 0.20, STATE_WEAK_NEGATION),
}

HISTORICAL_CEILING = 0.35


@dataclass(frozen=True)
class AssertionStatus:
    """How a finding is asserted, and what interval that implies."""

    polarity: str = POLARITY_AFFIRMED
    certainty: str = CERTAINTY_FACTUAL
    experiencer: str = EXPERIENCER_PATIENT
    temporality: str = TEMPORALITY_CURRENT
    source: str = SOURCE_UNSPECIFIED
    cues: tuple[str, ...] = ()

    @property
    def is_patient_finding(self) -> bool:
        """Whether this belongs among the patient's findings at all.

        A condition attributed to someone else is not a finding of this patient
        at any magnitude. Traversal is binary: a finding is an entry point, and
        the graph would be walked from a condition the patient does not have.
        """
        return self.experiencer == EXPERIENCER_PATIENT

    def bounds(self) -> tuple[float, float]:
        """Interval implied by the assertion, before any graph evidence."""
        if self.certainty in {CERTAINTY_HYPOTHETICAL, CERTAINTY_POSSIBLE}:
            # An open question is not evidence of presence or of absence.
            return (0.0, 1.0)
        lower, upper, _ = ASSERTION_INTERVALS[(self.polarity, self.source)]
        if self.temporality == TEMPORALITY_HISTORICAL and self.polarity == POLARITY_AFFIRMED:
            upper = min(upper, HISTORICAL_CEILING)
            lower = min(lower, upper)
        return (lower, upper)

    def state(self) -> str:
        if self.certainty in {CERTAINTY_HYPOTHETICAL, CERTAINTY_POSSIBLE}:
            return STATE_GAP
        return ASSERTION_INTERVALS[(self.polarity, self.source)][2]

    def as_dict(self) -> dict[str, Any]:
        lower, upper = self.bounds()
        return {
            "polarity": self.polarity,
            "certainty": self.certainty,
            "experiencer": self.experiencer,
            "temporality": self.temporality,
            "source": self.source,
            "lower": round(lower, 3),
            "upper": round(upper, 3),
            "state": self.state(),
            "is_patient_finding": self.is_patient_finding,
            "cues": list(self.cues),
        }


@dataclass
class AssertionDetector:
    """Cue-based assertion detection with bounded scope.

    Scope runs from a cue to the end of the clause or to a terminator, so
    "denies fever but reports cough" negates only the first finding. Without
    termination a single cue would negate the rest of the sentence, which is the
    classic failure of naive negation detection.
    """

    cues: CueSet = ENGLISH_CUES
    scope_characters: int = 120

    def detect(self, text: str, char_start: int, char_end: int) -> AssertionStatus:
        """Classify how the span between the offsets is asserted."""
        normalised = _normalise(text)
        clause_start, clause_end = self._clause_bounds(normalised, char_start, char_end)
        before = normalised[max(clause_start, char_start - self.scope_characters) : char_start]
        after = normalised[char_end : min(clause_end, char_end + self.scope_characters)]
        clause = normalised[clause_start:clause_end]

        fired: list[str] = []

        experiencer = EXPERIENCER_PATIENT
        for cue in self.cues.experiencer_other:
            if cue in clause:
                experiencer = EXPERIENCER_OTHER
                fired.append(f"experiencer:{cue}")
                break

        certainty = CERTAINTY_FACTUAL
        for cue in self.cues.hypothetical:
            if _contains_cue(before, cue):
                certainty = CERTAINTY_HYPOTHETICAL
                fired.append(f"hypothetical:{cue}")
                break
        if certainty == CERTAINTY_FACTUAL:
            for cue in self.cues.possible:
                if _contains_cue(before, cue) or _contains_cue(after, cue):
                    certainty = CERTAINTY_POSSIBLE
                    fired.append(f"possible:{cue}")
                    break

        polarity = POLARITY_AFFIRMED
        for cue in self.cues.negation_before:
            if _contains_cue(before, cue):
                polarity = POLARITY_NEGATED
                fired.append(f"negation:{cue}")
                break
        if polarity == POLARITY_AFFIRMED:
            for cue in self.cues.negation_after:
                if _contains_cue(after, cue):
                    polarity = POLARITY_NEGATED
                    fired.append(f"negation_after:{cue}")
                    break

        temporality = TEMPORALITY_CURRENT
        for cue in self.cues.historical:
            if _contains_cue(before, cue):
                temporality = TEMPORALITY_HISTORICAL
                fired.append(f"historical:{cue}")
                break

        source = SOURCE_UNSPECIFIED
        for cue in self.cues.objective:
            if cue in clause:
                source = SOURCE_OBJECTIVE
                fired.append(f"objective:{cue}")
                break
        if source == SOURCE_UNSPECIFIED:
            for cue in self.cues.subjective:
                if cue in clause:
                    source = SOURCE_SUBJECTIVE
                    fired.append(f"subjective:{cue}")
                    break

        return AssertionStatus(
            polarity=polarity,
            certainty=certainty,
            experiencer=experiencer,
            temporality=temporality,
            source=source,
            cues=tuple(fired),
        )

    def _clause_bounds(self, text: str, char_start: int, char_end: int) -> tuple[int, int]:
        """Bound the scope at sentence edges and at adversative terminators."""
        start = max(text.rfind(".", 0, char_start), text.rfind(";", 0, char_start)) + 1
        end_candidates = [text.find(".", char_end), text.find(";", char_end)]
        end = min([value for value in end_candidates if value >= 0], default=len(text))

        for terminator in self.cues.terminators:
            needle = f" {terminator} "
            position = text.rfind(needle, start, char_start)
            if position >= 0:
                start = max(start, position + len(needle))
            position = text.find(needle, char_end, end)
            if position >= 0:
                end = min(end, position)
        return (start, end)


DECIDED_BY_RULES = "rules"
DECIDED_BY_FALLBACK = "fallback"
DECIDED_BY_DEFAULT = "default"


@dataclass(frozen=True)
class ResolvedAssertion:
    """An assertion together with which layer decided it, and whether it is explained."""

    status: AssertionStatus
    decided_by: str

    @property
    def is_explained(self) -> bool:
        """Whether the decision can be traced to the words that produced it.

        Rules explain themselves — the cue is recorded. A fallback does not, but
        it only runs where no rule fired, so nothing that was previously
        explainable becomes opaque: the alternative there was not a worse
        explanation but no detection at all.
        """
        return self.decided_by == DECIDED_BY_RULES

    def as_dict(self) -> dict[str, Any]:
        return {**self.status.as_dict(), "decided_by": self.decided_by, "explained": self.is_explained}


@dataclass
class LayeredAssertionResolver:
    """Rules first, fallback only on what the rules do not cover.

    The two are not alternatives. Where a cue fires the rule decides and the
    explanation survives; where none fires — a paraphrase the lexicon lacks, a
    construction nobody anticipated — the fallback runs, and there the choice
    was never between a good explanation and a poor one but between a decision
    and silence.

    Coverage therefore rises without the explained cases becoming opaque, and
    the fallback is measured only on the residue, which is the subset where it
    would actually be used.
    """

    detector: AssertionDetector = field(default_factory=AssertionDetector)
    fallback: Any = None

    def resolve(self, text: str, char_start: int, char_end: int) -> ResolvedAssertion:
        status = self.detector.detect(text, char_start, char_end)
        if status.cues:
            return ResolvedAssertion(status=status, decided_by=DECIDED_BY_RULES)
        if self.fallback is not None:
            proposed = self.fallback(text, char_start, char_end)
            if isinstance(proposed, AssertionStatus):
                return ResolvedAssertion(status=proposed, decided_by=DECIDED_BY_FALLBACK)
        return ResolvedAssertion(status=status, decided_by=DECIDED_BY_DEFAULT)


@dataclass
class ResidueReport:
    """How much of a corpus the rules leave undecided.

    This is the number that decides whether a learned classifier is worth
    building at all. A small residue means the rules and the paraphrase lexicon
    have closed the gap and the classifier would be a solution to a problem that
    no longer exists.
    """

    total: int = 0
    decided_by_rules: int = 0
    residue: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return self.decided_by_rules / self.total if self.total else 0.0

    @property
    def residue_fraction(self) -> float:
        return 1.0 - self.coverage if self.total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "decided_by_rules": self.decided_by_rules,
            "coverage": round(self.coverage, 4),
            "residue_fraction": round(self.residue_fraction, 4),
            "residue_examples": self.residue[:20],
        }


def measure_residue(
    detector: AssertionDetector, samples: Sequence[tuple[str, int, int]]
) -> ResidueReport:
    """Fraction of spans the rules decide, over a sample of (text, start, end).

    Measured before building anything, because the residue is the only evidence
    that a classifier is needed — and it may show that it is not.
    """
    report = ResidueReport()
    for text, start, end in samples:
        report.total += 1
        if detector.detect(text, start, end).cues:
            report.decided_by_rules += 1
        else:
            report.residue.append(text[max(0, start - 40) : end + 40].strip())
    return report


def _normalise(text: str) -> str:
    """Lowercase and flatten punctuation to spaces, preserving offsets."""
    return "".join(
        character.lower() if character.isalnum() or character in ".;" else " " for character in text
    )


def _contains_cue(window: str, cue: str) -> bool:
    return f" {cue} " in f" {window.strip()} " or window.strip().startswith(f"{cue} ") or window.strip().endswith(f" {cue}")
