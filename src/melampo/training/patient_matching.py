"""Finding a case in needs_review when case_id is unknown or absent.

Built from a design worked through directly, with two technical
corrections made before writing any matching logic, not after:

**A cryptographic hash has no meaningful notion of "close enough".**
HMAC-SHA256 is designed so a single differing input character produces a
completely different digest (the avalanche effect) -- that is exactly
what makes it safe for anonymisation. There is no "vector search" over a
hash; two hashes either match exactly or say nothing about each other.
What can legitimately be matched by proximity is the *diagnostic
question's wording*, not an identifier's hash.

**This project's only embedding function is not semantic.**
`memory/vector_memory.py`'s `_text_embedding()` is documented in its own
code as "intentionally simple and dependency-free... a deterministic
local fallback" -- a byte-position bucket sum, not a trained model. Using
it to decide whether two differently-worded diagnostic questions refer to
the same case would risk exactly the false-positive failure mode already
identified and avoided for literature search (memory/literature_index.py),
except the stakes here are higher: a false match here means training on,
or merging into, the wrong patient's case. The same principle already
proven safe for literature is reused instead -- concept overlap against
the graph's own vocabulary (mentioned_concepts()), not raw vector
similarity.

**The resulting design**: three ways to identify a pending case, tried in
order of specificity, never combined loosely.
1. `case_id` -- exact, unchanged (pending_case_router.py).
2. Fiscal code (codice fiscale) hash -- exact match, when present on both
   sides. Preferred over name+surname when available: derived from name,
   birth date and birthplace with a check character, far less ambiguous
   than two free-text fields.
3. Name hash + surname hash (normalised and hashed *separately*, not
   concatenated -- a combined hash breaks silently on reordering or
   whitespace differences) + exact date + an overlapping diagnostic
   question -- all four required together, never any one alone. The
   question overlap itself tries the graph's own concept vocabulary
   first, falling back to plain normalised-word overlap when the graph
   recognises nothing on either side (found necessary while verifying
   this with real Italian phrasing against the English-named HPO graph --
   see _questions_overlap's own docstring).
"""

from __future__ import annotations

import hashlib
import hmac
import unicodedata
from dataclasses import dataclass
from typing import Any

from ..memory.concept_paths import ConceptGraphView, mentioned_concepts

# Below this fraction of the smaller side's concepts overlapping, two
# diagnostic questions are treated as unrelated -- chosen to require
# genuine, substantial overlap rather than a single shared common term.
MIN_CONCEPT_OVERLAP_RATIO = 0.5


def normalize_name_component(value: str) -> str:
    """Lowercase, accents stripped, internal whitespace collapsed to one space, trimmed.

    Plain trim() alone is not enough: "MARIO" vs "Mario", "José" vs
    "Jose", "Mario  Rossi" (double space) vs "Mario Rossi" would all hash
    differently without this. Deliberately does not attempt to resolve
    nicknames or diminutives ("Giuseppe"/"Beppe") -- the same principle
    already applied to diagnosis recognition in case_confirmation.py:
    guessing here costs more than it earns, the risk is matching the
    wrong patient.
    """
    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(without_accents.lower().split())


def hash_identifying_value(value: str, password: str) -> str:
    """HMAC-SHA256, keyed by the same DB_PASSWORD that protects ConfirmedCaseStore -- the same construction, not a new one."""
    return hmac.new(password.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()


def hash_name_component(value: str, password: str) -> str | None:
    """None for an absent/empty component -- never hash an empty string as if it were meaningful identifying data."""
    normalized = normalize_name_component(value or "")
    if not normalized:
        return None
    return hash_identifying_value(normalized, password)


@dataclass(frozen=True)
class PatientIdentifiers:
    """What a payload can supply to identify a patient without a case_id -- built once, compared many times."""

    fiscal_code_hash: str | None = None
    name_hash: str | None = None
    surname_hash: str | None = None
    case_date: str | None = None
    diagnostic_question: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "fiscal_code_hash": self.fiscal_code_hash,
            "name_hash": self.name_hash,
            "surname_hash": self.surname_hash,
            "case_date": self.case_date,
            "diagnostic_question": self.diagnostic_question,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any], password: str) -> PatientIdentifiers:
        fiscal_code = payload.get("patient_fiscal_code")
        return cls(
            fiscal_code_hash=hash_identifying_value(normalize_name_component(fiscal_code), password) if fiscal_code else None,
            name_hash=hash_name_component(payload.get("patient_name", ""), password),
            surname_hash=hash_name_component(payload.get("patient_surname", ""), password),
            case_date=payload.get("case_date"),
            diagnostic_question=payload.get("diagnostic_question"),
        )

    @classmethod
    def from_stored(cls, data: dict[str, Any]) -> PatientIdentifiers:
        return cls(
            fiscal_code_hash=data.get("fiscal_code_hash"),
            name_hash=data.get("name_hash"),
            surname_hash=data.get("surname_hash"),
            case_date=data.get("case_date"),
            diagnostic_question=data.get("diagnostic_question"),
        )


def _questions_overlap(a: str, b: str, graph: ConceptGraphView) -> bool:
    """Whether two diagnostic questions share enough graph concepts -- or, failing that, enough normalised words -- to be treated as the same question.

    Concept overlap first: the graph's own vocabulary, most precise when
    it applies. Found necessary, not assumed, while verifying this
    function with real Italian clinical phrasing: the HPO graph's concept
    names are English ("Aortic root aneurysm"), so an Italian question
    ("Valutare aneurisma della radice aortica") recognises nothing on
    either side and the concept check alone would never match real
    Italian text -- a material limitation for a project whose deployment
    is Italian-facing, stated here rather than left to be discovered
    silently in production.

    Falls back to normalised-word overlap (lowercase, accents stripped,
    punctuation removed, split on whitespace) only when concept overlap
    found nothing on either side -- language-agnostic, transparent and
    auditable (a human can see exactly which words overlapped), not
    another embedding: the same reasoning that ruled out
    vector_memory.py's non-semantic embedding applies equally to
    reaching for one here instead of a plain, inspectable word set.
    """
    concepts_a = set(mentioned_concepts(a or "", graph, max_results=20))
    concepts_b = set(mentioned_concepts(b or "", graph, max_results=20))
    if concepts_a and concepts_b:
        overlap = concepts_a & concepts_b
        return (len(overlap) / min(len(concepts_a), len(concepts_b))) >= MIN_CONCEPT_OVERLAP_RATIO

    words_a = set(normalize_name_component(a or "").split())
    words_b = set(normalize_name_component(b or "").split())
    if not words_a or not words_b:
        return False
    overlap = words_a & words_b
    return (len(overlap) / min(len(words_a), len(words_b))) >= MIN_CONCEPT_OVERLAP_RATIO


def find_matching_pending_records(
    payload: dict[str, Any], candidate_store: Any, graph: ConceptGraphView, password: str, *, statuses: tuple[str, ...] = ("candidate", "needs_review")
) -> list[Any]:
    """Every pending record whose stored patient identifiers match this payload's -- not just the first, potentially several.

    Used by both callers, for different reasons: pending_case_router.py's
    merge/new_case decision wants at most one (the most recent, when
    several exist); case_confirmation.py's training-and-close path wants
    every match, since more than one pending entry could genuinely belong
    to the same patient and question (recorded separately before a stable
    case_id existed) and all of them should close together, not just one.
    Returning the full list here and letting each caller decide how many
    it needs keeps that choice where the context that justifies it lives.
    """
    incoming = PatientIdentifiers.from_payload(payload, password)
    if not (incoming.fiscal_code_hash or (incoming.name_hash and incoming.surname_hash)):
        return []
    matches = []
    for record in candidate_store.list_by_status(statuses=list(statuses)):
        stored_data = record.payload.get("patient_identifiers")
        if not stored_data:
            continue
        stored = PatientIdentifiers.from_stored(stored_data)
        if identifiers_match(incoming, stored, graph):
            matches.append(record)
    return matches


def identifiers_match(incoming: PatientIdentifiers, stored: PatientIdentifiers, graph: ConceptGraphView) -> bool:
    """Whether `incoming` identifies the same patient/case as `stored` -- fiscal code alone, or all three of the fallback together.

    Fiscal code, when present on both sides, is decisive on its own --
    more specific and far less ambiguous than two free-text fields.
    Otherwise, name hash, surname hash, exact date, and a genuinely
    overlapping diagnostic question are ALL required -- any one or two of
    these alone is not enough evidence to act on.
    """
    if incoming.fiscal_code_hash and stored.fiscal_code_hash:
        return incoming.fiscal_code_hash == stored.fiscal_code_hash

    if not (incoming.name_hash and incoming.surname_hash and incoming.case_date):
        return False
    if not (stored.name_hash and stored.surname_hash and stored.case_date):
        return False
    if incoming.name_hash != stored.name_hash or incoming.surname_hash != stored.surname_hash:
        return False
    if incoming.case_date != stored.case_date:
        return False
    return _questions_overlap(incoming.diagnostic_question or "", stored.diagnostic_question or "", graph)
