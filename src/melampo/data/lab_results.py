"""Laboratory rows and antibiogram rows out of a report's text, deterministically -- step 3 of the document-processing plan.

Before this module no code anywhere in the project extracted a single
numeric laboratory value: a blood test's text reached report_text as prose,
`ClinicalObservation` existed but was never populated from a document, and
`Modality.LABS` was never referenced. This reads the text a document already
produced (a digital PDF's text layer via pdftotext -layout, a typed note, or
Nemotron-Parse markdown) and returns structured rows.

**Deterministic rules, never a trained classifier.** Every accepted row is a
line the grammar below consumes completely -- analyte, value, optional unit,
optional reference range, optional printed flag, and nothing else. Anything
the grammar does not fully consume is not a row. That anchoring is what
keeps prose out: "Hb 9.8 g/dL riferita dal curante" leaves "riferita dal
curante" unconsumed and is rejected, where a looser pattern would have
invented an observation from a sentence. Explainable and testable line by
line, which is the point in an MDR context.

**What is deliberately not done here, and why:**

- No unit conversion and no LOINC coding. The unit is kept exactly as
  printed; `code` is the analyte name as printed, marked
  `code_system="unmapped_local_name"`. LOINC import is ROADMAP A3.2, still
  open -- inventing a mapping here would be the kind of guess this project
  refuses.
- No clinical interpretation beyond the laboratory's own printed range.
  "above"/"below" means outside the range printed on the same line of the
  same report, nothing more. Mapping to phenotypes (HPO) is step 4.
- No guessing on ambiguous numbers. Italian reports use "," for decimals
  and "." for thousands, but many laboratory systems print "." as the
  decimal separator. "250.000" is 250000 platelets per mm3 or 250.0 of
  something else; it is read as thousands only when the unit is a count
  per volume or the printed range uses the same thousands shape, and is
  otherwise kept but marked `number_format_ambiguous` and never assessed
  against its range.

**Measured, not hidden:** in a document recognised as a laboratory report,
every line that looks like data (letters and digits) but is not accepted is
returned in `unparsed_lines` with its line number -- residue a reviewer can
see, instead of values that silently vanished.

**Validation status, stated plainly:** the grammar was written against the
layouts common to Italian laboratory information systems (column-aligned
pdftotext output, pipe-delimited markdown tables), not against a corpus of
real reports from the laboratories this will actually receive. The first
real reports are the test that matters; `unparsed_lines` is how the gaps
will show.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..types import ClinicalObservation

DOCUMENT_TYPE_LABORATORY = "laboratory"
DOCUMENT_TYPE_PROTEIN_ELECTROPHORESIS = "protein_electrophoresis"
DOCUMENT_TYPE_ANTIBIOGRAM = "antibiogram"
DOCUMENT_TYPE_NARRATIVE = "narrative"

RANGE_WITHIN = "within"
RANGE_ABOVE = "above"
RANGE_BELOW = "below"
RANGE_NOT_ASSESSABLE = "not_assessable"
RANGE_NO_REFERENCE = "no_reference_range"
RANGE_REFERENCE_NOT_PARSED = "reference_not_parsed"
RANGE_QUALITATIVE_MATCH = "qualitative_matches_reference"
RANGE_QUALITATIVE_MISMATCH = "qualitative_differs_from_reference"

FLAG_AGREES = "agrees"
FLAG_DISAGREES = "disagrees"
FLAG_LAB_DID_NOT_FLAG = "lab_did_not_flag"
FLAG_NOT_ASSESSABLE = "not_assessable"

# At least this many numeric rows before a document counts as a laboratory
# report: one accidental parse in a narrative report must not reclassify it.
MIN_NUMERIC_ROWS_FOR_LABORATORY = 2
# Distinct globulin fractions (alfa1, alfa2, beta, gamma) that mark a protein
# electrophoresis -- albumin alone is an ordinary chemistry test.
MIN_FRACTIONS_FOR_ELECTROPHORESIS = 3

_NUMBER = r"\d+(?:[.,]\d+)*"
_NUMBER_RE = re.compile(rf"^{_NUMBER}$")
_COMPARATOR_RE = re.compile(r"^(<=|>=|≤|≥|<|>)")
_COMPARATOR_NORMAL = {"≤": "<=", "≥": ">="}

_FLAG_MEANING = {
    "*": "abnormal", "**": "abnormal",
    "H": "high", "HH": "high", "↑": "high", "ALTO": "high",
    "L": "low", "LL": "low", "↓": "low", "BASSO": "low",
}
_ATTACHED_FLAG_RE = re.compile(r"^(?P<pre>\*{1,2})?(?P<body>.+?)(?P<post>\*{1,2}|HH|LL|H|L|↑|↓)?$")

# A unit is either a token containing "/", "%" or "^" (g/dL, x10^3/µL, %,
# mmol/L) or one of these bare units. A whitelist on purpose: an open
# pattern would accept any word as a "unit" and let prose through.
_BARE_UNITS = {"fL", "fl", "pg", "sec", "s", "U", "UI", "IU", "mU", "ratio", "mmHg", "INR", "g", "mg"}
_COUNT_UNIT_MARKERS = ("/mm3", "/mm³", "/µl", "/ul", "/μl", "/ml", "cellule", "ufc")

_QUALITATIVE_STEMS = {
    "negativ": "negative", "positiv": "positive", "assent": "absent", "present": "present",
    "tracce": "traces", "normal": "normal", "nella norma": "normal", "non rilevabil": "not_detectable",
    "rilevabil": "detectable",
}

_SIR = {"S": "S", "I": "I", "R": "R", "SENSIBILE": "S", "INTERMEDIO": "I", "RESISTENTE": "R"}
_MIC_RE = re.compile(r"^(?:<=|>=|≤|≥|<|>)?\d+(?:[.,]\d+)?(?:/\d+(?:[.,]\d+)?)?$")
_ORGANISM_RE = re.compile(
    r"^\s*(?:germe(?:\s+isolato)?|microrganismo(?:\s+isolato)?|isolato|organismo)\s*[:\-]\s*(?P<name>[A-Za-z][A-Za-z .\-]+?)"
    r"(?:\s{2,}.*|\s+carica.*)?$",
    re.IGNORECASE,
)

# Column-header rows ("ESAME   RISULTATO   UNITA'   VALORI DI RIFERIMENTO")
# are neither a section nor a data row.
_COLUMN_HEADER_WORDS = {"RISULTATO", "RISULTATI", "VALORI", "RIFERIMENTO", "UNITA", "UNITA'", "UNITÀ", "ESITO", "INTERPRETAZIONE", "MIC"}

_FRACTION_PATTERNS = {
    "alfa1": re.compile(r"\b(?:alfa|alpha|α)\s*-?\s*1\b", re.IGNORECASE),
    "alfa2": re.compile(r"\b(?:alfa|alpha|α)\s*-?\s*2\b", re.IGNORECASE),
    "beta": re.compile(r"\b(?:beta|β)\b", re.IGNORECASE),
    "gamma": re.compile(r"\b(?:gamma|γ)\b", re.IGNORECASE),
}


@dataclass(frozen=True)
class ReferenceRange:
    low: float | None
    high: float | None
    low_inclusive: bool = True
    high_inclusive: bool = True
    text: str = ""
    qualitative: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "low": self.low, "high": self.high, "low_inclusive": self.low_inclusive,
            "high_inclusive": self.high_inclusive, "text": self.text, "qualitative": self.qualitative,
        }


@dataclass(frozen=True)
class LabResult:
    analyte: str
    value: float | None
    value_text: str
    comparator: str | None
    unit: str | None
    reference: ReferenceRange | None
    reference_text: str | None
    printed_flag: str | None
    range_status: str
    flag_agreement: str
    section: str | None
    line_number: int
    qualitative_value: str | None = None
    notes: tuple[str, ...] = ()

    def to_observation(self, source_prefix: str) -> ClinicalObservation:
        return ClinicalObservation(
            code=self.analyte,
            value=self.value if self.value is not None else self.value_text,
            unit=self.unit,
            source=f"{source_prefix}:line-{self.line_number}",
            reference_range=self.reference.as_dict() if self.reference else None,
            interpretation=self.range_status,
            details={
                "kind": "laboratory_result",
                "code_system": "unmapped_local_name",
                "value_text": self.value_text,
                "comparator": self.comparator,
                "qualitative_value": self.qualitative_value,
                "reference_text": self.reference_text,
                "printed_flag": self.printed_flag,
                "flag_agreement": self.flag_agreement,
                "section": self.section,
                "notes": list(self.notes),
            },
        )


@dataclass(frozen=True)
class Susceptibility:
    antibiotic: str
    interpretation: str
    mic_text: str | None
    organism: str | None
    line_number: int

    def to_observation(self, source_prefix: str) -> ClinicalObservation:
        return ClinicalObservation(
            code=self.antibiotic,
            value=self.interpretation,
            unit=None,
            source=f"{source_prefix}:line-{self.line_number}",
            interpretation=self.interpretation,
            details={
                "kind": "antimicrobial_susceptibility",
                "code_system": "unmapped_local_name",
                "organism": self.organism,
                "mic": self.mic_text,
            },
        )


@dataclass(frozen=True)
class LabExtraction:
    results: tuple[LabResult, ...] = ()
    susceptibilities: tuple[Susceptibility, ...] = ()
    organisms: tuple[str, ...] = ()
    document_types: frozenset[str] = field(default_factory=frozenset)
    unparsed_lines: tuple[tuple[int, str], ...] = ()

    def observations(self, source_prefix: str) -> list[ClinicalObservation]:
        return [row.to_observation(source_prefix) for row in self.results] + [
            row.to_observation(source_prefix) for row in self.susceptibilities
        ]

    def summary(self) -> dict[str, Any]:
        """Counts only -- no values, no line text: what travels with the case result's attachment summary."""
        out_of_range = sum(1 for row in self.results if row.range_status in (RANGE_ABOVE, RANGE_BELOW))
        return {
            "document_types": sorted(self.document_types),
            "lab_result_count": len(self.results),
            "out_of_range_count": out_of_range,
            "flag_disagreement_count": sum(1 for row in self.results if row.flag_agreement == FLAG_DISAGREES),
            "susceptibility_count": len(self.susceptibilities),
            "unparsed_line_count": len(self.unparsed_lines),
        }


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


def _is_thousands_shape(text: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}(?:\.\d{3})+", text))


def _parse_number(text: str, thousands_corroborated: bool) -> tuple[float | None, bool]:
    """(value, ambiguous). Both separators present: the last one is the decimal separator."""
    if "," in text and "." in text:
        decimal = "," if text.rfind(",") > text.rfind(".") else "."
        thousands = "." if decimal == "," else ","
        return float(text.replace(thousands, "").replace(decimal, ".")), False
    if "," in text:
        if text.count(",") > 1:
            return None, True
        return float(text.replace(",", ".")), False
    if _is_thousands_shape(text):
        if thousands_corroborated:
            return float(text.replace(".", "")), False
        return float(text), True
    if text.count(".") > 1:
        return None, True
    return float(text), False


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def _split_comparator(token: str) -> tuple[str | None, str]:
    match = _COMPARATOR_RE.match(token)
    if not match:
        return None, token
    comparator = _COMPARATOR_NORMAL.get(match.group(1), match.group(1))
    return comparator, token[match.end():]


def _is_unit(token: str) -> bool:
    if token in _BARE_UNITS:
        return True
    if not any(char in token for char in "/%^"):
        return False
    if _NUMBER_RE.match(token) or (re.match(r"^\d", token) and not re.match(r"^10[\^*Ee]", token)):
        return False
    return not re.fullmatch(rf"[<>≤≥=]*{_NUMBER}[-–÷]{_NUMBER}", token)


def _qualitative(text: str) -> str | None:
    lowered = text.strip().lower()
    if not lowered or len(lowered) > 20:
        return None
    for stem, meaning in _QUALITATIVE_STEMS.items():
        if lowered.startswith(stem):
            return meaning
    return None


def _strip_brackets(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] in "([" and text[-1] in ")]":
        return text[1:-1].strip()
    return text


def _parse_range(text: str, thousands_corroborated: bool) -> ReferenceRange | None:
    """A printed reference range, or None if the text is not one of the accepted shapes."""
    raw = text.strip()
    body = _strip_brackets(raw)
    if not body:
        return None
    interval = re.fullmatch(rf"(?:da\s+)?({_NUMBER})\s*(?:[-–÷]|a)\s*({_NUMBER})", body, re.IGNORECASE)
    if interval:
        low, low_ambiguous = _parse_number(interval.group(1), thousands_corroborated)
        high, high_ambiguous = _parse_number(interval.group(2), thousands_corroborated)
        if low is None or high is None or low_ambiguous or high_ambiguous or low > high:
            return None
        return ReferenceRange(low=low, high=high, text=raw)
    upper = re.fullmatch(rf"(<=|≤|<|fino\s+a|inferiore\s+a|inf\.\s*a)\s*({_NUMBER})", body, re.IGNORECASE)
    if upper:
        high, ambiguous = _parse_number(upper.group(2), thousands_corroborated)
        if high is None or ambiguous:
            return None
        inclusive = upper.group(1).lower() in ("<=", "≤", "fino a") or upper.group(1).lower().startswith("fino")
        return ReferenceRange(low=None, high=high, high_inclusive=inclusive, text=raw)
    lower = re.fullmatch(rf"(>=|≥|>|superiore\s+a|sup\.\s*a)\s*({_NUMBER})", body, re.IGNORECASE)
    if lower:
        low, ambiguous = _parse_number(lower.group(2), thousands_corroborated)
        if low is None or ambiguous:
            return None
        return ReferenceRange(low=low, high=None, low_inclusive=lower.group(1) in (">=", "≥"), text=raw)
    qualitative = _qualitative(body)
    if qualitative:
        return ReferenceRange(low=None, high=None, text=raw, qualitative=qualitative)
    return None


# ---------------------------------------------------------------------------
# Assessment against the printed range
# ---------------------------------------------------------------------------


def _possible_interval(value: float, comparator: str | None) -> tuple[float, float, bool, bool]:
    """The values a printed result can stand for: "<0.5" is anything below 0.5."""
    infinity = float("inf")
    if comparator == "<":
        return -infinity, value, False, False
    if comparator == "<=":
        return -infinity, value, False, True
    if comparator == ">":
        return value, infinity, False, False
    if comparator == ">=":
        return value, infinity, True, False
    return value, value, True, True


def _assess(value: float, comparator: str | None, reference: ReferenceRange) -> str:
    low_v, high_v, low_v_inc, high_v_inc = _possible_interval(value, comparator)

    # A possible-interval end that sits exactly on a bound violates it only
    # when that end is itself a possible value and the bound excludes it:
    # ">0.5" against ">=0.5" never reaches 0.5, so it is inside.
    def below_low(x: float, inclusive_x: bool) -> bool:
        if reference.low is None:
            return False
        return x < reference.low or (x == reference.low and inclusive_x and not reference.low_inclusive)

    def above_high(x: float, inclusive_x: bool) -> bool:
        if reference.high is None:
            return False
        return x > reference.high or (x == reference.high and inclusive_x and not reference.high_inclusive)

    # Entirely above / below: every value the result can stand for is outside.
    if reference.high is not None and (low_v > reference.high or (low_v == reference.high and (not reference.high_inclusive or not low_v_inc))):
        return RANGE_ABOVE
    if reference.low is not None and (high_v < reference.low or (high_v == reference.low and (not reference.low_inclusive or not high_v_inc))):
        return RANGE_BELOW
    # Entirely within: both ends of the possible interval are inside.
    if not below_low(low_v, low_v_inc) and not above_high(high_v, high_v_inc):
        return RANGE_WITHIN
    return RANGE_NOT_ASSESSABLE


def _flag_agreement(printed_flag: str | None, range_status: str) -> str:
    if range_status not in (RANGE_WITHIN, RANGE_ABOVE, RANGE_BELOW):
        return FLAG_NOT_ASSESSABLE
    meaning = _FLAG_MEANING.get(printed_flag) if printed_flag else None
    if meaning is None:
        return FLAG_LAB_DID_NOT_FLAG if range_status != RANGE_WITHIN else FLAG_AGREES
    if meaning == "abnormal":
        return FLAG_AGREES if range_status != RANGE_WITHIN else FLAG_DISAGREES
    if meaning == "high":
        return FLAG_AGREES if range_status == RANGE_ABOVE else FLAG_DISAGREES
    return FLAG_AGREES if range_status == RANGE_BELOW else FLAG_DISAGREES


# ---------------------------------------------------------------------------
# Row parsing
# ---------------------------------------------------------------------------


@dataclass
class _Tail:
    comparator: str | None = None
    number_text: str | None = None
    qualitative_text: str | None = None
    unit: str | None = None
    range_text: str | None = None
    reference_text: str | None = None
    flag: str | None = None


def _take_flag(tail: _Tail, token: str) -> bool:
    if token in _FLAG_MEANING and tail.flag is None:
        tail.flag = token
        return True
    return False


def _parse_value_token(token: str, tail: _Tail, allow_qualitative: bool) -> bool:
    match = _ATTACHED_FLAG_RE.match(token)
    body = token
    attached_flag = None
    if match:
        body = match.group("body")
        attached_flag = match.group("pre") or match.group("post")
    comparator, number = _split_comparator(body)
    if _NUMBER_RE.match(number):
        tail.comparator, tail.number_text = comparator, number
        if attached_flag:
            tail.flag = attached_flag
        return True
    if allow_qualitative and _qualitative(token):
        tail.qualitative_text = token
        return True
    return False


def _parse_tail_tokens(tokens: list[str], allow_qualitative: bool) -> _Tail | None:
    """VALUE [UNIT] [RANGE] with a printed flag allowed around any of them -- and nothing left over."""
    tail = _Tail()
    i = 0
    if i < len(tokens) and _take_flag(tail, tokens[i]):
        i += 1
    # A comparator printed as its own token: "< 0.5".
    if i + 1 < len(tokens) and tokens[i] in ("<", ">", "<=", ">=", "≤", "≥") and _NUMBER_RE.match(tokens[i + 1]):
        tail.comparator = _COMPARATOR_NORMAL.get(tokens[i], tokens[i])
        tail.number_text = tokens[i + 1]
        i += 2
    elif i < len(tokens) and _parse_value_token(tokens[i], tail, allow_qualitative):
        i += 1
    else:
        return None
    if i < len(tokens) and _take_flag(tail, tokens[i]):
        i += 1
    # Unit, possibly "x 10^3/µL" as two tokens.
    if i + 1 < len(tokens) and tokens[i] == "x" and _is_unit(tokens[i + 1]):
        tail.unit = f"x{tokens[i + 1]}"
        i += 2
    elif i < len(tokens) and _is_unit(tokens[i]):
        tail.unit = tokens[i]
        i += 1
    if i < len(tokens) and _take_flag(tail, tokens[i]):
        i += 1
    # Range: the longest run of remaining tokens that parses as one,
    # optionally followed by a flag.
    remaining = tokens[i:]
    if remaining:
        trailing_flag = remaining[-1] if remaining[-1] in _FLAG_MEANING and tail.flag is None else None
        range_tokens = remaining[:-1] if trailing_flag else remaining
        if not range_tokens:
            tail.flag = trailing_flag
            return tail
        candidate = " ".join(range_tokens)
        if _parse_range(candidate, thousands_corroborated=True) is None:
            return None
        tail.range_text = candidate
        if trailing_flag:
            tail.flag = trailing_flag
    return tail


def _count_unit(unit: str | None) -> bool:
    return bool(unit) and any(marker in unit.lower() for marker in _COUNT_UNIT_MARKERS)


def _valid_analyte(text: str) -> bool:
    letters = sum(1 for char in text if char.isalpha())
    if letters < 2 or text.strip() in _FLAG_MEANING:
        return False
    # An analyte never ends with a result word: otherwise "Nitriti positivo
    # negativo" also reads as analyte "Nitriti positivo" = negativo, and the
    # real row is lost as ambiguous.
    words = text.split()
    if words and _qualitative(words[-1]):
        return False
    return bool(re.match(r"^[\wÀ-ÿµα-ωΑ-Ω(]", text.strip()))


def _build_result(analyte: str, tail: _Tail, section: str | None, line_number: int) -> LabResult:
    notes: list[str] = []
    analyte = re.sub(r"\s+", " ", analyte).strip().rstrip(":").strip()
    if tail.qualitative_text is not None:
        reference = _parse_range(tail.range_text, True) if tail.range_text else None
        qualitative_value = _qualitative(tail.qualitative_text)
        if reference is not None and reference.qualitative:
            status = RANGE_QUALITATIVE_MATCH if reference.qualitative == qualitative_value else RANGE_QUALITATIVE_MISMATCH
        else:
            status = RANGE_NO_REFERENCE if reference is None else RANGE_NOT_ASSESSABLE
        return LabResult(
            analyte=analyte, value=None, value_text=tail.qualitative_text, comparator=None, unit=tail.unit,
            reference=reference, reference_text=tail.reference_text, printed_flag=tail.flag, range_status=status,
            flag_agreement=FLAG_NOT_ASSESSABLE, section=section, line_number=line_number,
            qualitative_value=qualitative_value,
        )

    assert tail.number_text is not None
    range_thousands = bool(tail.range_text) and any(
        _is_thousands_shape(part) for part in re.findall(_NUMBER, tail.range_text or "")
    )
    corroborated = _count_unit(tail.unit) or range_thousands
    value, ambiguous = _parse_number(tail.number_text, corroborated)
    reference = _parse_range(tail.range_text, corroborated) if tail.range_text else None
    value_text = f"{tail.comparator or ''}{tail.number_text}"

    if ambiguous:
        notes.append("number_format_ambiguous")
        status = RANGE_NOT_ASSESSABLE
    elif tail.reference_text is not None:
        status = RANGE_REFERENCE_NOT_PARSED
    elif reference is None or reference.qualitative:
        status = RANGE_NO_REFERENCE
    else:
        status = _assess(value, tail.comparator, reference) if value is not None else RANGE_NOT_ASSESSABLE
    return LabResult(
        analyte=analyte, value=None if ambiguous else value, value_text=value_text, comparator=tail.comparator,
        unit=tail.unit, reference=reference, reference_text=tail.reference_text, printed_flag=tail.flag,
        range_status=status, flag_agreement=_flag_agreement(tail.flag, status), section=section,
        line_number=line_number, notes=tuple(notes),
    )


def _cells(line: str) -> list[str] | None:
    """Explicit columns: a markdown table row (Nemotron-Parse output) or 2+ spaces (pdftotext -layout)."""
    if "|" in line:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", cell) or not cell for cell in cells):
            return []  # markdown separator row
        return [cell for cell in cells if cell]
    cells = [cell for cell in re.split(r"\s{2,}|\t", line.strip()) if cell]
    return cells if len(cells) >= 2 else None


def _parse_row(line: str, section: str | None, line_number: int, allow_qualitative: bool) -> tuple[LabResult | None, bool]:
    """(result, ambiguous). Columns first; otherwise every analyte/value split point, accepted only if exactly one fits."""
    cells = _cells(line)
    if cells:
        analyte, rest = cells[0], cells[1:]
        # A flag printed right after the analyte name: "Creatinina *".
        head = analyte.split()
        if head and head[-1] in _FLAG_MEANING and len(head) > 1:
            analyte, rest = " ".join(head[:-1]), [head[-1], *rest]
        if _valid_analyte(analyte):
            tail = _parse_tail_tokens(" ".join(rest).split(), allow_qualitative)
            if tail is None and len(rest) >= 2:
                # Columns give one more safe option: value (and unit) in their
                # own columns, then reference columns that are not one of the
                # accepted range shapes ("M: 30-400  F: 13-150") -- kept as raw
                # text, never assessed. Longest value+unit prefix first.
                for cut in (2, 1):
                    if cut >= len(rest):
                        continue
                    candidate = _parse_tail_tokens(" ".join(rest[:cut]).split(), allow_qualitative)
                    if candidate is not None and candidate.range_text is None and (candidate.unit or cut == 1):
                        candidate.reference_text = "  ".join(rest[cut:])
                        tail = candidate
                        break
            if tail is not None:
                return _build_result(analyte, tail, section, line_number), False

    tokens = line.split()
    fits = []
    for split in range(1, len(tokens)):
        analyte = " ".join(tokens[:split])
        if not _valid_analyte(analyte):
            continue
        tail = _parse_tail_tokens(tokens[split:], allow_qualitative)
        # A bare trailing number is not a fit: without it "Vitamina B12 150
        # pg/mL 200 - 900" would also read as analyte "... 200 -" = 900, and
        # the real reading would be discarded as ambiguous.
        if tail is not None and (tail.unit or tail.range_text or tail.qualitative_text):
            fits.append((analyte, tail))
    if len(fits) == 1:
        analyte, tail = fits[0]
        return _build_result(analyte, tail, section, line_number), False
    return None, len(fits) > 1


def _parse_susceptibility(line: str, organism: str | None, line_number: int) -> Susceptibility | None:
    cells = _cells(line)
    tokens = (" ".join(cells).split() if cells else line.split())
    if len(tokens) < 2:
        return None
    sir_positions = [i for i, token in enumerate(tokens) if token.upper() in _SIR]
    if not sir_positions:
        return None
    position = sir_positions[-1]
    interpretation = _SIR[tokens[position].upper()]
    others = tokens[:position] + tokens[position + 1:]
    mic = None
    if others:
        # A MIC may be a ratio for combinations: trimethoprim/sulfamethoxazole ">4/76".
        if _MIC_RE.match(others[-1]):
            mic = others.pop()
        elif len(others) >= 2 and others[-2] in ("<", ">", "<=", ">=", "≤", "≥") and _MIC_RE.match(others[-1]):
            mic = f"{others[-2]}{others[-1]}"
            others = others[:-2]
    # Whatever order MIC and interpretation were printed in ("... >32  R" or
    # "... R  >32"), only the antibiotic's name may remain.
    name = " ".join(others).strip()
    if not _valid_analyte(name) or any(_NUMBER_RE.match(token) for token in name.split()):
        return None
    return Susceptibility(antibiotic=name, interpretation=interpretation, mic_text=mic, organism=organism, line_number=line_number)


def _section_header(line: str) -> str | None:
    text = line.strip().strip("#").strip().strip("*").strip()
    if not text or len(text) > 60 or any(char.isdigit() for char in text):
        return None
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 4 or sum(1 for char in letters if char.isupper()) / len(letters) < 0.8:
        return None
    words = {word.strip(".:'") for word in text.upper().split()}
    if words & _COLUMN_HEADER_WORDS:
        return None
    return re.sub(r"\s+", " ", text)


def _is_column_header(line: str) -> bool:
    """"Antibiotico   MIC   Interpretazione", "ESAME   RISULTATO   UNITA'" -- a table's own header row."""
    words = {word.strip(".:'").upper() for word in re.split(r"[\s|]+", line) if word}
    return len(words & _COLUMN_HEADER_WORDS) >= 1 and not re.search(r"\d", line)


def _looks_like_data(line: str) -> bool:
    """Digits and letters ("Glicemia seriati: 95 110 102"), or laid out in columns ("Colore   giallo paglierino")."""
    has_letters = sum(1 for char in line if char.isalpha()) >= 2
    if not has_letters:
        return False
    # Page furniture and lines already redacted as identifying (patient,
    # dates) are administrative by construction, never a missed result.
    if "[REDACTED" in line or re.search(r"\bpagina\s+\d+\s+di\s+\d+\b", line, re.IGNORECASE):
        return False
    return bool(re.search(r"\d", line)) or bool(_cells(line))


def extract_lab_results(text: str) -> LabExtraction:
    """Every laboratory and antibiogram row a text contains, plus which kinds of report it is."""
    lines = text.splitlines()
    section: str | None = None
    organism: str | None = None
    in_antibiogram = False
    numeric: list[LabResult] = []
    qualitative: list[LabResult] = []
    susceptibilities: list[Susceptibility] = []
    organisms: list[str] = []
    candidates_for_residue: list[tuple[int, str]] = []

    seen_section = False
    for number, raw in enumerate(lines, start=1):
        line = raw.rstrip()
        if not line.strip():
            continue
        header = _section_header(line)
        if header is not None:
            section = header
            seen_section = True
            in_antibiogram = "ANTIBIOGRAMMA" in header.upper()
            continue
        if _cells(line) == []:
            continue  # a markdown table's separator row
        organism_match = _ORGANISM_RE.match(line)
        if organism_match:
            organism = organism_match.group("name").strip()
            organisms.append(organism)
            in_antibiogram = True
            continue
        if in_antibiogram:
            susceptibility = _parse_susceptibility(line, organism, number)
            if susceptibility is not None:
                susceptibilities.append(susceptibility)
                continue
        if _is_column_header(line):
            continue
        result, ambiguous = _parse_row(line, section, number, allow_qualitative=True)
        if ambiguous:
            # More than one reading fits: never pick one, always surface it.
            candidates_for_residue.append((number, line.strip()))
            continue
        # A numeric row needs a unit or a reference to count at all: "Pagina 1
        # di 2" and "Glicemia seriati: 95 110 102 98" both parse as
        # analyte+number and are neither -- they fall through to the residue
        # below instead of being accepted or silently dropped.
        if result is not None and (
            result.qualitative_value is not None or result.unit or result.reference or result.reference_text
        ):
            (qualitative if result.qualitative_value is not None else numeric).append(result)
            continue
        # Residue from the first section header onwards (or everywhere when
        # the document has none): the administrative block above the first
        # section -- request number, page count -- is not clinical data.
        if _looks_like_data(line) and (seen_section or not any(_section_header(item) for item in lines)):
            candidates_for_residue.append((number, line.strip()))

    numeric_rows = numeric
    types: set[str] = set()
    is_laboratory = len(numeric_rows) >= MIN_NUMERIC_ROWS_FOR_LABORATORY
    if is_laboratory:
        types.add(DOCUMENT_TYPE_LABORATORY)
    fractions = {name for row in numeric for name, pattern in _FRACTION_PATTERNS.items() if pattern.search(row.analyte)}
    if len(fractions) >= MIN_FRACTIONS_FOR_ELECTROPHORESIS:
        types.add(DOCUMENT_TYPE_PROTEIN_ELECTROPHORESIS)
    # Susceptibility rows are only read inside an antibiogram context (its
    # header, or an isolated-organism line), so one row is already enough.
    if susceptibilities:
        types.add(DOCUMENT_TYPE_ANTIBIOGRAM)
    if not types:
        types.add(DOCUMENT_TYPE_NARRATIVE)

    # The row-count threshold decides the document's TYPE, never whether a
    # fully-consumed numeric row with a unit or range is data: a discharge
    # letter's single "Emoglobina   9,8   g/dL   12,0 - 16,0" line is kept.
    # Qualitative rows ("Nitriti  negativo") and the residue only inside a
    # laboratory report: in a narrative one "Linfonodi assenti" is a
    # sentence, and every prose line with a number would be "residue".
    if is_laboratory:
        results = sorted(numeric_rows + qualitative, key=lambda row: row.line_number)
        residue = tuple(candidates_for_residue)
    else:
        results = list(numeric_rows)
        residue = ()
    return LabExtraction(
        results=tuple(results),
        susceptibilities=tuple(susceptibilities),
        organisms=tuple(dict.fromkeys(organisms)),
        document_types=frozenset(types),
        unparsed_lines=residue,
    )
