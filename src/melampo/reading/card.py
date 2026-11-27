"""The one card of a finding, for every text Melampo reads (25 November 2026).

A patient's report and a medical publication speak the same language: what is observed, where, by which method,
with which interpretation. The name is a label. One card holds that description, and one validator reads it, for
every model and every text; only what the text *states about it* changes with the context:

- ``case`` (a report, a note, a letter of this patient): the state in this patient -- present, absent, uncertain;
  when (current, past, hypothetical); whose (the patient, a family member);
- ``publication`` (an article, a guideline, a book): the measures in a population -- frequency, accuracy, the
  direction without a number, and what the measures compare.

The card carries no code: a code (HPO, a concept of the thesaurus) is given afterwards, to the reading both
interpreters agree on (``reading.normalize``). Coding before comparing makes disagreements the text does not have:
two independent coders give the same SNOMED CT code to three diagnoses in four (Del-Pinto et al., 2023), and
codes that differ often mean the same thing.

What comes in another form is not read, and is counted for the model that sent it: the models are interchangeable,
with one directive (the owner's rule, 24 November 2026).
"""

from __future__ import annotations

import json
from typing import Any

CONTEXTS = ("case", "publication")
CATEGORIES = ("manifestation", "test", "score", "outcome", "risk_factor", "form")
OBSERVED_BY = ("history", "examination", "laboratory", "imaging", "endoscopy", "intraoperative", "histopathology",
               "course")
NATURES = ("pathological", "physiological")
# what the observation says of the thing (as SNOMED CT's "has interpretation"): "present" when the thing itself is
# the finding (fever, a mass); "increased"/"decreased" for a quantity; "normal"/"abnormal"; "positive"/"negative"
# for a sign or a test
INTERPRETATIONS = ("present", "increased", "decreased", "normal", "abnormal", "positive", "negative")
SIDES = ("left", "right", "both")
# the part of the document the sentence is in (27 November 2026), read by the interpreters on the whole document
# as a radiologist reads it: for a patient's document the sections of a report (the LOINC report sections, as
# memory/report_sections.py: the question asked, history, comparison, technique, findings, conclusion), for a
# publication its parts; "other" for anything else (a header, a table, a letter without parts)
PARTS = ("indication", "history", "comparison", "technique", "findings", "impression", "background", "methods",
         "results", "discussion", "other")
COMPARATORS = (">", ">=", "<", "<=", "=")
# the case
STATES = ("present", "absent", "uncertain")
TEMPORALITIES = ("current", "past", "hypothetical")
SUBJECTS = ("patient", "family_member")
# the publication
DIRECTIONS = ("more_likely", "less_likely")
COMPARES = ("disease_vs_without", "forms_of_the_disease", "other")
MEASURE_FIELDS = ("with_disease", "sensitivity", "specificity", "lr_positive", "lr_negative", "comparator", "setting",
                  "patients", "design", "pooled", "verbal")

DESCRIPTION = ("words", "name", "category", "specific", "what", "interpretation", "value", "threshold", "site",
               "side", "observed_by", "time", "nature", "part")
STATEMENT = {"case": ("state", "temporality", "subject"), "publication": ("direction", "compares", "measures")}
LISTED = {"category": CATEGORIES, "observed_by": OBSERVED_BY, "nature": NATURES, "interpretation": INTERPRETATIONS,
          "side": SIDES, "part": PARTS, "state": STATES, "temporality": TEMPORALITIES, "subject": SUBJECTS, "compares": COMPARES}

_DESCRIBED = {
    "words": "fever (temperature 38.6 °C)", "name": "fever", "category": "manifestation", "specific": True,
    "what": "body temperature", "interpretation": "increased", "value": {"number": 38.6, "unit": "°C"},
    "threshold": None, "site": "", "side": "", "observed_by": "examination", "time": "", "nature": "pathological",
    "part": "history"}
EXAMPLE = {
    "case": {**_DESCRIBED, "state": "present", "temporality": "current", "subject": "patient",
             "quote": "Febbre a 38,6 °C dalla notte."},
    "publication": {**_DESCRIBED, "part": "results", "words": "fever (temperature above 38 °C) in 30 of 100 patients",
                    "value": None, "threshold": {"comparator": ">", "number": 38, "unit": "°C"},
                    "direction": None, "compares": "disease_vs_without",
                    "measures": {"with_disease": {"count": 30, "total": 100, "percent": None}, "sensitivity": None,
                                 "specificity": None, "lr_positive": None, "lr_negative": None, "comparator": None,
                                 "setting": "emergency department", "patients": 100, "design": "prospective_cohort",
                                 "pooled": False, "verbal": None},
                    "quote": "Fever above 38 °C was present in 30 of 100 patients."},
}


def fields(context: str) -> tuple[str, ...]:
    return DESCRIPTION + STATEMENT[context] + ("quote",)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _one_of(value: Any, allowed: tuple[str, ...]) -> str:
    v = _text(value).lower().replace(" ", "_")
    return v if v in allowed else ""


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


def _quantity(value: Any, comparator: bool = False) -> dict | None:
    if not isinstance(value, dict):
        return None
    n = _number(value.get("number"))
    if n is None:
        return None
    out = {"number": n, "unit": _text(value.get("unit"))}
    if comparator:
        c = _text(value.get("comparator"))
        out["comparator"] = c if c in COMPARATORS else ""
    return out


def direction_of(value: Any) -> dict | None:
    """A statement of direction without a number: when present or absent, more or less likely, its words."""
    if not isinstance(value, dict):
        return None
    when = _one_of(value.get("when"), ("present", "absent"))
    effect = _one_of(value.get("effect"), DIRECTIONS)
    if not when or not effect:
        return None
    return {"when": when, "effect": effect, "words": _text(value.get("words")).lower()}


def problems(raw: dict, context: str) -> list[str]:
    """How an object departs from the one form: a field missing or of another context, a value not in its list,
    measures outside ``measures``."""
    out = [f"missing:{k}" for k in fields(context) if k not in raw]
    other = next(c for c in CONTEXTS if c != context)
    out += [f"field_of_the_{other}:{k}" for k in STATEMENT[other] if k in raw and raw[k] not in (None, "", "null")]
    if any(raw.get(k) not in (None, "", "null") for k in MEASURE_FIELDS):
        out.append("measures_outside")
    if raw.get("measures") not in (None, "null") and not isinstance(raw.get("measures"), dict):
        out.append("measures_not_an_object")
    for k, allowed in LISTED.items():
        v = _text(raw.get(k)).lower().replace(" ", "_")
        if v and v not in allowed:
            out.append(f"value_not_in_list:{k}")
    return out


def read(raw: dict, context: str) -> dict:
    """The card of one object, its values checked against their lists (a value not in its list is empty: a
    doubt for the comparison, never a value of the code's)."""
    specific = raw.get("specific")
    card = {
        "words": _text(raw.get("words")), "name": _text(raw.get("name")),
        "category": _one_of(raw.get("category"), CATEGORIES),
        # None when not given (26 November 2026): a field not given is not a field read otherwise
        "specific": specific if isinstance(specific, bool) else (
            None if specific in (None, "") else _text(specific).lower() == "true"),
        "what": _text(raw.get("what")), "interpretation": _one_of(raw.get("interpretation"), INTERPRETATIONS),
        "value": _quantity(raw.get("value")), "threshold": _quantity(raw.get("threshold"), comparator=True),
        "site": _text(raw.get("site")), "side": _one_of(raw.get("side"), SIDES),
        "observed_by": _one_of(raw.get("observed_by"), OBSERVED_BY), "time": _text(raw.get("time")),
        "nature": _one_of(raw.get("nature"), NATURES), "part": _one_of(raw.get("part"), PARTS), "quote": _text(raw.get("quote")),
    }
    if context == "case":
        card.update(state=_one_of(raw.get("state"), STATES), temporality=_one_of(raw.get("temporality"), TEMPORALITIES),
                    subject=_one_of(raw.get("subject"), SUBJECTS))
    else:
        nested = raw.get("measures")
        card.update(direction=direction_of(raw.get("direction")), compares=_one_of(raw.get("compares"), COMPARES),
                    measures={k: nested[k] for k in MEASURE_FIELDS if k in nested} if isinstance(nested, dict)
                    else {})
    return card


def example(context: str) -> str:
    return json.dumps(EXAMPLE[context], ensure_ascii=False)


__all__ = ["CATEGORIES", "COMPARES", "CONTEXTS", "DESCRIPTION", "DIRECTIONS", "EXAMPLE", "INTERPRETATIONS",
           "MEASURE_FIELDS", "NATURES", "OBSERVED_BY", "PARTS", "SIDES", "STATEMENT", "STATES", "SUBJECTS", "TEMPORALITIES",
           "direction_of", "example", "fields", "problems", "read"]
