"""Laboratory rows to LOINC codes, deterministically, from the LOINC table and its Italian variant (16 October 2026).

A row of a report ("Emoglobina  11,8  g/dL  12,0 - 16,0") names an analyte as the laboratory prints it. LOINC
names an observation by six parts: component, property, time, system (specimen), scale, method. The printed row
gives two of them directly and a third by context:

* **component**: the analyte name, matched against the LOINC component in English and in the Italian
  linguistic variant (both normalised: case, accents, punctuation; the abbreviations LOINC lists as synonyms);
* **property**: from the unit (g/dL -> mass concentration MCnc, mmol/L -> substance concentration SCnc,
  10^3/uL -> number concentration NCnc, % -> a fraction or a ratio, U/L -> catalytic concentration CCnc, ...);
* **system**: from the section or the analyte words (urine, siero/plasma, sangue intero), else left open;
* time is a point in time (Pt) and scale quantitative (Qn) for a numeric row, ordinal/nominal for a
  qualitative one.

A row gets a code only when **exactly one** active LOINC term fits every part known; several fits return the
candidates for a person, none returns nothing. Never a guess: "the closest name" is not a mapping.

The LOINC table is not in the repository: it is downloaded from loinc.org (free account, accept the LOINC licence)
and indexed with ``scripts/build_loinc_index.py``; the index path is ``data/loinc/index.json`` (ignored by git
until the licence's redistribution terms are checked). This material contains content from LOINC
(http://loinc.org), copyright Regenstrief Institute, Inc. and the LOINC Committee, available at no cost under the
license at http://loinc.org/license.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

INDEX_PATH = Path(__file__).resolve().parents[3] / "data" / "loinc" / "index.json"

# unit (normalised) -> LOINC properties it can carry
UNIT_PROPERTIES = {
    "g/dl": ("MCnc",), "g/l": ("MCnc",), "mg/dl": ("MCnc",), "mg/l": ("MCnc",), "ug/dl": ("MCnc",),
    "ug/l": ("MCnc",), "ng/ml": ("MCnc",), "ng/dl": ("MCnc",), "pg/ml": ("MCnc",), "ng/l": ("MCnc",),
    "mmol/l": ("SCnc",), "umol/l": ("SCnc",), "nmol/l": ("SCnc",), "pmol/l": ("SCnc",), "meq/l": ("SCnc",),
    "u/l": ("CCnc",), "ui/l": ("CCnc",), "iu/l": ("CCnc",), "mu/l": ("ACnc", "CCnc"), "ui/ml": ("ACnc",),
    "u/ml": ("ACnc",), "mui/ml": ("ACnc",), "uui/ml": ("ACnc",),
    "10^3/ul": ("NCnc",), "10^6/ul": ("NCnc",), "10^9/l": ("NCnc",), "10^12/l": ("NCnc",), "/mm3": ("NCnc",),
    "/ul": ("NCnc",), "x10^3/ul": ("NCnc",), "x10^6/ul": ("NCnc",), "/mmc": ("NCnc",),
    "%": ("NFr", "MFr", "VFr", "SFr", "Fr"), "fl": ("EntVol",), "pg": ("EntMass",), "sec": ("Time",), "s": ("Time",),
    "mm/h": ("Vel",), "mosm/kg": ("Osmol",), "ratio": ("Rto", "MRto", "SRto", "Ratio"), "inr": ("INR", "RelTime"),
}
SYSTEM_WORDS = (
    (re.compile(r"\burin\w*\b", re.I), ("Urine",)),
    (re.compile(r"\b(?:siero|sierico|serum|plasma|plasmatic\w*|ematochimic\w*)\b", re.I), ("Ser/Plas", "Ser", "Plas")),
    (re.compile(r"\b(?:emocromo|sangue|blood|emogas\w*)\b", re.I), ("Bld", "BldA", "BldV", "BldC")),
    (re.compile(r"\b(?:liquor|csf|cefalorachid\w*)\b", re.I), ("CSF",)),
)


def norm(text: str) -> str:
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"[^a-z0-9%^/+]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def unit_properties(unit: str | None) -> tuple[str, ...]:
    if not unit:
        return ()
    u = norm(unit).replace(" ", "").replace("µ", "u").replace("μ", "u")
    return UNIT_PROPERTIES.get(u, ())


@dataclass
class LoincTerm:
    code: str
    component: str
    property: str
    time: str
    system: str
    scale: str
    method: str = ""
    names: tuple[str, ...] = ()        # normalised English and Italian names of the component and synonyms
    long_name: str = ""


@dataclass
class LoincMatch:
    code: str | None
    status: str                       # "mapped", "ambiguous", "unmapped", "no_index"
    candidates: list[dict[str, Any]] = field(default_factory=list)
    why: str = ""

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class LoincIndex:
    def __init__(self, terms: list[LoincTerm]):
        self.terms = terms
        self.by_name: dict[str, list[LoincTerm]] = {}
        for t in terms:
            for n in t.names:
                self.by_name.setdefault(n, []).append(t)

    @classmethod
    def load(cls, path: Path = INDEX_PATH) -> LoincIndex | None:
        if not Path(path).exists():
            return None
        data = json.loads(Path(path).read_text("utf-8"))
        return cls([LoincTerm(**{**t, "names": tuple(t.get("names", ()))}) for t in data["terms"]])

    def match(self, analyte: str, unit: str | None, section: str | None = None, numeric: bool = True) -> LoincMatch:
        name = norm(re.sub(r"\(.*?\)", " ", analyte))
        found = list(self.by_name.get(name, []))
        if not found:
            return LoincMatch(None, "unmapped", why="component_not_found")
        scale = ("Qn",) if numeric else ("Ord", "Nom", "OrdQn")
        found = [t for t in found if t.scale in scale and t.time == "Pt"]
        props = unit_properties(unit)
        if props:
            found = [t for t in found if t.property in props]
        elif numeric and unit:
            return LoincMatch(None, "unmapped", why=f"unit_not_known:{unit}")
        context = " ".join(x for x in (section or "", analyte) if x)
        for rx, systems in SYSTEM_WORDS:
            if rx.search(context):
                narrowed = [t for t in found if t.system in systems]
                if narrowed:
                    found = narrowed
                break
        # a method-less term is the general one: prefer it when it is the only one without a method
        if len(found) > 1:
            plain = [t for t in found if not t.method]
            if len(plain) == 1:
                found = plain
        cands = [{"code": t.code, "long_name": t.long_name, "system": t.system, "property": t.property}
                 for t in found]
        if len(found) == 1:
            return LoincMatch(found[0].code, "mapped", cands)
        if not found:
            return LoincMatch(None, "unmapped", why="no_term_with_this_unit_and_specimen")
        return LoincMatch(None, "ambiguous", cands, why=f"{len(found)}_terms_fit")


def map_rows(rows, index: LoincIndex | None) -> list[dict[str, Any]]:
    """The LOINC mapping of each laboratory row (``lab_results.LabResult``), as JSON-safe dicts."""
    out = []
    for r in rows:
        if index is None:
            m = LoincMatch(None, "no_index", why="LOINC index not installed (scripts/build_loinc_index.py)")
        else:
            m = index.match(r.analyte, r.unit, r.section, numeric=r.value is not None)
        out.append({"analyte": r.analyte, "unit": r.unit, "line": r.line_number, **m.as_dict()})
    return out


__all__ = ["INDEX_PATH", "LoincIndex", "LoincMatch", "LoincTerm", "map_rows", "norm", "unit_properties"]
