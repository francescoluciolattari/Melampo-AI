"""The reference library (4 November 2026): what a measure is compared with.

A radiologist who reads "the portal vein measures 16 mm" knows how large a portal vein usually is, and from a
book, not from a guess. The models remember values, and two models remember different ones (claims3: one wrote
"usually up to 13 mm, it differs", the other only the measure), so a measure compared from memory sends the
sentence to a person, or worse attaches an unfounded judgement. Here the comparison is made with the library:

* **the table** (``data/reference/values.yml``): each row a reference value with its source (URL, section,
  reference number), its population (age, sex) and technique, its licence, the words of the source it rests on
  and whether they were found on the page (``memory/sources_index.py``). Only the facts are in the repository;
* **the paragraphs** of the sources (a local copy, ``MELAMPO_REFERENCE_PASSAGES``): paragraphs, never whole
  chapters, each with its source, section and the sections it links to; not committed, as the texts of PARROT.

The models receive the rows (and paragraphs) that apply to the structure and the exam, compare and cite the row
they used; the reader recomputes the comparison of the cited row with the number written in the sentence
(``check``). The reader never chooses a row: when the readers do not agree, a person decides.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
VALUES = ROOT / "data" / "reference" / "values.yml"

BELOW, WITHIN, ABOVE, UNKNOWN = "below", "within", "above", "unknown"

# units the rows use, and the factor to them from what a report writes
_LENGTH = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
_VOLUME = {"ml": 1.0, "cm3": 1.0, "cc": 1.0, "cm³": 1.0, "l": 1000.0}
_NUM = r"\d+(?:[.,]\d+)?"
_MEASURE = re.compile(rf"(?<![\w.,])(?P<a>{_NUM})(?:\s*[x×]\s*(?P<b>{_NUM}))?(?:\s*[x×]\s*(?P<c>{_NUM}))?\s*"
                      r"(?P<u>mm|cm³|cm3|cm|cc|ml|l)(?P<per>\s*/\s*m2|\s*/\s*m²)?(?![\w])", re.IGNORECASE)


@dataclass(frozen=True)
class Row:
    id: str
    structures: tuple[str, ...]
    aspect: str
    detail: str
    techniques: tuple[str, ...]
    population: dict[str, str]
    low: float | None
    high: float | None
    unit: str
    above_means: str = ""
    below_means: str = ""
    source: dict[str, str] = field(default_factory=dict)
    status: str = "to_verify"
    note: str = ""

    def applies(self, cid: str, technique: str = "", sex: str = "") -> bool:
        if cid not in self.structures:
            return False
        if technique and "any" not in self.techniques and technique.upper() not in self.techniques:
            return False
        want = self.population.get("sex", "any")
        return not sex or want in ("any", sex)

    def line(self) -> str:
        """The row as the models read it."""
        rng = (f"{self.low:g}-{self.high:g}" if self.low is not None and self.high is not None
               else f"at least {self.low:g}" if self.low is not None else f"up to {self.high:g}")
        pop = ", ".join(v for k, v in self.population.items() if v and v != "any")
        means = f"; above: {self.above_means}" if self.above_means else ""
        cite = self.source.get("publisher", "")
        return (f"[{self.id}] {', '.join(self.structures)}, {self.aspect} ({self.detail}); "
                f"{'/'.join(self.techniques)}; {pop}: normal {rng} {self.unit}{means}. Source: {cite}, "
                f"\"{self.source.get('title', '')}\", {self.source.get('section', '')}.")


class ReferenceValues:
    """The table of reference values, and the paragraphs of the sources when a local copy is given."""

    def __init__(self, rows: list[Row], passages: list[dict[str, Any]] | None = None):
        self.rows = rows
        self.by_id = {r.id: r for r in rows}
        self.passages = passages or []

    @classmethod
    def load(cls, path: Path | None = None, passages: Path | None = None) -> ReferenceValues:
        import yaml

        data = yaml.safe_load(Path(path or VALUES).read_text("utf-8")) or {}
        rows = []
        for r in data.get("rows", []):
            normal = r.get("normal") or {}
            rows.append(Row(
                id=r["id"], structures=tuple(r.get("structures", [])), aspect=r.get("aspect", ""),
                detail=r.get("detail", ""), techniques=tuple(t.upper() if t != "any" else t
                                                             for t in r.get("techniques", ["any"])),
                population={k: str(v) for k, v in (r.get("population") or {}).items()},
                low=_float(normal.get("low")), high=_float(normal.get("high", r.get("above"))),
                unit=r.get("unit", "mm"), above_means=str(r.get("above_means", "")),
                below_means=str(r.get("below_means", "")), source=dict(r.get("source") or {}),
                status=r.get("status", "to_verify"), note=str(r.get("note", ""))))
        path = passages or os.environ.get("MELAMPO_REFERENCE_PASSAGES")
        found = []
        if path and Path(path).exists():
            found = [json.loads(line) for line in Path(path).read_text("utf-8").splitlines() if line.strip()]
        return cls(rows, found)

    def rows_for(self, cid: str, technique: str = "", sex: str = "") -> list[Row]:
        return [r for r in self.rows if r.applies(cid, technique, sex)]

    def passages_for(self, rows: list[Row], limit: int = 3) -> list[dict[str, Any]]:
        """The paragraphs of the rows' sources (same URL and section), then the sections they link to; never
        more than ``limit`` paragraphs, never a whole chapter."""
        out: list[dict[str, Any]] = []
        keys = {(r.source.get("url", ""), r.source.get("section", "")) for r in rows}
        for p in self.passages:
            if (p.get("url", ""), p.get("section", "")) in keys and p not in out:
                out.append(p)
        linked = {link for p in out for link in p.get("links", [])}
        out += [p for p in self.passages if p.get("id") in linked and p not in out]
        return out[:limit]


def _float(v: Any) -> float | None:
    return None if v is None or v == "" else float(v)


def measures_in(text: str) -> list[dict[str, Any]]:
    """Every measure written in the text, in the units of the rows (mm, mL, mL/m2), with the largest dimension
    of a size ("18 x 6 mm": 18)."""
    out = []
    for m in _MEASURE.finditer(text or ""):
        u = m.group("u").lower()
        dims = [float(v.replace(",", ".")) for v in (m.group("a"), m.group("b"), m.group("c")) if v]
        if u in _LENGTH and not m.group("per"):
            unit, factor = "mm", _LENGTH[u]
        elif u in _VOLUME:
            unit, factor = ("mL/m2" if m.group("per") else "mL"), _VOLUME[u]
        else:
            continue
        out.append({"text": m.group(0), "unit": unit, "value": max(dims) * factor})
    return out


def compare(value: float, row: Row) -> str:
    if row.low is not None and value < row.low:
        return BELOW
    if row.high is not None and value > row.high:
        return ABOVE
    return WITHIN


def check(words: str, row_id: str, said: str, rows: list[Row]) -> str:
    """Does the comparison a model states for the measure in ``words`` hold with the row it cites?
    "" when it holds; otherwise why not. A row not given to the model, a measure in another unit, or a sentence
    with no number are not checked by the reader: they are said so, and a person sees them."""
    by_id = {r.id: r for r in rows}
    row = by_id.get(row_id)
    if row is None:
        return "the_cited_reference_was_not_given"
    found = [m for m in measures_in(words) if m["unit"] == row.unit]
    if not found:
        return "no_measure_in_the_unit_of_the_reference"
    got = compare(found[0]["value"], row)
    if said not in (BELOW, WITHIN, ABOVE):
        return "the_comparison_is_not_stated"
    return "" if got == said else f"the_reference_says_{got}"


__all__ = ["ABOVE", "BELOW", "UNKNOWN", "WITHIN", "ReferenceValues", "Row", "check", "compare", "measures_in"]
