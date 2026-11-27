"""A confirmed sentence without the person (27 November 2026, the owner's rule: once a report is confirmed, its
personal data are not needed -- it is de-identified, and used without them).

The sentence is kept as it was written, never reworded (a rewording may change what it says, and teach an error);
only what identifies the person is replaced, each with a placeholder of its kind:

* the patient's own names (from the case), and any name after a label ("Paziente:", "Sig.ra", "Dott.") -> ``[NOME]``;
* fiscal codes -> ``[CF]``; record, admission, accession and similar numbers after their label -> ``[ID]``;
* telephone numbers and e-mail addresses -> ``[CONTATTO]``; street addresses -> ``[INDIRIZZO]``;
* places of birth or residence after their words ("nato a", "residente in") -> ``[LUOGO]``;
* dates -> the same date moved by one fixed number of days per case (``shift_days``), so that the intervals
  between the exams of the person stay exact; a year alone is kept;
* an age above 89 -> ``[ETÀ>89]`` (a rare age identifies).

What it cannot see (a rare disease in a small place) is why a de-identified sentence still enters training only
after the two interpreters read it to the same card as the confirmed one (``review_feedback``), and why the
adapter is tested for memorisation before use (``docs/COMPLIANCE_RISK.md``, the plan of predetermined changes).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass, field

from .patient_check import _CF, _plain

_LABELLED_NAME = re.compile(
    r"(?P<label>(?i:paziente|patient|cognome\s+e\s+nome|nome\s+e\s+cognome|nominativo|sig\.(?:ra|na)?|signor[ae]?|"
    r"dott\.(?:ssa)?|dr\.(?:ssa)?|prof\.(?:ssa)?)\s*:?[ \t]+)"
    r"(?P<name>[A-ZÀ-Ý][A-Za-zÀ-ÿ']+(?:[ \t]+[A-ZÀ-Ý][A-Za-zÀ-ÿ']+){0,3})")
_ID = re.compile(r"(?P<label>(?i:n\.?\s*(?:di\s+)?(?:cartella|ricovero|nosologico|accettazione|referto|prenotazione|"
                 r"pratica)|cartella\s+clinica|codice\s+paziente|id\s+paziente|patient\s+id|mrn|accession(?:\s+number)?)"
                 r"\s*[:.#nN°]*\s*)(?P<id>[A-Za-z0-9/\-]{3,})")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?<![\w/.,])(?:\+\d{2}[ \t]?)?(?:\d[ \t.\-]?){8,11}\d(?![\w/]|[.,]\d)")
_ADDRESS = re.compile(r"\b(?i:via|viale|piazza|piazzale|corso|largo|vicolo|strada|località)\s+[A-ZÀ-Ý][^\n,;]{1,40}?"
                      r"(?:,?\s*\d+[A-Za-z]?)?(?=[,;\n.]|$)")
_PLACE = re.compile(r"(?P<label>(?i:nat[oa]\s+a|residente\s+(?:a|in)|domiciliat[oa]\s+(?:a|in))\s+)"
                    r"(?P<place>[A-ZÀ-Ý][A-Za-zÀ-ÿ']+(?:[ \t]+[A-ZÀ-Ý][A-Za-zÀ-ÿ']+){0,2})")
_DATE = re.compile(r"(?<!\d)(?P<d>\d{1,2})(?P<s>[/.\-])(?P<m>\d{1,2})(?P=s)(?P<y>\d{4}|\d{2})(?!\d)")
_ISO = re.compile(r"(?<!\d)(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})(?!\d)")
_AGE = re.compile(r"(?P<n>\d{2,3})(?P<w>\s*(?i:anni|aa\.?|years?|yo|y/o))")


@dataclass
class Deidentified:
    text: str
    kinds: dict[str, int] = field(default_factory=dict)   # how many of each kind were replaced (never what)


def shift_of(case_id: str) -> int:
    """The fixed number of days a case's dates move: 30 to 364, forward or back, from the case id."""
    n = int(hashlib.sha256(f"melampo-date-shift|{case_id}".encode()).hexdigest(), 16)
    return (30 + n % 335) * (1 if (n >> 64) % 2 else -1)


def _shifted(d: int, m: int, y: int, days: int) -> dt.date | None:
    try:
        return dt.date(y, m, d) + dt.timedelta(days=days)
    except ValueError:
        return None


def deidentify(text: str, names: list[str] | tuple[str, ...] = (), shift_days: int = 0) -> Deidentified:
    """The text with what identifies the person replaced (``names``: the patient's names and surnames)."""
    kinds: dict[str, int] = {}

    def count(kind: str) -> None:
        kinds[kind] = kinds.get(kind, 0) + 1

    def sub(pattern: re.Pattern, repl, s: str) -> str:
        return pattern.sub(repl, s)

    out = text or ""
    out = sub(_CF, lambda m: (count("fiscal_code"), "[CF]")[1], out)
    out = sub(_EMAIL, lambda m: (count("contact"), "[CONTATTO]")[1], out)
    out = sub(_ID, lambda m: (count("id"), m.group("label") + "[ID]")[1], out)

    def date(m: re.Match, iso: bool = False) -> str:
        y = int(m.group("y"))
        y = y + (2000 if y < 50 else 1900) if y < 100 else y
        moved = _shifted(int(m.group("d")), int(m.group("m")), y, shift_days)
        if moved is None:
            return m.group(0)
        count("date")
        if iso:
            return moved.isoformat()
        sep = m.group("s")
        year = f"{moved.year}" if len(m.group("y")) == 4 else f"{moved.year % 100:02d}"
        return f"{moved.day:0{len(m.group('d'))}d}{sep}{moved.month:0{len(m.group('m'))}d}{sep}{year}"

    out = sub(_ISO, lambda m: date(m, True), out)
    out = sub(_DATE, date, out)
    out = sub(_PHONE, lambda m: (count("contact"), "[CONTATTO]")[1], out)
    out = sub(_ADDRESS, lambda m: (count("address"), "[INDIRIZZO]")[1], out)
    out = sub(_PLACE, lambda m: (count("place"), m.group("label") + "[LUOGO]")[1], out)
    out = sub(_LABELLED_NAME, lambda m: (count("name"), m.group("label") + "[NOME]")[1], out)
    for name in sorted({w for n in names for w in _plain(n).split() if len(w) > 1}, key=len, reverse=True):
        out, n = re.subn(r"(?<![\wÀ-ÿ])" + re.escape(name) + r"(?![\wÀ-ÿ])", "[NOME]", out, flags=re.IGNORECASE)
        for _ in range(n):
            count("name")
    out = sub(_AGE, lambda m: ((count("age"), "[ETÀ>89]" + m.group("w"))[1] if int(m.group("n")) > 89
                               else m.group(0)), out)
    return Deidentified(out, kinds)


__all__ = ["Deidentified", "deidentify", "shift_of"]
