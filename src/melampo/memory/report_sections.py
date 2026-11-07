"""The sections of a report (6 November 2026): what a radiologist reads before the findings, and after them.

A radiologist reads a report as a document: first the exam and why it was asked (the referral, the clinical
question), then the technique and the comparison, then the findings organ by organ, last the conclusion, which
repeats what matters. "Signs of heart failure" in the referral is the question, not a finding of the heart
(claims4, 5 November 2026: four sentences of the referral were attached as current findings).

The kinds of section are those of the LOINC report sections of the radiology templates (RSNA RadReport, DICOM
SR): indication (18785-6, "Indications for procedure"), history (11329-0), comparison (18834-2), technique
(55111-9, "Current imaging procedure descriptions"), findings (18782-3, "Study observation"), impression (19005-8,
"Impression"). The headings below are the names reports give them, in English and Italian (and the English of
PARROT's translations): names of parts of a document, which say nothing ambiguous of the body. A heading that is
none of them (an organ: "Bladder:", "TZ:") is not a section kind, and is left as the section's name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

INDICATION, HISTORY, COMPARISON, TECHNIQUE, FINDINGS, IMPRESSION = (
    "indication", "history", "comparison", "technique", "findings", "impression")
LOINC = {INDICATION: "18785-6", HISTORY: "11329-0", COMPARISON: "18834-2", TECHNIQUE: "55111-9",
         FINDINGS: "18782-3", IMPRESSION: "19005-8"}

_HEADINGS = {
    INDICATION: (
        "indication", "indications", "clinical indication", "clinical indications", "reason for exam",
        "reason for examination", "reason for study", "clinical question", "clinical query", "clinical information",
        "clinical data", "clinical details", "referral", "referral data", "referral information",
        "data on the referral", "data from the referral", "clinical data from the referral", "exam request",
        "request", "diagnostic question", "question",
        "indicazione", "indicazioni", "indicazione clinica", "quesito", "quesito clinico", "quesito diagnostico",
        "motivo", "motivo dell'esame", "notizie cliniche", "dati clinici", "richiesta",
        "indicazione e note anamnestiche"),
    HISTORY: ("history", "clinical history", "medical history", "past history", "anamnesis", "anamnesi",
              "storia clinica"),
    COMPARISON: ("comparison", "comparisons", "prior study", "prior studies", "previous study", "previous studies",
                 "previous examination", "confronto", "esami precedenti", "precedenti"),
    TECHNIQUE: ("technique", "technical details", "examination technique", "protocol", "procedure", "acquisition",
                "method", "methods", "sequences", "tecnica", "tecnica d'esame", "dati tecnici", "protocollo",
                "metodica", "sequenze"),
    FINDINGS: ("findings", "finding", "results", "result", "description", "examination description",
               "observations", "study observation", "reperti", "reperto", "descrizione", "esito", "risultati"),
    IMPRESSION: ("impression", "impressions", "conclusion", "conclusions", "summary", "assessment",
                 "diagnostic impression", "opinion", "interpretation", "final impression", "conclusioni",
                 "conclusione", "giudizio", "impressione", "impressione diagnostica", "sintesi", "valutazione"),
}
_KIND = {name: kind for kind, names in _HEADINGS.items() for name in names}
_HEADING = re.compile(r"(?:^|[\n.;])\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ' /\-]{1,48}):", re.MULTILINE)


def kind_of(heading: str) -> str:
    """The section kind a heading names, or "" (an organ, a label of a table). "Findings and assessment",
    "Befund / Beurteilung": a heading of two parts takes the kind of its first."""
    h = " ".join(re.sub(r"[^a-zà-ÿ' /-]", " ", heading.lower()).split())
    if h in _KIND:
        return _KIND[h]
    first = re.split(r"\s+(?:and|e|ed)\s+|\s*/\s*", h)[0].strip()
    return _KIND.get(first, "")


@dataclass(frozen=True)
class Section:
    start: int          # where the heading starts
    body: int           # where the text after the heading starts
    end: int            # where the next section starts (or the report ends)
    heading: str
    kind: str           # INDICATION ... IMPRESSION, or "" for a heading that is not a section kind


def sections(report: str) -> list[Section]:
    """The sections of the report, in order. Only headings of a section kind open a section; an organ heading
    ("Liver:") stays inside its section (the findings). A heading followed by a number is a measure ("Ao: 33 mm")."""
    found = []
    for m in _HEADING.finditer(report or ""):
        if re.match(r"\s*[\d<>~≈]", report[m.end():m.end() + 6]):
            continue
        kind = kind_of(m.group(1))
        if kind:
            found.append((m.start(1), m.end(), m.group(1).strip(), kind))
    out = []
    for i, (start, body, heading, kind) in enumerate(found):
        end = found[i + 1][0] if i + 1 < len(found) else len(report)
        out.append(Section(start, body, end, heading, kind))
    return out


def section_at(report: str, position: int) -> Section | None:
    """The section a position of the report is in, or None before any heading of a section kind."""
    current = None
    for s in sections(report):
        if s.start <= position:
            current = s
        else:
            break
    return current


def impression_of(report: str, limit: int = 600) -> str:
    """The text of the report's conclusion (the last impression section), at most ``limit`` characters."""
    found = [s for s in sections(report) if s.kind == IMPRESSION]
    if not found:
        return ""
    s = found[-1]
    return " ".join(report[s.body:s.end].split())[:limit]


__all__ = ["COMPARISON", "FINDINGS", "HISTORY", "IMPRESSION", "INDICATION", "LOINC", "TECHNIQUE", "Section",
           "impression_of", "kind_of", "section_at", "sections"]
