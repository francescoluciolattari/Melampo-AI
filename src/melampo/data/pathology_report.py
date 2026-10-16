"""The pathology report, read deterministically: sections, staging and grading systems, margins, lymph nodes,
receptors and biomarkers, the diagnosis handed to the reader (16 October 2026).

The report router (``report_router``) sends a document here when it holds a histology or cytology report. What a
pathologist writes in a structured way is read with grammars of the published systems -- never guessed, never
completed:

* **sections**: material, clinical information, macroscopy, microscopy, immunohistochemistry, molecular tests,
  diagnosis, comment (Italian and English headings, "Diagnosi:" inline or as a heading line);
* **TNM** (AJCC/UICC: prefixes y, r, a, p, c; T/N/M with their sub-letters and suffixes "(m)", "(sn)", "mi",
  "(i+)"), the edition when printed;
* **grading and staging systems**: G1-G4, Gleason "3+4=7", ISUP / Grade Group, Nottingham "grado 2 (3+2+2)",
  FIGO, Ishak / METAVIR (grading-staging of chronic hepatitis), Breslow thickness, Clark level, R0/R1/R2;
* **margins**: free / involved, the minimum distance when printed;
* **lymph nodes**: "2/15", "2 su 15", "2 of 15 lymph nodes";
* **invasion**: lymphovascular, perineural (present / absent);
* **receptors and biomarkers**: ER, PgR, HER2 (score, ISH), Ki-67, PD-L1 (TPS, CPS), mismatch-repair proteins
  (retained / lost), and the like, with the percentage or score as printed;
* **diagnosis**: its text and offsets, for the reader and the linker (the structure, the role, the qualifiers).

Each item keeps the words as printed and their offsets. A value is never converted, an item never inferred from
another (a pT2 does not make a stage). Presence and absence come from the words of the item itself; the clause
certainty from the ConText detector. Lines of a structured section (immunohistochemistry, molecular) that no
grammar reads are returned as residue: nothing vanishes silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..memory.assertion import AssertionDetector, select_cues

# --- sections ----------------------------------------------------------------------------------------------
SECTIONS = {
    "material": r"materiale(?: inviato| in esame)?|campione|prelievo|specimen(?:s)?(?: submitted)?|material",
    "clinical": r"notizie cliniche|quesito(?: clinico)?|dati clinici|clinical (?:information|history|data)",
    "macroscopy": r"(?:esame |descrizione )?macroscopic[ao]|macroscopia|gross(?: description| examination)?",
    "microscopy": r"(?:esame |descrizione )?microscopic[ao]|microscopia|microscopic(?: description| examination)?",
    "immunohistochemistry": r"(?:indagini |esame |profilo )?immunoistochimic[ao]|immunoistochimica|ihc|"
                            r"immunohistochemi(?:stry|cal (?:stains|study|profile))",
    "molecular": r"(?:indagini|biologia|analisi) molecolar[ei]|molecular(?: tests| studies| analysis)?|ngs",
    "diagnosis": r"diagnosi(?: istologica| citologica| conclusiva| finale)?|conclusion[ei]|"
                 r"(?:final |pathologic(?:al)? )?diagnosis|conclusion",
    "comment": r"commento|note|nota|comment",
}
_SECTION_RE = re.compile(
    r"^\s*(?:#+\s*)?(?P<head>" + "|".join(f"(?P<{k}>{v})" for k, v in SECTIONS.items()) + r")\s*(?::|\.|$)\s*",
    re.IGNORECASE,
)
STRUCTURED_SECTIONS = frozenset({"immunohistochemistry", "molecular"})

# --- grammars of the published systems ---------------------------------------------------------------------
_TNM_T = r"(?:(?:y|r|a){0,2}[pc]?)T(?:is|x|0|[1-4])[a-d]?\d?(?:\((?:m|\d+)\))?"
_TNM_N = r"(?:(?:y|r|a){0,2}[pc]?)N(?:x|0|[1-3])[a-c]?(?:mi)?(?:\((?:sn|i[+-]|mol[+-]|\d+/\d+)\))?"
_TNM_M = r"(?:(?:y|r|a){0,2}[pc]?)M(?:x|0|1)[a-d]?"
_TNM_RE = re.compile(rf"(?<![A-Za-z])(?P<T>{_TNM_T})\s*(?P<N>{_TNM_N})?\s*(?P<M>{_TNM_M})?(?![A-Za-z])")
_TNM_LONE_RE = re.compile(rf"(?<![A-Za-z])(?P<N>{_TNM_N}|{_TNM_M})(?![A-Za-z])")
_EDITION_RE = re.compile(r"(?:AJCC|UICC|TNM)[^.;\n]{0,12}?(\d{1,2})\s*(?:a|th|°|ª)?\s*(?:ed(?:izione|ition)?\.?)?",
                         re.IGNORECASE)

_SYSTEMS = (
    ("gleason", re.compile(r"\bgleason(?:\s+score)?\s*:?\s*(?P<v>\d\s*\+\s*\d(?:\s*=\s*\d+)?|\d+)", re.I)),
    ("grade_group", re.compile(r"\b(?:isup|grade\s+group|gruppo\s+di\s+grado|grado\s+isup)\s*:?\s*(?P<v>[1-5])\b", re.I)),
    ("nottingham", re.compile(r"\b(?:nottingham|elston[- ]ellis|grado\s+istologico)[^.;\n]{0,25}?"
                              r"(?P<v>[1-3](?:\s*\(\s*\d\s*\+\s*\d\s*\+\s*\d\s*\))?)", re.I)),
    ("grade", re.compile(r"\b(?:grad(?:o|ing|e)|differenziazione)\s*:?\s*(?P<v>G[1-4X]|[1-4](?!\s*\+))\b", re.I)),
    ("grade", re.compile(r"(?<![A-Za-z])(?P<v>G[1-4])(?![A-Za-z0-9])")),
    ("figo", re.compile(r"\bFIGO\s*(?:stadio|stage|grado|grade)?\s*:?\s*(?P<v>(?:IV|I{1,3})[A-C]?\d?|[1-3])\b")),
    ("ishak", re.compile(r"\b(?:grading\s*(?P<g>\d{1,2})\s*,?\s*staging\s*(?P<s>\d)\s*\(?\s*ishak|ishak[^.;\n]{0,20}?"
                         r"(?:grading|grado|activity)\s*(?P<g2>\d{1,2})[^.;\n]{0,20}?(?:staging|stadio|fibrosis)\s*(?P<s2>\d))",
                         re.I)),
    ("metavir", re.compile(r"\b(?:metavir[^.;\n]{0,10}?)?(?P<v>A[0-3]\s*F[0-4])\b")),
    ("breslow", re.compile(r"\bbreslow[^.;\n]{0,25}?(?P<v>\d+(?:[.,]\d+)?)\s*mm", re.I)),
    ("clark", re.compile(r"\bclark[^.;\n]{0,15}?(?P<v>I{1,3}|IV|V|[1-5])\b", re.I)),
    ("residual_tumour", re.compile(r"(?<![A-Za-z])(?P<v>R[0-2X])(?![A-Za-z0-9])")),
)

_MARGIN_RE = re.compile(
    r"\bmargin(?:[ie]|s)?\b[^.;\n]{0,60}?\b(?P<status>indenn\w+|liber\w+|negativ\w+|esent\w+|non\s+(?:coinvolt|infiltrat)\w+|"
    r"positiv\w+|infiltrat\w+|coinvolt\w+|interessat\w+|free|clear|negative|uninvolved|not\s+involved|positive|involved)",
    re.I,
)
_MARGIN_FREE = re.compile(r"indenn|liber|negativ|esent|non\s+(?:coinvolt|infiltrat)|free|clear|uninvolved|not\s+involved",
                          re.I)
_DISTANCE_RE = re.compile(r"(?:distanza|distance)[^.;\n]{0,25}?(?P<v><?\s*\d+(?:[.,]\d+)?)\s*(?P<u>mm|cm)", re.I)
_NODES_RE = re.compile(
    r"(?P<pos>\d{1,3})\s*(?:/|su|of|out of)\s*(?P<tot>\d{1,3})\s*(?:\w+\s+){0,3}?"
    r"(?:linfonod\w*|lymph\s*nodes?|nodes?|ln)\b"
    r"|(?:linfonod\w*|lymph\s*nodes?)\b[^.;\n]{0,40}?(?P<pos2>\d{1,3})\s*(?:/|su|of|out of)\s*(?P<tot2>\d{1,3})",
    re.I,
)
_INVASION_RE = re.compile(
    r"\b(?P<kind>invasione\s+(?:linfo-?vascolare|vascolare|linfatica|perineurale|perinervosa)|"
    r"(?:lympho-?vascular|vascular|lymphatic|perineural)\s+invasion|LVI|PNI)\b[^.;\n]{0,25}?"
    r"(?P<status>present\w*|assent\w*|non\s+(?:evidente|osservat\w+|identificat\w+)|not\s+identified|identified|"
    r"absent|negativ\w*|positiv\w*)",
    re.I,
)
_INVASION_ABSENT = re.compile(r"assent|non\s|not\s|absent|negativ", re.I)

BIOMARKERS = {
    "ER": r"ER|RE|recettor[ei]\s+(?:per\s+)?(?:gli\s+)?estrogen\w*|estrogen\s+receptors?",
    "PgR": r"PgR|PR|RP|recettor[ei]\s+(?:per\s+)?(?:il\s+)?progesteron\w*|progesterone\s+receptors?",
    "HER2": r"HER-?2(?:/neu)?|c-?erb-?B2",
    "Ki-67": r"Ki-?67|MIB-?1",
    "PD-L1": r"PD-?L1",
    "MLH1": r"MLH1", "PMS2": r"PMS2", "MSH2": r"MSH2", "MSH6": r"MSH6",
    "p16": r"p16", "p53": r"p53", "ALK": r"ALK", "ROS1": r"ROS1", "TTF-1": r"TTF-?1", "CK7": r"CK7", "CK20": r"CK20",
    "CDX2": r"CDX-?2", "AR": r"AR|recettor[ei]\s+(?:per\s+)?(?:gli\s+)?androgen\w*",
    "p40": r"p40", "p63": r"p63", "GATA3": r"GATA-?3", "SOX10": r"SOX-?10", "S100": r"S-?100", "PAX8": r"PAX-?8",
    "WT1": r"WT-?1", "Napsin A": r"napsin(?:a)?\s*A", "CD20": r"CD20", "CD3": r"CD3", "CD30": r"CD30",
    "CD117": r"CD117|c-?kit", "DOG1": r"DOG-?1", "synaptophysin": r"sinaptofisina|synaptophysin",
    "chromogranin": r"cromogranina|chromogranin(?:\s*A)?", "CK5/6": r"CK\s*5/6",
}
# Genes tested for variants, and microsatellite status (molecular section or inline).
GENES = ("EGFR", "KRAS", "NRAS", "BRAF", "ALK", "ROS1", "MET", "RET", "NTRK", "PIK3CA", "IDH1", "IDH2", "BRCA1",
         "BRCA2", "ERBB2", "KIT", "PDGFRA", "TP53", "POLE")
_GENE_RE = re.compile(r"(?<![\w-])(?P<gene>" + "|".join(GENES) + r")(?![\w-])")
_VARIANT_RE = re.compile(r"\b(?:p\.)?[A-Z]\d{2,4}[A-Z*]\b|\bexon(?:e)?\s*\d{1,2}\s*(?:del\w*|ins\w*)|\b(?:del|ins)\s*\d+", re.I)
_GENE_STATUS = (
    (re.compile(r"wild[\s-]?type|\bwt\b|non\s+mutat\w*|not\s+mutated|no\s+mutation|assenza\s+di\s+mutazion\w*|"
                r"negativ\w*", re.I), "wild_type"),
    (re.compile(r"non\s+riarrangiat\w*|not\s+rearranged", re.I), "not_rearranged"),
    (re.compile(r"riarrangiat\w*|rearranged|fusion\w*|fusione", re.I), "rearranged"),
    (re.compile(r"non\s+amplificat\w*|not\s+amplified", re.I), "not_amplified"),
    (re.compile(r"amplificat\w*|amplified", re.I), "amplified"),
    (re.compile(r"mutat\w*|mutated|mutation\s+detected|presenza\s+di\s+mutazion\w*|positiv\w*", re.I), "mutated"),
)
_MSI_RE = re.compile(r"\b(?P<v>MSI-?H|MSI-?L|MSS|dMMR|pMMR|microsatellite[\s-]+(?:stable|instab\w+)|"
                     r"(?:instabilit[aà]|stabilit[aà])\s+(?:dei\s+)?microsatellit\w*|microsatelliti\s+stabili)\b", re.I)
_MARKER_NAME_RE = re.compile(r"(?<![\w-])(?P<name>" + "|".join(f"(?:{v})" for v in BIOMARKERS.values()) + r")(?![\w-])",
                             re.IGNORECASE)
_BM_SCORE = re.compile(r"(?:score\s*)?(?<![\d.,])(?P<score>[0-3]\s*\+|\+{1,3})")
_BM_PERCENT = re.compile(r"(?P<cmp>[<>≤≥]?)\s*(?P<pct>\d{1,3}(?:[.,]\d+)?)\s*%")
_BM_INDEX = re.compile(r"\b(?P<idx>TPS|CPS|IC|TC)\b\s*:?\s*(?P<cmp>[<>≤≥]?)\s*(?P<v>\d{1,3}(?:[.,]\d+)?)\s*%?", re.I)
_BM_STATUS = re.compile(
    r"(?P<st>positiv\w*|negativ\w*|conservat\w*|mantenut\w*|pers[aoie]\b|assent\w*|espress\w*|"
    r"non\s+amplificat\w*|amplificat\w*|retained|intact|lost|loss|absent|present|positive|negative|"
    r"not\s+amplified|amplified|equivocal|equivoc\w*|dubbi\w*)",
    re.I,
)
_STATUS_MEANING = (
    (re.compile(r"non\s+amplificat|not\s+amplified", re.I), "not_amplified"),
    (re.compile(r"amplificat|amplified", re.I), "amplified"),
    (re.compile(r"conservat|mantenut|retained|intact", re.I), "retained"),
    (re.compile(r"pers[aoie]\b|lost|loss|assent|absent", re.I), "lost_or_absent"),
    (re.compile(r"equivocal|equivoc|dubbi", re.I), "equivocal"),
    (re.compile(r"negativ", re.I), "negative"),
    (re.compile(r"positiv|espress|present", re.I), "positive"),
)
_SIZE_RE = re.compile(r"(?:dimension[ei](?:\s+massim[ae])?|diametro(?:\s+massimo)?|size|greatest dimension|"
                      r"misura)[^.;\n]{0,20}?(?P<v>\d+(?:[.,]\d+)?(?:\s*[x×]\s*\d+(?:[.,]\d+)?){0,2})\s*(?P<u>mm|cm)",
                      re.I)


@dataclass(frozen=True)
class Item:
    kind: str                 # tnm, grade, gleason, ..., margin, lymph_nodes, invasion, biomarker, size, edition
    value: dict[str, Any]     # the value as printed, in parts ("T": "pT2", "score": "2+", "percent": "90")
    text: str
    start: int
    end: int
    section: str

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "value": dict(self.value), "text": self.text, "start": self.start,
                "end": self.end, "section": self.section}


@dataclass
class PathologyReading:
    sections: dict[str, tuple[int, int]] = field(default_factory=dict)   # name -> (start, end) offsets
    items: list[Item] = field(default_factory=list)
    diagnosis: str = ""
    diagnosis_span: tuple[int, int] | None = None
    residue: list[tuple[int, str]] = field(default_factory=list)          # (line number, line)

    def of_kind(self, kind: str) -> list[Item]:
        return [i for i in self.items if i.kind == kind]

    def summary(self) -> dict[str, Any]:
        return {"sections": {k: list(v) for k, v in self.sections.items()},
                "items": [i.as_dict() for i in self.items], "diagnosis": self.diagnosis,
                "diagnosis_span": list(self.diagnosis_span) if self.diagnosis_span else None,
                "residue": [{"line": n, "text": t} for n, t in self.residue]}


def split_sections(text: str) -> list[tuple[str, int, int]]:
    """(section, start, end) over the text; the part before the first heading is "header"."""
    marks: list[tuple[str, int, int]] = []   # (name, heading start, content start)
    offset = 0
    for line in text.splitlines(keepends=True):
        m = _SECTION_RE.match(line)
        if m and (len(line.strip()) < 60 or line[m.end("head"):m.end("head") + 2].strip().startswith(":")):
            name = next(k for k in SECTIONS if m.group(k))
            marks.append((name, offset, offset + m.end()))
        offset += len(line)
    out: list[tuple[str, int, int]] = []
    if not marks or marks[0][1] > 0:
        out.append(("header", 0, marks[0][1] if marks else len(text)))
    for i, (name, _, content) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else len(text)
        out.append((name, content, end))
    return out


def _section_at(sections: list[tuple[str, int, int]], at: int) -> str:
    for name, lo, hi in sections:
        if lo <= at < hi:
            return name
    return "header"


def _meaning(status: str) -> str:
    return next((m for rx, m in _STATUS_MEANING if rx.search(status)), status.lower())


def read_pathology(text: str, language: str = "it") -> PathologyReading:
    """Every structured item of a pathology report, its sections and its diagnosis."""
    sections = split_sections(text)
    reading = PathologyReading()
    for name, lo, hi in sections:
        if name not in reading.sections:
            reading.sections[name] = (lo, hi)
    items: list[Item] = []

    def add(kind, value, m, group=0):
        items.append(Item(kind, {k: v for k, v in value.items() if v not in (None, "")},
                          m.group(group), m.start(group), m.end(group), _section_at(sections, m.start(group))))

    taken: list[tuple[int, int]] = []
    for m in _TNM_RE.finditer(text):
        # a bare "T2" is a TNM category only with its N ("T2 N0"); alone it is an MRI weighting ("T2 pesate") or a
        # vertebra ("T11"); a prefixed one ("pT2", "ypT1") is a category by itself
        if not (re.match(r"[yra]*[pc]", m.group("T")) or m.group("N")):
            continue
        add("tnm", {"T": m.group("T"), "N": m.group("N"), "M": m.group("M")}, m)
        taken.append((m.start(), m.end()))
    for m in _TNM_LONE_RE.finditer(text):
        if not any(a <= m.start() < b for a, b in taken) and re.match(r"[ycrap]*[pc]", m.group("N")):
            add("tnm", {("N" if "N" in m.group("N") else "M"): m.group("N")}, m)
    for m in _EDITION_RE.finditer(text):
        add("tnm_edition", {"edition": m.group(1)}, m)
    for kind, rx in _SYSTEMS:
        for m in rx.finditer(text):
            if kind == "grade" and any(i.start <= m.start() < i.end for i in items):
                continue  # "grado 2 (3+2+2)" already read as Nottingham, "grading 2, staging 1" as Ishak
            if kind == "ishak":
                add("ishak", {"grading": m.group("g") or m.group("g2"), "staging": m.group("s") or m.group("s2")}, m)
            elif kind == "residual_tumour" and _section_at(sections, m.start()) not in ("diagnosis", "comment", "header"):
                continue
            else:
                add(kind, {"value": re.sub(r"\s+", "", m.group("v"))}, m)
    for m in _MARGIN_RE.finditer(text):
        value = {"status": "free" if _MARGIN_FREE.search(m.group("status")) else "involved", "as_printed": m.group("status")}
        d = _DISTANCE_RE.search(text, m.end(), min(len(text), m.end() + 80))
        if d and "\n" not in text[m.end():d.start()]:
            value.update(distance=d.group("v").replace(" ", ""), unit=d.group("u"))
        add("margin", value, m)
    for m in _NODES_RE.finditer(text):
        pos, tot = m.group("pos") or m.group("pos2"), m.group("tot") or m.group("tot2")
        if int(pos) <= int(tot):
            add("lymph_nodes", {"positive": pos, "examined": tot}, m)
    for m in _INVASION_RE.finditer(text):
        kind = "perineural" if re.search(r"perineur|perinerv|PNI", m.group("kind"), re.I) else "lymphovascular"
        add("invasion", {"type": kind, "status": "absent" if _INVASION_ABSENT.search(m.group("status")) else "present",
                         "as_printed": m.group("status")}, m)
    for m in _SIZE_RE.finditer(text):
        add("size", {"value": m.group("v"), "unit": m.group("u")}, m)
    names = [(m.start("name"), m.end("name"), m.group("name")) for m in _MARKER_NAME_RE.finditer(text)]
    for n, (start, name_end, written) in enumerate(names):
        name = next(k for k, v in BIOMARKERS.items() if re.fullmatch(v, written, re.I))
        # the words of this marker run to the next marker, the end of the sentence or 60 characters
        stop = min([names[n + 1][0]] if n + 1 < len(names) else [len(text)])
        tail = text[name_end:min(stop, name_end + 60)]
        tail = re.split(r"[.;\n]", tail, maxsplit=1)[0]
        value: dict[str, Any] = {"marker": name}
        ish = re.search(r"\b(?:FISH|SISH|CISH|ISH)\b", tail)
        ihc_part, ish_part = (tail[: ish.start()], tail[ish.end():]) if ish else (tail, "")
        if (sc := _BM_SCORE.search(ihc_part)) and name in ("HER2", "PD-L1", "p16", "p53", "ER", "PgR", "AR"):
            value["score"] = re.sub(r"\s+", "", sc.group("score"))
        if (i := _BM_INDEX.search(ihc_part)):
            value["index"] = i.group("idx").upper()
            value["percent"] = (i.group("cmp") or "") + i.group("v")
        elif (pc := _BM_PERCENT.search(ihc_part)):
            value["percent"] = (pc.group("cmp") or "") + pc.group("pct")
        if (st := _BM_STATUS.search(ihc_part)):
            value["status"] = _meaning(st.group("st"))
            value["as_printed"] = st.group("st")
        if ish and (st := _BM_STATUS.search(ish_part)):
            value["ish"] = _meaning(st.group("st"))
            value["ish_as_printed"] = ish.group(0) + ": " + st.group("st")
        if len(value) == 1:
            continue
        end = name_end + len(tail.rstrip(" ,"))
        items.append(Item("biomarker", value, text[start:end], start, end, _section_at(sections, start)))
    genes = list(_GENE_RE.finditer(text))
    for n, m in enumerate(genes):
        stop = genes[n + 1].start() if n + 1 < len(genes) else len(text)
        tail = re.split(r"[.;\n]", text[m.end():min(stop, m.end() + 60)], maxsplit=1)[0]
        status = next((meaning for rx, meaning in _GENE_STATUS if rx.search(tail)), "")
        variant = _VARIANT_RE.search(tail) or _VARIANT_RE.match(text, m.end() + 1)
        if not status and not variant:
            continue
        if any(i.kind == "biomarker" and i.start == m.start() for i in items):
            continue  # "ALK negative" by immunohistochemistry, already read as a marker
        value = {"gene": m.group("gene")}
        if status:
            value["status"] = status
        if variant:
            # "BRAF V600E: wild type" -- the variant tested, not found; "KRAS G12C mutato" -- the variant found
            value["variant_tested" if status == "wild_type" else "variant"] = variant.group(0)
        end = m.end() + len(tail.rstrip(" ,"))
        items.append(Item("molecular", value, text[m.start():end], m.start(), end, _section_at(sections, m.start())))
    for m in _MSI_RE.finditer(text):
        add("microsatellite", {"value": m.group("v")}, m, "v")
    items.sort(key=lambda i: i.start)
    reading.items = items

    for name, lo, hi in sections:
        if name == "diagnosis" and not reading.diagnosis:
            reading.diagnosis = text[lo:hi].strip()
            lead = len(text[lo:hi]) - len(text[lo:hi].lstrip())
            reading.diagnosis_span = (lo + lead, lo + lead + len(reading.diagnosis))

    # residue: a line of a structured section (immunohistochemistry, molecular) that no item covers
    offset = 0
    for number, line in enumerate(text.splitlines(keepends=True), start=1):
        lo, hi = offset, offset + len(line.rstrip("\n"))
        offset += len(line)
        if (line.strip() and _section_at(sections, lo) in STRUCTURED_SECTIONS and not _SECTION_RE.match(line)
                and re.search(r"\d|\+|positiv|negativ|conservat|pers", line, re.I)
                and not any(i.start < hi and i.end > lo for i in items)):
            reading.residue.append((number, line.strip()))
    return reading


def certainty_of(text: str, item: Item, language: str = "it") -> dict[str, str]:
    """The ConText reading of the clause of an item (a suspected margin, a "possible" invasion)."""
    status = AssertionDetector(cues=select_cues(language)).detect(text, item.start, item.end)
    return {"certainty": status.certainty, "polarity": status.polarity}


__all__ = ["BIOMARKERS", "SECTIONS", "Item", "PathologyReading", "certainty_of", "read_pathology", "split_sections"]
