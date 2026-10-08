"""The state of a report, read before any mention is linked (stage 1 of the nine, T2).

A radiologist reads a report from the top: the clinical question and the technique say which
region and which modality to expect, and every later mention is read against that expectation
(predictive comprehension; see docs: ``architettura_parallela_predittiva_linker``). This module
reads the cheap, certain part of that state and nothing else:

* the sections the report states itself ("Quesito clinico:", "Dati tecnici:", "Referto",
  "Conclusioni:", "Technique:", "Findings:", "Impression:") and the technique sentences that
  are not labelled ("L'esame è stato eseguito con sequenze T1 e T2");
* the language, the modalities and the region the header names: the spine scope, and the area of
  the exam (``exam_area``: the exam's name in the title, the technique or the first sentence);
* sentence spans that never include a header label, so a sentence handed to the linker is the
  sentence of the finding and not "Quesito clinico: sospetta colelitiasi Il fegato ...".

It makes no decision about any structure. What the state is used for is decided by the linker,
stream by stream, and every use is recorded in the trace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from melampo.memory.exam_area import ExamAreas
from melampo.memory.word_senses import detect_language


@lru_cache(maxsize=1)
def _areas() -> ExamAreas:
    return ExamAreas.load()


CLINICAL, TECHNIQUE, COMPARISON, FINDINGS, CONCLUSIONS = (
    "clinical",
    "technique",
    "comparison",
    "findings",
    "conclusions",
)

_LABELS = {
    CLINICAL: (
        "quesito clinico",
        "quesito diagnostico",
        "quesito",
        "indicazione clinica",
        "indicazioni",
        "indicazione",
        "indicazione e note anamnestiche",
        "note anamnestiche",
        "dati anamnestici",
        "anamnesi",
        "notizie cliniche",
        "clinical history",
        "clinical information",
        "clinical indication",
        "clinical",
        "indication",
        "history",
        "reason for exam",
    ),
    TECHNIQUE: (
        "tecnica di acquisizione",
        "tecnica d'esame",
        "tecnica",
        "dati tecnici",
        "protocollo",
        "sequenze",
        "technique",
        "protocol",
        "exam",
        "examination",
    ),
    COMPARISON: ("confronto", "esami precedenti", "comparison", "prior studies"),
    FINDINGS: ("referto", "descrizione", "reperti", "findings", "report"),
    CONCLUSIONS: (
        "conclusioni",
        "conclusione",
        "impressione diagnostica",
        "impression",
        "conclusion",
        "conclusions",
    ),
}
_KIND_OF_LABEL = {label: kind for kind, labels in _LABELS.items() for label in labels}
_ALTERNATION = "|".join(
    re.escape(label) for label in sorted(_KIND_OF_LABEL, key=len, reverse=True)
)
# A label with its colon, at the start or after the end of a sentence or a line.
_LABEL = re.compile(
    rf"(?:^|(?<=[.!?\n”\"\)\]]))\s*(?P<label>{_ALTERNATION})\s*:", re.IGNORECASE
)
# "Referto Assenza di lesioni": the report heading without a colon, before a capital letter.
_BARE_FINDINGS = re.compile(
    r"(?:^|(?<=[.!?\n”\"\)\]]))\s*(?P<label>(?i:referto|findings))\s+(?=[A-ZÀ-Ý])"
)
# A clinical question often runs into the findings with no full stop:
# "Quesito clinico: sospetta colelitiasi Il fegato è nei limiti".
_CLINICAL_RUN_ON = re.compile(
    r"(?<=[A-Za-zà-ÿ0-9])\s+(?=(?:Il|La|Lo|Le|I|Gli|L['’]|Un|Una|Sul|Sulla|Nel|Nella|Nei|Nelle|In|Si|Non"
    r"|A livello|Presenza|Assenza|Regolare|Normale|Esame)\b\s*[a-zà-ÿ])"
)
# The exam name in capitals at the start: "RM RACHIDE LOMBOSACRALE", "RX TORACE IN 2 PROIEZIONI".
_TITLE = re.compile(
    r"^\s*(?P<title>[A-ZÀ-Ý][A-ZÀ-Ý0-9\-/]+(?:\s+[A-ZÀ-Ý0-9\-/]+){1,12})(?=\s+[A-ZÀ-Ý][a-zà-ÿ]|\s*\n|\s*$)"
)
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[A-ZÀ-Ý0-9“\"(\[])|\n+")
# "L'esame è stato eseguito con ...", "Esame TC eseguito senza mdc", "Sequenze: assiali TSE T1".
_TECHNIQUE_SENTENCE = re.compile(
    r"^\W*(?:l['’]\s*)?(?:esame|indagine|studio|rm|tc)\b[^.;]{0,60}?\b(?:eseguit\w+|acquisit\w+|condott\w+)\b"
    r"|^\W*sequenze\b|^\W*(?:the\s+)?(?:study|examination|exam)\s+(?:was\s+)?(?:performed|obtained)\b",
    re.IGNORECASE,
)

_MODALITIES = {
    "ct": re.compile(
        r"\b(?:tc|tac|ct|tomografia|computed tomography)\b", re.IGNORECASE
    ),
    "mri": re.compile(
        r"\b(?:rm|mri|risonanza|sequenze|sequences?|tse|stir|flair)\b", re.IGNORECASE
    ),
    "ultrasound": re.compile(
        r"\b(?:ecograf\w*|ultrasound|sonograph\w*|doppler)\b", re.IGNORECASE
    ),
    "xray": re.compile(
        r"\b(?:rx|radiograf\w*|x-?ray|radiograph\w*|frontal and lateral|pa and lateral)\b",
        re.IGNORECASE,
    ),
    "pet": re.compile(r"\bpet\b", re.IGNORECASE),
}
# The header names the spine as the region studied. Narrower than REGION_CUES["spine"] on
# purpose: "osseo", "canale" or "frattura" appear in reports of any region.
_SPINE_SCOPE = re.compile(
    r"\b(?:rachide|colonna(?:\s+vertebrale)?|spine|spinal|vertebral\w*|vertebra|lombo\w*|lumbar|lumbosacral\w*"
    r"|dorsolombar\w*|cervicodorsal\w*|rachid\w*|lordosi|cifosi|scoliosi|kyphosis|lordosis|scoliosis"
    r"|somi|soma|somatic\w*|metameri|interapofisari\w*|intersomatic\w*|uncoartrosi|cervicouncoartrosi"
    r"|spondil\w*|spondyl\w*|limitant\w*|schmorl|\w*listesi|\w*listhesis)\b",
    re.IGNORECASE,
)
_OTHER_REGIONS = re.compile(
    r"\b(?:addome|addominale|torace|toracico|encefalo|cranio|capo|collo|pelvi|pelvico|abdomen|abdominal"
    r"|chest|thorax|brain|head|neck|pelvis|ginocchio|spalla|anca|knee|shoulder|hip|mammell\w*"
    r"|fegato|epatic\w*|milza|rene|reni|renal\w*|polmon\w*|pleur\w*|cuore|cardiac\w*|vescica|prostata"
    r"|utero|ovai\w*|pancreas|tiroide|surren\w*|colecisti|liver|spleen|kidney\w*|lungs?|heart|bladder"
    r"|stomaco|intestin\w*|colon|aorta|mediastin\w*)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Span:
    kind: str
    start: int
    end: int


@dataclass(frozen=True)
class ReportState:
    text: str
    sentences: tuple[Span, ...] = ()
    language: str | None = None
    modalities: frozenset[str] = frozenset()
    spine_scope: bool = False
    labelled: bool = False
    labels: tuple[tuple[str, int, int], ...] = field(default=(), compare=False)
    areas: frozenset[str] = frozenset()
    exam_sides: frozenset[str] = frozenset()

    @classmethod
    def parse(cls, text: str) -> "ReportState":
        segments = _segments(text)
        sentences: list[Span] = []
        for kind, start, end in segments:
            for index, (s_start, s_end) in enumerate(_split(text, start, end, kind)):
                sentence_kind = kind
                technique = bool(_TECHNIQUE_SENTENCE.search(text[s_start:s_end]))
                if kind == FINDINGS and technique:
                    sentence_kind = TECHNIQUE
                elif kind in (CLINICAL, TECHNIQUE, COMPARISON) and index > 0:
                    # a labelled header is one sentence; what follows it is the report
                    sentence_kind = TECHNIQUE if technique else FINDINGS
                sentences.append(Span(sentence_kind, s_start, s_end))
        header = " ".join(
            text[s.start : s.end] for s in sentences if s.kind in (CLINICAL, TECHNIQUE)
        )
        modalities = frozenset(
            name for name, pattern in _MODALITIES.items() if pattern.search(header)
        )
        # The spine is what the report is about: the header says so, or at least two sentences
        # speak of it, and nothing in the whole report names another region.
        spine_sentences = sum(
            1 for s in sentences if _SPINE_SCOPE.search(text[s.start : s.end])
        )
        spine = bool(
            (_SPINE_SCOPE.search(header) or spine_sentences >= 2)
            and not _OTHER_REGIONS.search(text)
        )
        labels = tuple(
            (m.group("label").lower(), m.start("label"), m.end())
            for m in _all_labels(text)
        )
        # The area of the exam: where the exam is named (title, technique, first sentence), never the
        # clinical history, which speaks of other regions as a matter of course.
        named = [text[s.start : s.end] for s in sentences if s.kind == TECHNIQUE]
        if sentences:
            named.append(text[sentences[0].start : sentences[0].end])
        areas = frozenset().union(*(_areas().of(chunk) for chunk in named))
        # The side the exam name states ("RM ginocchio destro"), read the same way as the area.
        exam_sides = frozenset().union(*(_areas().sides_of(chunk) for chunk in named))
        return cls(
            text=text,
            sentences=tuple(sentences),
            language=detect_language(text),
            modalities=modalities,
            spine_scope=spine,
            labelled=bool(labels),
            labels=labels,
            areas=areas,
            exam_sides=exam_sides,
        )

    def span_at(self, position: int) -> Span | None:
        for span in self.sentences:
            if span.start <= position < span.end:
                return span
        return None

    def section_at(self, position: int) -> str:
        """The section of the sentence that holds ``position``; ``label`` inside a header label."""
        span = self.span_at(position)
        if span:
            return span.kind
        for _, start, end in self.labels:
            if start <= position < end:
                return "label"
        return FINDINGS

    def sentence_at(self, start: int, end: int) -> str:
        """The sentence that holds the mention: no header label, no neighbour sentence."""
        span = self.span_at(start)
        if span is None:
            return self.text[start:end]
        return self.text[span.start : span.end].strip()

    def header(self, limit: int = 600) -> str:
        """Clinical question and technique, as context for an ambiguous form."""
        parts = [
            self.text[s.start : s.end].strip()
            for s in self.sentences
            if s.kind in (CLINICAL, TECHNIQUE)
        ]
        return " ".join(parts)[:limit]


_LABEL_AFTER_TITLE = re.compile(rf"\s*(?P<label>{_ALTERNATION})\s*:", re.IGNORECASE)


def _title(text: str) -> re.Match | None:
    """The exam name in capitals at the start, unless it is only the anonymisation mask."""
    match = _TITLE.match(text)
    if match and set(match.group("title").replace(" ", "")) != {"X"}:
        return match
    return None


def _all_labels(text: str) -> list[re.Match]:
    found = list(_LABEL.finditer(text)) + list(_BARE_FINDINGS.finditer(text))
    title = _title(text)
    if title:
        after = _LABEL_AFTER_TITLE.match(text, title.end())
        if after:
            found.append(after)
    seen: set[int] = set()
    unique = []
    for match in sorted(found, key=lambda m: m.start("label")):
        if match.start("label") not in seen:
            seen.add(match.start("label"))
            unique.append(match)
    return unique


def _segments(text: str) -> list[tuple[str, int, int]]:
    """(kind, start, end) of each stretch of text, labels excluded."""
    labels = _all_labels(text)
    title = _title(text)
    out: list[tuple[str, int, int]] = []
    cursor = 0
    if title:
        out.append((TECHNIQUE, title.start("title"), title.end("title")))
        cursor = title.end()
    if not labels:
        if text[cursor:].strip():
            out.append((FINDINGS, cursor, len(text)))
        return out or [(FINDINGS, 0, len(text))]
    if (
        labels[0].start("label") > cursor
        and text[cursor : labels[0].start("label")].strip()
    ):
        out.append((FINDINGS, cursor, labels[0].start("label")))
    for index, match in enumerate(labels):
        kind = _KIND_OF_LABEL[match.group("label").lower()]
        end = labels[index + 1].start("label") if index + 1 < len(labels) else len(text)
        out.append((kind, match.end(), end))
    return out


def _split(
    text: str, start: int, end: int, kind: str = FINDINGS
) -> list[tuple[int, int]]:
    """Sentence spans inside text[start:end]; blank stretches are dropped."""
    spans = []
    cursor = start
    chunk = text[start:end]
    breaks = list(_SENTENCE_END.finditer(chunk))
    if kind == CLINICAL:
        run_on = _CLINICAL_RUN_ON.search(chunk)
        if run_on and not any(b.start() < run_on.start() for b in breaks):
            breaks = [run_on] + breaks
            breaks.sort(key=lambda m: m.start())
    for match in breaks:
        spans.append((cursor, start + match.start()))
        cursor = start + match.end()
    spans.append((cursor, end))
    return [(s, e) for s, e in spans if text[s:e].strip()]
