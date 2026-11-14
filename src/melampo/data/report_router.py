"""The report router: which kind of report this is, before any interpreter reads it (16 October 2026).

Frank: an orchestrator must recognise the type of report first -- a language model that only says what kind of
report it is -- and then pass it to the deterministic interpreter (laboratory tables), to the reader (narrative
reports), to the imaging pipeline (reports with images).

Before this module the type was decided *after* reading, by counting the rows the laboratory grammar accepted: a
biopsy report with two immunohistochemistry rows came out "laboratory", and a radiology report with one value
in it was read by nothing else. Now:

1. **The model says the kind, and only the kind.** One question, a closed list of kinds (``KINDS``), several
   allowed (a discharge letter can carry a table of blood tests and a CT report), with the lines where each kind
   is. It never reads a value, never interprets: its answer chooses interpreters, nothing else. The model is a
   local one (the same Gemma 4 / Nemotron endpoint as the reader): patient reports never go to an external
   service (GDPR), and the text is pseudonymised before it arrives.
2. **The deterministic cues say the same thing in their own way**: the rows the laboratory grammar reads, the
   protein fractions, an antibiogram, the section headings of a pathology report (macroscopia, microscopia,
   immunoistochimica), the words of an imaging report (tecnica, reperto, conclusioni, TC, RM, ecografia), the
   images attached.
3. **Routing is the union, the disagreement is recorded.** Every interpreter that either side calls for is run:
   a deterministic interpreter that finds nothing costs nothing and deletes nothing, while an interpreter not
   run loses what only it would read. When the model and the cues disagree, or the model cannot answer, the
   decision says so (``needs_review``) and the cues alone route. The model is never the only gate.

Interpreters (``INTERPRETERS``): ``lab_table`` (``lab_results``: values, ranges, antibiograms, scores),
``electrophoresis_trace`` (the curve: today recorded as present and not read, see ``ELECTROPHORESIS_NOTE``),
``pathology`` (sections, immunohistochemistry rows through ``lab_table``, the diagnosis to the reader),
``narrative_reader`` (the reader and the linker), ``imaging`` (DICOM and images).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .lab_results import (
    DOCUMENT_TYPE_ANTIBIOGRAM,
    DOCUMENT_TYPE_LABORATORY,
    DOCUMENT_TYPE_PROTEIN_ELECTROPHORESIS,
    extract_lab_results,
)

LABORATORY = "laboratory"
ELECTROPHORESIS = "protein_electrophoresis"
MICROBIOLOGY = "microbiology"
PATHOLOGY = "pathology"
IMAGING_REPORT = "imaging_report"
CLINICAL_NOTE = "clinical_note"
OTHER = "other"

KINDS = {
    LABORATORY: "blood, urine or other laboratory tests: analytes with values, units and reference ranges",
    ELECTROPHORESIS: "serum or urine protein electrophoresis, immunofixation: fractions and a trace",
    MICROBIOLOGY: "cultures, isolated organisms, antibiogram (susceptibility S/I/R, MIC)",
    PATHOLOGY: "histology or cytology: biopsy or surgical specimen, macroscopy, microscopy, immunohistochemistry, "
               "grading, staging, margins, diagnosis",
    IMAGING_REPORT: "radiology, ultrasound, nuclear medicine or endoscopy report: technique, findings, conclusions",
    CLINICAL_NOTE: "discharge letter, visit, history, consultation: narrative about the patient",
    OTHER: "none of these (administrative, consent, prescription)",
}

INTERPRETERS_OF_KIND = {
    LABORATORY: ("lab_table",),
    ELECTROPHORESIS: ("lab_table", "electrophoresis_trace"),
    MICROBIOLOGY: ("lab_table",),
    PATHOLOGY: ("pathology", "lab_table", "narrative_reader"),
    IMAGING_REPORT: ("narrative_reader",),
    CLINICAL_NOTE: ("narrative_reader", "lab_table"),
    OTHER: (),
}
INTERPRETERS = ("lab_table", "electrophoresis_trace", "pathology", "narrative_reader", "imaging")

ELECTROPHORESIS_NOTE = (
    "electrophoresis trace present and not read: the fractions in the table are read, the shape of the curve is "
    "not (a small monoclonal band inside a normal gamma fraction shows only in the curve or in the laboratory's "
    "comment); the comment is read as text, the trace is sent to a person"
)

# Cues read from the text itself, in Italian and English.
_PATHOLOGY_HEADINGS = re.compile(
    r"\b(esame istologico|istologic\w*|citologic\w*|macroscopi\w*|microscopi\w*|immunoistochimic\w*|"
    r"immunohistochem\w*|histolog\w*|cytolog\w*|gross description|microscopic description|"
    r"anatomia patologica|surgical pathology|margini di resezione|resection margins?)\b",
    re.IGNORECASE,
)
_IMAGING_WORDS = re.compile(
    r"\b(tecnica|reperto|conclusioni|impression|findings|technique|"
    r"tc|tac|rm|rmn|rx|radiografi\w*|ecografi\w*|ecotomografi\w*|eco|ecocolordoppler|scintigrafi\w*|pet|"
    r"mammografi\w*|ct|mri|radiograph\w*|ultrasound|sonograph\w*|x-ray|mezzo di contrasto|mdc|contrast)\b",
    re.IGNORECASE,
)
_MICROBIOLOGY_WORDS = re.compile(
    r"\b(urinocoltura|emocoltura|coprocoltura|coltur\w*|tampone|isolat\w*|antibiogramma|mic|ufc|cfu|"
    r"culture|isolate\w*|susceptib\w*)\b",
    re.IGNORECASE,
)
_CLINICAL_WORDS = re.compile(
    r"\b(lettera di dimissione|dimissione|anamnesi|esame obiettivo|terapia|ricovero|visita|consulenza|"
    r"discharge|history of present illness|physical examination|admission)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Cues:
    numeric_rows: int = 0
    electrophoresis_fractions: bool = False
    antibiogram: bool = False
    pathology_headings: tuple[str, ...] = ()
    imaging_words: tuple[str, ...] = ()
    microbiology_words: tuple[str, ...] = ()
    clinical_words: tuple[str, ...] = ()
    images: int = 0

    def kinds(self) -> set[str]:
        """The kinds the cues alone point to."""
        out: set[str] = set()
        if self.numeric_rows >= 2:
            out.add(LABORATORY)
        if self.electrophoresis_fractions:
            out.add(ELECTROPHORESIS)
        if self.antibiogram or len(set(self.microbiology_words)) >= 2:
            out.add(MICROBIOLOGY)
        if len(set(self.pathology_headings)) >= 2:
            out.add(PATHOLOGY)
        if len(set(self.imaging_words)) >= 2:
            out.add(IMAGING_REPORT)
        if len(set(self.clinical_words)) >= 2:
            out.add(CLINICAL_NOTE)
        return out


def read_cues(text: str, images: int = 0) -> Cues:
    lab = extract_lab_results(text)
    numeric = sum(1 for r in lab.results if r.value is not None)

    def found(rx):
        return tuple(dict.fromkeys(m.group(0).lower() for m in rx.finditer(text)))

    return Cues(
        numeric_rows=numeric,
        electrophoresis_fractions=DOCUMENT_TYPE_PROTEIN_ELECTROPHORESIS in lab.document_types,
        antibiogram=DOCUMENT_TYPE_ANTIBIOGRAM in lab.document_types,
        pathology_headings=found(_PATHOLOGY_HEADINGS),
        imaging_words=found(_IMAGING_WORDS),
        microbiology_words=found(_MICROBIOLOGY_WORDS),
        clinical_words=found(_CLINICAL_WORDS),
        images=images,
    )


def kind_prompt(text: str, max_lines: int = 200) -> str:
    """The one question asked of the model: which kinds of report, and on which lines."""
    lines = text.splitlines()[:max_lines]
    numbered = "\n".join(f"{i + 1:>3}| {line}" for i, line in enumerate(lines))
    kinds = "\n".join(f"- {name}: {what}" for name, what in KINDS.items())
    return (
        "You sort clinical documents. Say only which kinds of report this document contains; do not read, "
        "interpret or repeat any value, and do not diagnose.\n\n"
        f"Kinds:\n{kinds}\n\n"
        "A document can contain more than one kind (a discharge letter with a table of blood tests and a CT "
        "report). For each kind present give the first and last line numbers where it is.\n"
        'Answer with JSON only: {"kinds": [{"kind": "<one of the kinds>", "from": <line>, "to": <line>}]}\n\n'
        f"Document (line numbers on the left):\n{numbered}"
    )


def parse_kind_answer(answer: str, n_lines: int) -> list[dict[str, Any]] | None:
    """The kinds in the model's answer, checked: known kinds, lines inside the document. None if unusable."""
    m = re.search(r"\{.*\}", answer or "", re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = []
    for item in data.get("kinds", []) if isinstance(data, dict) else []:
        if not isinstance(item, dict) or item.get("kind") not in KINDS:
            return None
        lo, hi = item.get("from", 1), item.get("to", n_lines)
        if not (isinstance(lo, int) and isinstance(hi, int) and 1 <= lo <= hi <= max(n_lines, 1)):
            lo, hi = 1, n_lines
        out.append({"kind": item["kind"], "from": lo, "to": hi})
    return out or None


@dataclass
class RoutingDecision:
    kinds: tuple[str, ...]
    interpreters: tuple[str, ...]
    model_kinds: tuple[str, ...] | None          # None: the model did not answer usably (or there is none)
    cue_kinds: tuple[str, ...]
    sections: tuple[dict[str, Any], ...] = ()     # the model's kinds with their lines
    agreement: str = ""                          # "agree", "model_only", "cues_only", "both_differ", "no_model"
    needs_review: bool = False
    notes: tuple[str, ...] = ()
    cues: Cues = field(default_factory=Cues)

    def as_dict(self) -> dict[str, Any]:
        return {"kinds": list(self.kinds), "interpreters": list(self.interpreters),
                "model_kinds": None if self.model_kinds is None else list(self.model_kinds),
                "cue_kinds": list(self.cue_kinds), "sections": list(self.sections), "agreement": self.agreement,
                "needs_review": self.needs_review, "notes": list(self.notes)}


def route(text: str, chat: Callable[[str], str] | list | tuple | None = None, images: int = 0) -> RoutingDecision:
    """Recognise the kinds of report, then choose the interpreters (the union of the models' and the cues').

    ``chat`` may be one model or several (9 November 2026: each model alone missed an interpreter on texts the
    other read right; an interpreter run in vain costs nothing, one not run loses what only it reads): the kinds
    any of them gives are joined."""
    cues = read_cues(text, images)
    cue_kinds = cues.kinds()
    model: list[dict[str, Any]] | None = None
    notes: list[str] = []
    chats = [] if chat is None else list(chat) if isinstance(chat, (list, tuple)) else [chat]
    if chats and text.strip():
        for one in chats:
            try:
                got = parse_kind_answer(one(kind_prompt(text)), len(text.splitlines()))
            except Exception as error:  # noqa: BLE001 -- an unreachable model must not stop the reading
                notes.append(f"router_model_error:{type(error).__name__}")
                continue
            if got is None:
                notes.append("router_model_answer_unusable")
            else:
                model = (model or []) + got
    model_kinds = None if model is None else {m["kind"] for m in model}
    kinds = set(cue_kinds) | (model_kinds or set())
    if model_kinds is None:
        agreement = "no_model"
    elif model_kinds == cue_kinds:
        agreement = "agree"
    elif not cue_kinds:
        agreement = "model_only"
    elif cue_kinds <= model_kinds:
        agreement = "model_adds"
    elif model_kinds <= cue_kinds:
        agreement = "cues_add"
    else:
        agreement = "both_differ"
    if text.strip() and not any(INTERPRETERS_OF_KIND[k] for k in kinds):
        # a text no interpreter would read (no kind, or "other": 9 November 2026, an EEG impression a model called
        # other) is read as narrative, never dropped
        kinds.add(CLINICAL_NOTE)
        notes.append("no_kind_recognised_read_as_narrative")
    interpreters = {i for k in kinds for i in INTERPRETERS_OF_KIND[k]}
    if images:
        interpreters.add("imaging")
    if ELECTROPHORESIS in kinds:
        notes.append(ELECTROPHORESIS_NOTE)
        if images:
            notes.append(f"electrophoresis_trace_images:{images}")
    needs_review = agreement in ("both_differ", "cues_add") or (bool(chats) and model_kinds is None)
    return RoutingDecision(
        kinds=tuple(sorted(kinds)), interpreters=tuple(i for i in INTERPRETERS if i in interpreters),
        model_kinds=None if model_kinds is None else tuple(sorted(model_kinds)), cue_kinds=tuple(sorted(cue_kinds)),
        sections=tuple(model or ()), agreement=agreement, needs_review=needs_review, notes=tuple(notes), cues=cues,
    )


LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


def router_chat_from_env(environ: dict[str, str] | None = None) -> Callable[[str], str] | list | None:
    """The local model the router asks, from the environment (8 November 2026), or None (the cues alone).

    On since 9 November 2026, when the reader's local server is configured: the router probe passed its criteria
    (270 public texts labelled first: an interpreter missed in 1.1% of the texts with Gemma 4, 0.4% with Nemotron 3
    Ultra, against 20.4% with the cues alone; no unusable answer). ``MELAMPO_REPORT_ROUTER=cues`` turns it off,
    ``=model`` asks for it and stops the start when the server is not given. The server is the reader's
    (``MELAMPO_LOCAL_CHAT_ENDPOINT``), the model ``MELAMPO_ROUTER_MODEL`` or the first of
    ``MELAMPO_LOCAL_CHAT_MODELS``. A host that is not this machine stops the start: a patient's report is never
    sent away (GDPR). Without ``MELAMPO_ROUTER_MODEL`` every local model is asked and their kinds are joined (9
    November 2026: together they miss less than either alone)."""
    import os

    env = os.environ if environ is None else environ
    asked = env.get("MELAMPO_REPORT_ROUTER", "").strip().lower()
    if asked == "cues":
        return None
    endpoint = env.get("MELAMPO_LOCAL_CHAT_ENDPOINT", "").strip()
    models = [m.strip() for m in env.get("MELAMPO_LOCAL_CHAT_MODELS", "").split(",") if m.strip()]
    model = env.get("MELAMPO_ROUTER_MODEL", "").strip()
    chosen = [model] if model else models
    if asked != "model" and (not endpoint or not chosen):
        return None                     # no local server: the cues alone, as before
    if not endpoint or not chosen:
        raise ValueError("MELAMPO_REPORT_ROUTER=model needs MELAMPO_LOCAL_CHAT_ENDPOINT and a model "
                         "(MELAMPO_ROUTER_MODEL or MELAMPO_LOCAL_CHAT_MODELS)")
    # the router asks only the kind, as it was measured: without reasoning ("+reasoning" is the readers' flag)
    chats = [local_chat(endpoint, m.partition("+")[0]) for m in chosen]
    return chats[0] if len(chats) == 1 else chats


REASONING_REQUEST = {"chat_template_kwargs": {"enable_thinking": True}}


def local_chat(endpoint: str, model: str, timeout: float = 60.0, max_tokens: int = 300,
               reasoning: bool = False, extra: dict | None = None) -> Callable[[str], str]:
    """A chat callable for an OpenAI-compatible server on this machine (vLLM, llama.cpp, Ollama). Any other
    host is refused: a patient report never leaves the machine without a legal basis (GDPR).

    ``reasoning`` (14 November 2026): the model reasons before it answers, as the readers were measured
    (ctrate8, ctrate9); the request asks it through the chat template (``REASONING_REQUEST``, the form vLLM and
    llama.cpp read), and ``extra`` (``MELAMPO_LOCAL_CHAT_EXTRA``, JSON) replaces it for a server that asks it
    otherwise. Only the answer is read, never the reasoning."""
    import urllib.parse
    import urllib.request

    host = urllib.parse.urlparse(endpoint).hostname or ""
    if host not in LOCAL_HOSTS:
        raise ValueError(f"the report router only talks to a local model, not {host!r}")
    url = endpoint.rstrip("/") + "/chat/completions"

    def chat(prompt: str) -> str:
        body = json.dumps({"model": model, "temperature": 0, "max_tokens": max_tokens,
                           "messages": [{"role": "user", "content": prompt}],
                           **(REASONING_REQUEST if reasoning and extra is None else {}),
                           **(extra if reasoning and extra else {})}).encode()
        req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 -- local host checked above
            return json.loads(r.read())["choices"][0]["message"]["content"]

    return chat


def lab_reading(text: str, decision: RoutingDecision):
    """The laboratory grammar, told whether the document is a laboratory report by the router."""
    kinds = set(decision.kinds)
    # a laboratory document through and through: every unread data line is residue. A document that also has
    # narrative (a biopsy, a CT report, a letter with a table) keeps its tables read and only unread table rows
    # as residue, never its sentences
    laboratory = bool({LABORATORY, ELECTROPHORESIS, MICROBIOLOGY} & kinds) and not (
        {PATHOLOGY, IMAGING_REPORT, CLINICAL_NOTE} & kinds)
    return extract_lab_results(text, laboratory=laboratory)


__all__ = ["CLINICAL_NOTE", "ELECTROPHORESIS", "ELECTROPHORESIS_NOTE", "IMAGING_REPORT", "INTERPRETERS",
           "INTERPRETERS_OF_KIND", "KINDS", "LABORATORY", "MICROBIOLOGY", "OTHER", "PATHOLOGY", "Cues",
           "RoutingDecision", "kind_prompt", "lab_reading", "local_chat", "parse_kind_answer", "read_cues", "route",
           "router_chat_from_env"]
