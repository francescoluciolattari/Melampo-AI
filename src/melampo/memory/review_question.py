"""The question a person is asked, complete in itself (1 November 2026).

When the readers send a reading to a person, the record must hold everything the review mask needs to show,
so that the mask only displays it: the kind of question, the piece of the sentence and the structure, the
closed answers, and the readings of the system, to be shown **only after** the person has answered (a
suggestion seen first steers the reading: Dratsch et al., Radiology 2023; the same rule as the gold set guide).

The answers are the kinds of statement the readers themselves use, so that what the person decides can be
compared with them and, once a second person confirms it, teach them (``review_feedback``):

* the finding is of the structure, or the structure is involved (the finding extends to it, presses on it, is at
  the level it forms): the finding stays at the structure;
* the sentence says only a measure of it, a fact of its history, or that it is normal: it stays, as that;
* it is only a place of reference, or the words name another structure, or it is named only as a hypothesis:
  nothing stays at this structure (the other structure, when given, is recorded);
* the sentence does not let one say: the item stays open.

The questions are written in Italian for the physicians and in English for the record; the kind decides the
wording, never the answers offered.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from typing import Any

KEEP, ASIDE, OPEN = "keep", "aside", "open"

# code -> (Italian label, English label, outcome, the statement it records, needs a text)
STATEMENTS: dict[str, tuple[str, str, str, str, bool]] = {
    "finding": ("È un suo reperto", "It is a finding of it", KEEP, "finding", False),
    "involved": ("È coinvolta: il reperto si estende fino a lei, la comprime, o sta al livello che lei forma",
                 "It is involved: the finding extends to it, presses on it, or is at the level it forms",
                 KEEP, "involved", False),
    "measure": ("La frase dice solo una sua misura", "The sentence says only a measure of it", KEEP, "measure", False),
    "history": ("È un dato della sua storia (malattia o intervento passati)",
                "It is a fact of its history (past disease or surgery)", KEEP, "history", False),
    "normal": ("La frase dice che è normale", "The sentence says it is normal", KEEP, "normal", False),
    "reference": ("È solo un riferimento: il reperto è di…", "It is only a reference: the finding is of…",
                  ASIDE, "reference", True),
    "other_structure": ("Le parole nominano un'altra struttura:", "The words name another structure:",
                        ASIDE, "other_structure", True),
    "hypothesis": ("È nominata solo come ipotesi o diagnosi differenziale",
                   "It is named only as a hypothesis or in a differential diagnosis", ASIDE, "hypothesis", False),
    "cannot_say": ("Dalla frase non si può dire", "The sentence does not let one say", OPEN, "", False),
}

# kind -> (Italian question, English question); {s} the structure, {m} the words, {e} the exam, {b} what is described
KINDS: dict[str, tuple[str, str]] = {
    "outside_the_exam": ("Le parole «{m}» sono collegate a «{s}», che è fuori dal campo di vista dell'esame «{e}». "
                         "Nominano davvero «{s}»? Che cosa dice la frase di «{s}»?",
                         "The words \"{m}\" are linked to \"{s}\", outside the field of view of the exam \"{e}\". "
                         "Do they name \"{s}\"? What does the sentence say of \"{s}\"?"),
    "differential": ("Le parole «{m}» (collegate a «{s}») compaiono in una diagnosi differenziale. "
                     "La frase afferma qualcosa di «{s}»?",
                     "The words \"{m}\" (linked to \"{s}\") appear in a differential diagnosis. "
                     "Does the sentence state anything of \"{s}\"?"),
    "other_organ": ("La frase descrive «{b}». Che cosa dice della struttura «{s}» (parole «{m}»)?",
                    "The sentence describes \"{b}\". What does it say of \"{s}\" (words \"{m}\")?"),
    "reference_or_finding": ("Il reperto descritto riguarda la struttura «{s}» (parole «{m}»), "
                             "o «{s}» serve solo a dire dove si trova?",
                             "Is the finding of \"{s}\" (words \"{m}\"), or does \"{s}\" only say where it is?"),
    "what_does_it_say": ("Che cosa dice questa frase della struttura «{s}» (parole «{m}»)?",
                         "What does this sentence say of \"{s}\" (words \"{m}\")?"),
}


@dataclass
class Question:
    id: str
    kind: str
    report: str
    language: str
    sentence: str
    mention: str
    start: int                       # the mention's offset in the sentence
    structure: str                   # the linked class
    structure_name: str
    exam: str = ""
    section: str = ""
    question_it: str = ""
    question_en: str = ""
    options: list[dict[str, Any]] = field(default_factory=list)
    system: dict[str, Any] = field(default_factory=dict)      # shown only after the answer
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def for_the_person(self) -> dict[str, Any]:
        """What the mask shows before the answer: everything but the system's readings."""
        out = self.as_dict()
        out.pop("system")
        return out


def kind_of(evidence: str, models: dict[str, dict]) -> str:
    """The question to ask, from the reader's evidence and the models' answers (the first that applies)."""
    if "outside_the_exam" in evidence:
        return "outside_the_exam"
    if "named_only_in_a_differential" in evidence:
        return "differential"
    if "describes:" in evidence:
        return "other_organ"
    if "landmark" in evidence or any(a.get("linked") == "reference" or a.get("type") == "reference"
                                     for a in models.values()):
        return "reference_or_finding"
    return "what_does_it_say"


def build_question(ownership, *, report: str, language: str, sentence: str, mention: str, start: int,
                   structure: str, structure_name: str, context: dict | None = None) -> Question:
    """The question for one reading sent to a person (``finding_owner.Ownership``)."""
    context = context or {}
    evidence = ownership.reader or ""
    kind = kind_of(evidence, ownership.models or {})
    described = ownership.placed or next((e.split(":", 1)[1] for e in evidence.split(",") if e.startswith("describes:")), "")
    name_it = _class_name(structure, "it") or structure_name
    name_en = structure_name or _class_name(structure, "en")
    fill = {"m": mention, "e": context.get("exam", "") or "?", "b": described or "?"}
    it, en = KINDS[kind]
    key = f"{report}|{start}|{mention}|{structure}|{sentence}"
    return Question(
        id=hashlib.sha256(key.encode()).hexdigest()[:16], kind=kind, report=report, language=language,
        sentence=sentence, mention=mention, start=start, structure=structure, structure_name=structure_name,
        exam=context.get("exam", ""), section=context.get("section", ""),
        question_it=it.format(s=name_it, **fill), question_en=en.format(s=name_en, **fill),
        options=[{"code": code, "label_it": v[0], "label_en": v[1], "needs_text": v[4]} for code, v in STATEMENTS.items()],
        system={"verdict": ownership.verdict, "reason": ownership.reason, "reader": evidence,
                "models": ownership.models, "placed": ownership.placed,
                "statement": getattr(ownership, "statement", ""), "piece": getattr(ownership, "piece", "")},
    )


_NAMES: dict[str, dict[str, list[str]]] = {}


def _class_name(cid: str, language: str) -> str:
    """The class's name in the lexicon of the linker, in the language asked (the first, the usual one)."""
    if not _NAMES:
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parents[3] / "data" / "linking" / "anatomy_lexicon.json"
        try:
            _NAMES.update(json.loads(path.read_text("utf-8"))["classes"])
        except Exception:  # noqa: BLE001
            pass
    names = (_NAMES.get(cid) or {}).get(language) or []
    return names[0] if names else ""


def decide(question: Question | dict, code: str, text: str = "") -> dict[str, Any]:
    """What an answer means for the item: keep it at the structure (as which statement), set it aside (and of
    what it is), or leave it open."""
    if code not in STATEMENTS:
        raise ValueError(f"answer must be one of {', '.join(STATEMENTS)}")
    _, _, outcome, statement, needs_text = STATEMENTS[code]
    if needs_text and not text.strip():
        raise ValueError(f"the answer {code!r} needs the structure it names")
    q = question if isinstance(question, dict) else question.as_dict()
    return {"question": q["id"], "answer": code, "outcome": outcome, "statement": statement,
            "of": text.strip() if needs_text else "", "structure": q["structure"]}


# the roles the review record keeps for each answer (llm_reading.ROLES), so that the store can teach the readers
ROLES_OF_ANSWER = {
    "finding": ["structure"], "involved": ["structure", "landmark"], "measure": ["inherent_location"],
    "history": ["structure"], "normal": ["structure"], "reference": ["landmark"],
    "other_structure": ["structure"], "hypothesis": ["structure"], "cannot_say": ["uncertain"],
}


def review_record(question: Question | dict, code: str, decided_by: str, text: str = "",
                  pseudonymised: bool = False):
    """The answer as a ``review_feedback.ReviewRecord`` (a proposal until a second person confirms it)."""
    from ..evaluation.llm_reading import SITE_ROLES
    from ..evaluation.review_feedback import ReviewRecord

    q = question if isinstance(question, dict) else question.as_dict()
    decision = decide(q, code, text)
    roles = ROLES_OF_ANSWER[code]
    return ReviewRecord(report_id=q["report"], sentence=q["sentence"], mention=q["mention"], start=q["start"],
                        language=q["language"], decided_by=decided_by, roles=roles,
                        site=any(r in SITE_ROLES for r in roles),
                        structure=decision["of"] if code == "other_structure" else q["structure"],
                        system={**q.get("system", {}), "question": q["id"], "kind": q["kind"],
                                "answer": code, "statement": decision["statement"], "of": decision["of"]},
                        pseudonymised=pseudonymised)


__all__ = ["ASIDE", "KEEP", "KINDS", "OPEN", "STATEMENTS", "Question", "build_question", "decide", "kind_of",
           "review_record"]
