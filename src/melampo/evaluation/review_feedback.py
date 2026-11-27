"""What the physicians confirm of a reading, kept to measure the readers and, under the plan of predetermined
changes, to train the reading skill (14 October 2026; on the cards of the one procedure since 27 November 2026).

Two kinds of record:

* ``ReadingConfirmation`` (the one procedure): a card a person was asked about in the review mask, because the two
  interpreters read a field differently, or only one read the finding, or the report's heading contradicted the
  part. What the physician answers is a proposal; it becomes a confirmation when a **second person** confirms it,
  and when the physician's answer is none of the interpreters' readings, only after a **third person** adjudicates
  it (human labels err too: Melampo's rule, after a published study where a second person found the first
  annotator wrong in 48% of the items every model contested). The sentence is kept **de-identified** -- written as
  it was, with what identifies the person replaced (``data.deidentify``) -- and enters training only after both
  interpreters read the de-identified sentence to the confirmed card (``check_deidentified``): de-identification
  must not teach an error. Each case falls in one part, fixed by its key: one in five **measures**, never trains;
  the others **train**.
* ``ReviewRecord`` (the reader before, the roles of a word): kept until that reader is removed after the bench of
  December 2026.

The training rows are the one directive (``reading.prompt``) on the de-identified sentence, and the confirmed card
as the answer: the same rows for every model (``docs/en/COMPLIANCE_RISK.md``, the plan of predetermined changes:
one archive, one procedure, one adapter per model derived from them). Nothing here trains a model.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from melampo.evaluation import llm_reading as lr

SOURCES = ("review_queue", "confirmed_report")
FOLDS = 5


@dataclass
class ReviewRecord:
    report_id: str              # pseudonymous key of the report
    sentence: str               # pseudonymised sentence
    mention: str
    start: int                  # offset of the mention in the sentence
    language: str
    decided_by: str             # pseudonymous id of the physician who corrected or confirmed
    roles: list[str]            # the roles the physician decided (one or more, from llm_reading.ROLES)
    site: bool                  # the word names a body site here
    structure: str = ""         # the class linked, when a site
    confirmed_by: str = ""      # pseudonymous id of the second person who confirmed
    system: dict = field(default_factory=dict)   # what the reader and the model said
    source: str = "review_queue"
    pseudonymised: bool = False
    note: str = ""
    at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def problems(self) -> list[str]:
        out = []
        if not self.pseudonymised:
            out.append("the text is not marked as pseudonymised")
        if self.source not in SOURCES:
            out.append(f"source must be one of {SOURCES}")
        if not self.roles or any(r not in lr.ROLES for r in self.roles):
            out.append(f"roles must be one or more of {', '.join(lr.ROLES)}")
        if self.sentence[self.start:self.start + len(self.mention)] != self.mention:
            out.append("the mention is not at its offset in the sentence")
        if self.site != any(r in lr.SITE_ROLES for r in self.roles):
            out.append("site and roles disagree")
        return out

    @property
    def confirmed(self) -> bool:
        return bool(self.confirmed_by) and self.confirmed_by != self.decided_by


class FeedbackStore:
    """An append-only JSONL file of review records."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def add(self, record: ReviewRecord) -> None:
        problems = record.problems()
        if problems:
            raise ValueError("; ".join(problems))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def records(self) -> list[ReviewRecord]:
        if not self.path.exists():
            return []
        return [ReviewRecord(**json.loads(x)) for x in self.path.read_text("utf-8").splitlines() if x.strip()]

    def training_rows(self) -> list[dict]:
        """Confirmed records as LoRA rows: the role question and the roles as answer, folds by report."""
        rows = []
        for r in self.records():
            if not r.confirmed:
                continue
            end = r.start + len(r.mention)
            question = lr.role_prompt(r.sentence, r.start, end, language={"it": "Italian", "en": "English"}.get(r.language, ""))
            fold = int(hashlib.sha256(r.report_id.encode()).hexdigest(), 16) % FOLDS
            rows.append({
                "item_id": hashlib.sha256(f"{r.report_id}|{r.start}|{r.mention}|{r.at}".encode()).hexdigest()[:16],
                "report_id": r.report_id, "language": r.language, "mention": r.mention, "sentence": r.sentence,
                "role": r.roles[0], "roles": r.roles, "fold": fold, "source": r.source,
                "prompt": [{"role": "user", "content": question}],
                "completion": [{"role": "assistant", "content": json.dumps({"role": r.roles[0], "roles": r.roles})}],
            })
        return rows


# -- the one procedure: the confirmations of the cards ---------------------------------------------------------

MEASURE_SHARE = 5          # one case in five measures and never trains
KINDS = ("field", "one_interpreter", "part")


def person_key(name: str) -> str:
    """A pseudonymous key of a person who answered (the name stays in the case, on this computer)."""
    return "p-" + hashlib.sha256(f"melampo-person|{(name or '').strip().lower()}".encode()).hexdigest()[:12]


def case_key(case_id: str) -> str:
    return "c-" + hashlib.sha256(f"melampo-case|{case_id}".encode()).hexdigest()[:12]


def part_of(case: str) -> str:
    """The part a case falls in, fixed by its key: "measure" (one in five) or "train"."""
    return "measure" if int(hashlib.sha256(case.encode()).hexdigest(), 16) % MEASURE_SHARE == 0 else "train"


@dataclass
class ReadingConfirmation:
    case: str                   # pseudonymous key of the case (``case_key``)
    document: str               # the document's id (``case_attachments.document_id``) or "note"
    sentence: str               # the de-identified sentence
    card: dict                  # the card as confirmed (the one form; its quote is the de-identified sentence)
    doubt: str                  # what was in doubt: a field, "one_interpreter", "part_differs_from_heading"
    answer: str                 # the closed answer
    system: dict                # what each interpreter read of what was in doubt
    decided_by: str             # pseudonymous keys (``person_key``)
    confirmed_by: str = ""
    adjudicated_by: str = ""
    needs_adjudication: bool = False   # the answer is none of the interpreters' readings
    deidentified: bool = False
    deidentified_kinds: dict = field(default_factory=dict)
    deidentified_checked: bool | None = None   # both interpreters read the de-identified sentence to the card
    part: str = ""
    at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def __post_init__(self) -> None:
        self.part = self.part or part_of(self.case)

    def problems(self) -> list[str]:
        out = []
        if not self.deidentified:
            out.append("the sentence is not de-identified")
        if self.card.get("quote") != self.sentence:
            out.append("the card's quote is not the de-identified sentence")
        if self.confirmed_by and self.confirmed_by == self.decided_by:
            out.append("the second person is the first")
        if self.adjudicated_by and self.adjudicated_by in (self.decided_by, self.confirmed_by):
            out.append("the third person is the first or the second")
        return out

    @property
    def confirmed(self) -> bool:
        if not self.confirmed_by or self.confirmed_by == self.decided_by:
            return False
        return not self.needs_adjudication or (bool(self.adjudicated_by)
                                                and self.adjudicated_by not in (self.decided_by, self.confirmed_by))

    @property
    def trains(self) -> bool:
        return self.confirmed and self.deidentified_checked is True and self.part == "train"


def _compared(card: dict) -> dict:
    from melampo.reading.agreement import COMPARED, COMPARED_IN

    return {k: card.get(k) for k in COMPARED + COMPARED_IN["case"]}


def check_deidentified(record: ReadingConfirmation, read) -> bool:
    """Both interpreters read the de-identified sentence (``read(sentence)`` -> the agreed cards of the one
    procedure); the record passes when one agreed card has every compared field of the confirmed card."""
    wanted = json.dumps(_compared(record.card), sort_keys=True)
    record.deidentified_checked = any(json.dumps(_compared(c), sort_keys=True) == wanted
                                      for c in read(record.sentence) if "one_interpreter" not in c.get("doubts", ()))
    return record.deidentified_checked


class ConfirmationStore:
    """An append-only JSONL file of reading confirmations, on this computer (never committed)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def add(self, record: ReadingConfirmation) -> None:
        problems = record.problems()
        if problems:
            raise ValueError("; ".join(problems))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def records(self) -> list[ReadingConfirmation]:
        if not self.path.exists():
            return []
        return [ReadingConfirmation(**json.loads(x)) for x in self.path.read_text("utf-8").splitlines() if x.strip()]

    def rows(self, part: str = "train") -> list[dict]:
        """The rows of one part: the one directive on the de-identified sentence, the confirmed card as answer.
        "train": only confirmed records that passed the de-identification check; "measure": every confirmed one."""
        from melampo.reading import card as c
        from melampo.reading.prompt import reading_prompt

        out = []
        for r in self.records():
            if r.part != part or not r.confirmed or (part == "train" and not r.trains):
                continue
            answer = {k: r.card.get(k) for k in c.DESCRIPTION + c.STATEMENT["case"] if k != "measures"}
            answer["quote"] = r.sentence
            out.append({"case": r.case, "document": r.document, "doubt": r.doubt, "part": r.part,
                        "prompt": [{"role": "user", "content": reading_prompt("case", [("", r.sentence)])}],
                        "completion": [{"role": "assistant", "content": json.dumps([answer], ensure_ascii=False)}]})
        return out


__all__ = [
    "KINDS",
    "SOURCES",
    "ConfirmationStore",
    "FeedbackStore",
    "ReadingConfirmation",
    "ReviewRecord",
    "case_key",
    "check_deidentified",
    "part_of",
    "person_key",
]
