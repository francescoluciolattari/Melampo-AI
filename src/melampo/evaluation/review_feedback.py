"""What the physicians decide in the review queue, kept as training data for LoRA (14 October 2026).

Frank: training is the evolution of the system. When a report goes through the linker and a reading is sent
to a person (reader and model contradict each other, or one cannot say), the physician corrects or confirms
it; once a second person (the interpreting physician) confirms the decision, the item can teach the models.
Reports confirmed as a whole are a second source.

What is kept, and what is not:

* only pseudonymised text (``pseudonymised=True`` is required: the store refuses anything else); no name,
  no identifier of the patient; the report id is a pseudonymous key;
* the readings of the system (reader, model), the decision (site or not, the roles, the structure), who
  decided and who confirmed (pseudonymous ids), when, and the source (``review_queue`` or ``confirmed_report``);
* an item becomes training data only when it is confirmed by a second person (``confirmed_by`` set and
  different from ``decided_by``): a single correction is a proposal, not a label.

The training rows have the format of ``scripts/lora_roles.py`` (the role question, the roles as answer, five
folds grouped by report), so the same procedure trains Gemma 4 31B or Nemotron 3 Ultra. Under the MDR a
retrained model is a new version: it is validated on held-out gold before it replaces the one in use.
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


__all__ = ["FeedbackStore", "ReviewRecord", "SOURCES"]
