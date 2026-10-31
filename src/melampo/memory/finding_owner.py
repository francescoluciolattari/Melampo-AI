"""Of what is the finding said? The reader and the model read the sentence, each on its own (31 October 2026).

A report names a structure to describe it ("the liver is enlarged") or to place something else ("enlarged nodes
along the aortic arch", "fluid between the loops of the small intestine"). Before a finding is attached to the
structure the linker found, two readers say whether the sentence describes that structure:

* the Kintsch reader (``ci_reader``): its second stage weighs the lexical evidence that the word is a body
  structure against the relation the sentence puts it in (``spatial_relation``) and reads ``structure`` or
  ``landmark``;
* the model, with the reader's method (``llm_reading.method_prompt``: phrase, governor, predicate, readings,
  roles, and now what a landmark places), never shown the reader's answer, so that their agreement keeps its
  value. Two models are one stream: they must agree with each other first.

The decision is on the outcome, as in option b (14 October 2026): the finding is of this structure, or it is not.
Both say it is: the finding stays. Both say it is not: no finding at this site; the clause goes to a person with
what it places. They contradict each other, or one cannot say: the finding stays and goes to a person. Without a
model the reader alone never removes a finding: what it reads as a landmark goes to a person.

Patients' reports are read only by local models (``report_router.local_chat``); the probes use public reports.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

HERE, ELSEWHERE, UNSURE = "here", "elsewhere", "unsure"


@dataclass(frozen=True)
class Ownership:
    verdict: str                                   # HERE, ELSEWHERE, UNSURE
    reader: str = ""                               # the reader's reading ("structure", "landmark", ...)
    models: dict[str, list[str]] = field(default_factory=dict)   # the roles each model gave
    placed: str = ""                               # what the structure places, when it is a landmark
    reason: str = ""

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "reader": self.reader, "models": self.models, "placed": self.placed,
                "reason": self.reason}


def _verdict_of_roles(roles: list[str]) -> str:
    if not roles or roles[0] == "uncertain":
        return UNSURE
    if "landmark" in roles and "structure" not in roles:
        return ELSEWHERE
    return HERE


class FindingOwner:
    """``reader``: a ``CIReader`` (its ``read`` is used); ``chats``: model name -> chat function (local)."""

    def __init__(self, reader, chats: dict[str, Callable[[str], str]] | None = None):
        self.reader = reader
        self.chats = dict(chats or {})

    def _reader(self, mention: str, clause: str, at: int, language: str) -> tuple[str, str]:
        if self.reader is None:
            return UNSURE, ""
        got = self.reader.read(mention, clause, at, language=language)
        if got.decision == "landmark":
            return ELSEWHERE, got.decision
        if got.decision == "underspecified" and {got.top, sorted(got.shares, key=got.shares.get)[-2]} == \
                {"structure", "landmark"}:
            return UNSURE, got.decision
        return HERE, got.decision if got.decision != "underspecified" else got.top

    def _models(self, mention: str, clause: str, at: int, language: str) -> tuple[str, dict, str]:
        from ..evaluation.llm_reading import ask, method_prompt

        name = {"it": "Italian", "en": "English"}.get(language, "")
        roles: dict[str, list[str]] = {}
        placed = ""
        for model, chat in self.chats.items():
            try:
                got = ask(chat, method_prompt(clause, at, at + len(mention), language=name), clause, at,
                          at + len(mention))
                answer = got.final.answer if got.final.valid else {}
            except Exception:  # noqa: BLE001 -- a lost answer is "cannot say", never a verdict
                answer = {}
            roles[model] = list(answer.get("roles") or [])
            placed = placed or str(answer.get("placed") or "")
        verdicts = {_verdict_of_roles(r) for r in roles.values()}
        verdict = verdicts.pop() if len(verdicts) == 1 else UNSURE     # two models are one stream
        return verdict, roles, placed

    def __call__(self, mention: str, clause: str, at: int, language: str = "en") -> Ownership:
        from .spatial_relation import read_relation

        reader, reading = self._reader(mention, clause, at, language)
        relation_placed = read_relation(clause, at, language).placed
        if not self.chats:
            if reader == ELSEWHERE:
                return Ownership(UNSURE, reading, {}, relation_placed, "the_reader_alone_reads_a_landmark")
            if reader == UNSURE:
                return Ownership(UNSURE, reading, {}, relation_placed, "the_reader_cannot_say")
            return Ownership(HERE, reading)
        model, roles, placed = self._models(mention, clause, at, language)
        if reader == model and reader != UNSURE:
            return Ownership(reader, reading, roles, placed or relation_placed if reader == ELSEWHERE else "",
                             "reader_and_model_agree")
        why = "reader_and_model_disagree" if UNSURE not in (reader, model) else "one_reader_cannot_say"
        return Ownership(UNSURE, reading, roles, placed or relation_placed, why)


__all__ = ["ELSEWHERE", "HERE", "UNSURE", "FindingOwner", "Ownership"]
