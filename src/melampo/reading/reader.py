"""One procedure of reading for every text (25 November 2026): read, compare, code afterwards.

``read_text`` is called with the same functions and the same interpreters for a patient's document and for a
publication. The context changes only what the text states about a finding (``card.STATEMENT``) and where the
models are: a patient's text goes only to the interpreters on this machine (``report_router.local_chat``, marked
``local_only``); a publication may go to OpenRouter.

1. every interpreter reads the text with the one directive (``prompt.reading_prompt``); what comes in another form
   is not read and is counted (``format``);
2. their cards are compared on the sentence and on the description (``agreement.agree``), never on a code;
3. the agreed cards are coded afterwards (``normalize.normalize``): HPO, the thesaurus, or none.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from . import card as c
from .agreement import agree
from .normalize import HpoIndex, normalize
from .prompt import reading_prompt
from .thesaurus import Thesaurus


def objects(answer: str) -> list[dict] | None:
    """The JSON list of an answer, or None when there is none."""
    text = re.sub(r"<think>.*?</think>", "", answer or "", flags=re.S)
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        got = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return [x for x in got if isinstance(x, dict)] if isinstance(got, list) else None


def read_text(sections: list[tuple[str, str]], context: str, chats: dict[str, Callable[[str], str]], *,
              subject: str = "", hpo: HpoIndex | None = None, thesaurus: Thesaurus | None = None,
              evidence: dict | None = None, answers: dict[str, str] | None = None) -> dict[str, Any]:
    """The reading of one text. ``chats``: interpreter name -> chat (prompt -> answer); for a patient's text every
    chat must be ``local_only``. ``answers``: the interpreters' answers to the reading already given (a cache: the
    same text is not asked again). Returns the agreed cards, each with its concept, and what was asked."""
    if context not in c.CONTEXTS:
        raise ValueError(f"context {context!r}")
    if context == "case":
        far = [name for name, chat in chats.items() if not getattr(chat, "local_only", False)]
        if far:
            raise ValueError(f"a patient's text is read only on this machine: {', '.join(far)} is not local")
    hpo = hpo or HpoIndex()
    thesaurus = thesaurus if thesaurus is not None else Thesaurus()
    prompt = reading_prompt(context, sections, subject)
    full = "\n".join(x for _, x in sections)
    readings: dict[str, list[dict]] = {}
    out: dict[str, Any] = {"context": context, "format": {}, "not_read": {}, "asked": [], "cards": []}
    for name, chat in chats.items():
        answer = (answers or {}).get(name)
        if answer is None:
            answer = chat(prompt)
        out.setdefault("answers", {})[name] = answer
        got = objects(answer)
        if got is None:
            out["not_read"][name] = "no_json_list"
            continue
        fmt = out["format"].setdefault(name, {"objects": len(got)})
        cards = []
        for raw in got:
            for problem in c.problems(raw, context):
                fmt[problem] = fmt.get(problem, 0) + 1
            card = c.read(raw, context)
            if not card["quote"] or _plain(card["quote"]) not in _plain(full):
                fmt["quote_not_in_text"] = fmt.get("quote_not_in_text", 0) + 1
                continue
            cards.append(card)
        readings[name] = cards
    if len(readings) < len(chats):
        out["status"] = "not_every_interpreter_answered"

    def ask(name: str, question: str) -> str:
        return chats[name](question)

    agreed, asked = agree(readings, context, ask, expected=list(chats))
    out["asked"] += asked
    readers = list(chats)
    for card in agreed:
        if "one_interpreter" in card["doubts"] or not card.get("specific", True):
            card["concept"] = {"concept": "", "code": "", "how": "not_coded:" + (
                "one_interpreter" if "one_interpreter" in card["doubts"] else "not_specific")}
        else:
            card["concept"] = normalize(card, context, hpo, thesaurus, ask, readers,
                                        {**(evidence or {}), "quote": card.get("quote", ""), "readers": readers},
                                        out["asked"])
        out["cards"].append(card)
    out.setdefault("status", "read")
    return out


def _plain(text: str) -> str:
    return re.sub(r"[^\w]+", "", str(text).lower())


__all__ = ["objects", "read_text"]
