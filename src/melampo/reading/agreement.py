"""Where the interpreters agree, for every text (25 November 2026).

As two reviewers extract the data of a study, each on a common form, and compare it field by field (Cochrane,
MECIR C45-C46), the interpreters' cards are compared on the text and on their description, never on a code:

1. **The same sentence.** Two cards can describe the same finding only when their quotes are the same sentence,
   or one contains the other.
2. **The same finding.** Within one sentence, two cards with the same name and the same thing observed are the
   same finding. Otherwise every interpreter is asked, on its own and in the sentence, whether the two readings
   describe the same finding in other words ("elevated CRP", "elevated C-reactive protein"); they are paired only
   when every interpreter says so. A different threshold, site, side, time or thing is not the same finding.
3. **Field by field.** In a pair, each field both readings give alike is the agreed value; a field they give
   differently is empty, and named among the doubts (principle of doubt: kept, never resolved by the code).

A card no other interpreter paired is kept with the doubt ``one_interpreter``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from . import card as c

COMPARED = ("category", "specific", "interpretation", "value", "threshold", "side", "observed_by", "nature")
COMPARED_IN = {"case": ("state", "temporality", "subject"), "publication": ("direction", "compares")}
LABELS = ("words", "name", "what", "site", "time")
MAX_ASKED = 6     # cards of one interpreter in one sentence beyond which only cards sharing a word are asked


def letters(text: str) -> str:
    return re.sub(r"[^\w]+", " ", str(text).lower()).strip()


def _content(card: dict) -> set[str]:
    return {w for w in letters(f"{card.get('name', '')} {card.get('what', '')}").split() if len(w) > 2}


def same_sentence(a: str, b: str) -> bool:
    a, b = letters(a), letters(b)
    return bool(a and b) and (a in b or b in a)


def described(card: dict, context: str) -> dict:
    """What an interpreter is shown of a reading: its description and what it states, no measures."""
    keys = c.DESCRIPTION + tuple(k for k in c.STATEMENT[context] if k != "measures")
    return {k: card.get(k) for k in keys if card.get(k) not in (None, "", {}, [])}


def sameness_prompt(context: str, quote: str, a: dict, b: dict) -> str:
    where = "a document of one patient" if context == "case" else "a medical publication"
    return (f"Two readers described a finding in the same sentence of {where}.\n"
            f"Sentence: \"{quote}\"\n"
            f"Reading A: {json.dumps(described(a, context), ensure_ascii=False)}\n"
            f"Reading B: {json.dumps(described(b, context), ensure_ascii=False)}\n"
            "Do A and B describe the same finding in this sentence -- the same thing observed, at the same site, by "
            "the same method, with the same interpretation and threshold -- in other words? Other words for the same "
            "finding are the same (\"elevated CRP\" and \"elevated C-reactive protein\"); a different threshold, "
            "site, side, time or thing is not (an appendix wider than 6 mm and wider than 7 mm; the peak systolic "
            "velocity and the colour Doppler flow; vomiting before the pain and vomiting). "
            "Answer only JSON: {\"same\": true or false}")


def read_same(answer: str) -> bool | None:
    m = re.search(r"\"same\"\s*:\s*(true|false)", answer or "", re.I)
    return None if m is None else m.group(1).lower() == "true"


def _alike(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def merged(cards: list[tuple[str, dict]], context: str) -> dict:
    """One card from the readings of one finding: what all give alike; each other field empty and a doubt."""
    first = cards[0][1]
    out = {k: first.get(k) for k in LABELS + ("quote",)}
    doubts = []
    for k in COMPARED + COMPARED_IN[context]:
        values = [card.get(k) for _, card in cards]
        if all(_alike(v, values[0]) for v in values):
            out[k] = values[0]
        else:
            out[k] = None
            doubts.append(k)
    out["labels"] = {reader: card.get("name", "") for reader, card in cards}
    out["readings"] = {reader: card for reader, card in cards}
    out["readers"] = [reader for reader, _ in cards]
    out["doubts"] = doubts
    return out


def agree(readings: dict[str, list[dict]], context: str,
          ask: Callable[[str, str], str] | None = None) -> tuple[list[dict], list[dict]]:
    """The agreed cards and every question asked. ``readings``: each interpreter's cards (``card.read``), the
    first interpreter's order kept; ``ask(reader, prompt)``: the interpreter's answer (None: no question is asked,
    and cards with different names stay apart)."""
    readers = list(readings)
    used = {r: set() for r in readers}
    asked: list[dict] = []
    out: list[dict] = []
    if not readers:
        return out, asked
    anchor, others = readers[0], readers[1:]
    groups = [[(anchor, a)] for a in readings[anchor]]
    # first the same name and the same thing in the same sentence; only what is left is asked
    for other in others:
        for group in groups:
            a = group[0][1]
            j = next((j for j, b in enumerate(readings[other]) if j not in used[other]
                      and same_sentence(a.get("quote", ""), b.get("quote", ""))
                      and letters(b.get("name", "")) == letters(a.get("name", ""))
                      and letters(b.get("what", "")) == letters(a.get("what", ""))), None)
            if j is not None:
                used[other].add(j)
                group.append((other, readings[other][j]))
    for other in others:
        for group in groups:
            a = group[0][1]
            if ask is None or any(r == other for r, _ in group):
                continue
            pool = [(j, b) for j, b in enumerate(readings[other])
                    if j not in used[other] and same_sentence(a.get("quote", ""), b.get("quote", ""))]
            if len(pool) > MAX_ASKED:
                pool = [(j, b) for j, b in pool if _content(a) & _content(b)]
            yes = []
            for j, b in pool:
                quote = max((a.get("quote", ""), b.get("quote", "")), key=len)
                prompt = sameness_prompt(context, quote, a, b)
                answers = {r: read_same(ask(r, prompt)) for r in readers}
                asked.append({"quote": quote, "a": a.get("name"), "b": b.get("name"), "answers": answers})
                if all(v is True for v in answers.values()):
                    yes.append(j)
            if len(yes) == 1:       # two readings said the same as one: which is a doubt, none is paired
                used[other].add(yes[0])
                group.append((other, readings[other][yes[0]]))
    for group in groups:
        a = group[0][1]
        card = merged(group, context)
        card["paired_by"] = "one_interpreter" if len(group) < len(readers) else (
            "same_name" if all(letters(x.get("name", "")) == letters(a.get("name", "")) for _, x in group)
            else "same_in_context")
        if len(group) < len(readers):
            card["doubts"] = card["doubts"] + ["one_interpreter"]
        out.append(card)
    for other in others:
        for j, b in enumerate(readings[other]):
            if j not in used[other]:
                card = merged([(other, b)], context)
                card["paired_by"] = "one_interpreter"
                card["doubts"] = card["doubts"] + ["one_interpreter"]
                out.append(card)
    return out, asked


__all__ = ["agree", "described", "letters", "merged", "read_same", "same_sentence", "sameness_prompt"]
