"""Where the interpreters agree, for every text (25 November 2026).

As two reviewers extract the data of a study, each on a common form, and compare it field by field (Cochrane,
MECIR C45-C46), the interpreters' cards are compared on the text and on their description, never on a code:

1. **The same name in the same sentence.** Two cards with the same name and the same thing observed, whose quotes
   are the same sentence (or one contains the other), are the same finding.
2. **What one found and the other did not name so: each looks at the other's** (25 November 2026, declared after
   the sixth run). As a reviewer looks again at what only the other extracted, each interpreter is shown, once
   for the whole text, the findings only the other named, each with its sentence and its name (never the other's
   fields), and its own findings in those sentences not yet paired. For each it answers: the same finding as one
   of its own (in other words: "elevated CRP" and "elevated C-reactive protein"), or its own card for it, or that
   the sentence does not state it. A pair by "the same" is made only when both say it of each other; a card given
   for the other's finding is a second, independent reading of it. A different threshold, site, side, time or
   thing is not the same finding.
3. **Field by field.** In a pair, each field both readings give alike is the agreed value; a field they give
   differently is empty, and named among the doubts (principle of doubt: kept, never resolved by the code).

A card the other interpreter did not read is kept with the doubt ``one_interpreter`` (and, when the other said
the sentence does not state it, ``not_stated_for_the_other``).

In the sixth run, asking each pair of cards on its own took 722 questions for 13 texts, and still left three
readings in four to one interpreter: Muse had not written 160 of Nemotron's findings at all, and no question
between two cards could pair what one of them never wrote.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from . import card as c

COMPARED = ("category", "specific", "interpretation", "value", "threshold", "side", "observed_by", "nature")
# the fields every card must give (26 November 2026, declared after the seventh run: in 270 second readings one
# interpreter gave its card without a category, and "not given" was taken for "read otherwise"): when one reading
# does not give one and the other does, the field is the other's, with the doubt "<field>_given_by_one"; a field
# both give differently stays empty, a doubt. The optional fields (value, threshold, side, the direction, ...) are
# readings when empty: "no threshold" is what the reader read.
REQUIRED = ("category", "specific", "observed_by", "nature")
COMPARED_IN = {"case": ("state", "temporality", "subject"), "publication": ("direction", "compares")}
LABELS = ("words", "name", "what", "site", "time")


def letters(text: str) -> str:
    return re.sub(r"[^\w]+", " ", str(text).lower()).strip()


def same_sentence(a: str, b: str) -> bool:
    a, b = letters(a), letters(b)
    return bool(a and b) and (a in b or b in a)


def described(card: dict, context: str) -> dict:
    """What an interpreter is shown of a reading: its description and what it states, no measures."""
    keys = c.DESCRIPTION + tuple(k for k in c.STATEMENT[context] if k != "measures")
    return {k: card.get(k) for k in keys if card.get(k) not in (None, "", {}, [])}


def _alike(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def merged(cards: list[tuple[str, dict]], context: str) -> dict:
    """One card from the readings of one finding: what all give alike; each other field empty and a doubt."""
    first = cards[0][1]
    out = {k: first.get(k) for k in LABELS + ("quote",)}
    doubts = []
    for k in COMPARED + COMPARED_IN[context]:
        values = [card.get(k) for _, card in cards]
        given = [v for v in values if v not in (None, "")]
        if all(_alike(v, values[0]) for v in values):
            out[k] = values[0]
        elif k in REQUIRED and given and all(_alike(v, given[0]) for v in given):
            out[k] = given[0]
            doubts.append(f"{k}_given_by_one")
        else:
            out[k] = None
            doubts.append(k)
    out["labels"] = {reader: card.get("name", "") for reader, card in cards}
    out["readings"] = {reader: card for reader, card in cards}
    out["readers"] = [reader for reader, _ in cards]
    out["doubts"] = doubts
    return out


def crosscheck_prompt(context: str, theirs: list[dict], mine: list[dict]) -> str:
    """What one interpreter is shown: the findings only the other named (name and sentence), and its own not yet
    paired in those sentences (name and sentence), lettered."""
    where = "a document of one patient" if context == "case" else "a medical publication"
    first = "\n".join(f"{i}. \"{x.get('quote', '')}\" -- {x.get('name', '')}" for i, x in enumerate(theirs, 1))
    own = "\n".join(f"{_letter(j)}. {x.get('name', '')} -- \"{x.get('quote', '')}\""
                     for j, x in enumerate(mine)) or "(none)"
    return (f"You and another reader read the same {where}, each on your own. The other reader named the findings "
            "below, which you did not name the same way; each with the sentence it is in:\n" + first + "\n\n"
            "Your findings in those sentences that are not yet paired:\n" + own + "\n\n"
            "For each numbered finding answer one of: the letter of your finding when it is the same finding -- the "
            "same thing observed, at the same site, by the same method, with the same interpretation and threshold, "
            "in other words (\"elevated CRP\" and \"elevated C-reactive protein\"; a different threshold, site, side, "
            "time or thing is not the same: an appendix wider than 6 mm and wider than 7 mm; vomiting before the "
            "pain and vomiting); otherwise your own card for it, read from its sentence, complete -- every field of "
            "this form, the category too:\n"
            + c.example(context) + "\n"
            "or null when the sentence does not state such a finding. "
            "Answer only JSON: [{\"item\": 1, \"same_as\": \"a\" or null, \"card\": {...} or null}, ...]")


def _letter(j: int) -> str:
    out = ""
    j += 1
    while j:
        j, r = divmod(j - 1, 26)
        out = chr(97 + r) + out
    return out


def read_crosscheck(answer: str, n: int, m: int, context: str) -> dict[int, tuple[str, Any]]:
    """item -> ("same", j) | ("card", card) | ("not_stated", None); an item not answered is absent."""
    text = re.sub(r"<think>.*?</think>", "", answer or "", flags=re.S)
    found = re.search(r"\[.*\]", text, re.S)
    try:
        rows = json.loads(found.group(0)) if found else []
    except json.JSONDecodeError:
        rows = []
    letters_ = {_letter(j): j for j in range(m)}
    out: dict[int, tuple[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        try:
            i = int(row.get("item")) - 1
        except (TypeError, ValueError):
            continue
        if not 0 <= i < n or i in out:
            continue
        same = str(row.get("same_as") or "").strip().lower()
        if same in letters_:
            out[i] = ("same", letters_[same])
        elif isinstance(row.get("card"), dict):
            out[i] = ("card", c.read(row["card"], context))
        else:
            out[i] = ("not_stated", None)
    return out


def _named_alike(a: dict, b: dict) -> bool:
    return letters(b.get("name", "")) == letters(a.get("name", "")) and \
        letters(b.get("what", "")) == letters(a.get("what", ""))


def agree(readings: dict[str, list[dict]], context: str,
          ask: Callable[[str, str], str] | None = None,
          expected: list[str] | None = None) -> tuple[list[dict], list[dict]]:
    """The agreed cards and every question asked. ``readings``: each interpreter's cards (``card.read``);
    ``ask(reader, prompt)``: the interpreter's answer (None: no question is asked, and cards with different names
    stay apart); ``expected``: every interpreter that should have read the text (25 November 2026, declared after
    the fifth run: when one did not answer, the other's cards counted as agreed with themselves). When an
    expected interpreter gave no reading, nothing is agreed: every card is a card of one interpreter, and no
    question is asked. Two interpreters (the pair of the readers)."""
    def alone(reader: str, b: dict, why: str = "") -> dict:
        card = merged([(reader, b)], context)
        card["paired_by"] = "one_interpreter"
        card["doubts"] = card["doubts"] + ["one_interpreter"] + ([why] if why else [])
        return card

    if expected and any(r not in readings for r in expected):
        return [alone(r, b) for r, cards in readings.items() for b in cards], []
    readers = list(readings)
    if len(readers) == 1:
        return [alone(readers[0], b) for b in readings[readers[0]]], []
    if len(readers) != 2:
        raise ValueError("the comparison is between two interpreters")
    one, two = readers
    A, B = readings[one], readings[two]
    pairs: list[tuple[int, int, str]] = []
    used_a, used_b = set(), set()
    for i, a in enumerate(A):                    # 1. the same name in the same sentence
        j = next((j for j, b in enumerate(B) if j not in used_b and _named_alike(a, b)
                  and same_sentence(a.get("quote", ""), b.get("quote", ""))), None)
        if j is not None:
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j, "same_name"))
    left_a = [i for i in range(len(A)) if i not in used_a]
    left_b = [j for j in range(len(B)) if j not in used_b]
    asked: list[dict] = []
    out: list[dict] = []
    second: dict[tuple[str, int], dict] = {}     # an independent second reading of the other's finding
    not_stated: set[tuple[str, int]] = set()
    if ask is not None and (left_a or left_b):
        def near(cards: list[int], own: list[dict], theirs: list[dict], others: list[int]) -> list[int]:
            return [k for k in cards if any(same_sentence(own[k].get("quote", ""), theirs[o].get("quote", ""))
                                            for o in others)]
        mine_b = near(left_b, B, A, left_a)      # B's unpaired cards in the sentences of A's unpaired ones
        mine_a = near(left_a, A, B, left_b)
        answer_b = read_crosscheck(ask(two, crosscheck_prompt(context, [A[i] for i in left_a],
                                                              [B[j] for j in mine_b])) if left_a else "[]",
                                   len(left_a), len(mine_b), context)
        answer_a = read_crosscheck(ask(one, crosscheck_prompt(context, [B[j] for j in left_b],
                                                              [A[i] for i in mine_a])) if left_b else "[]",
                                   len(left_b), len(mine_a), context)
        asked.append({"reader": two, "items": len(left_a), "answers": len(answer_b)})
        asked.append({"reader": one, "items": len(left_b), "answers": len(answer_a)})
        said_b = {left_a[k]: (kind, mine_b[v] if kind == "same" else v) for k, (kind, v) in answer_b.items()}
        said_a = {left_b[k]: (kind, mine_a[v] if kind == "same" else v) for k, (kind, v) in answer_a.items()}
        for i in left_a:                         # mutual "the same": a pair
            kind, j = said_b.get(i, ("", None))
            if kind == "same" and j not in used_b and said_a.get(j) == ("same", i):
                used_a.add(i)
                used_b.add(j)
                pairs.append((i, j, "same_in_context"))
        for i in left_a:
            if i in used_a:
                continue
            kind, v = said_b.get(i, ("", None))
            if kind == "card" and same_sentence(v.get("quote", "") or A[i].get("quote", ""), A[i].get("quote", "")):
                second[(one, i)] = {**v, "quote": v.get("quote") or A[i].get("quote", "")}
            elif kind == "not_stated":
                not_stated.add((one, i))
        for j in left_b:
            if j in used_b:
                continue
            kind, v = said_a.get(j, ("", None))
            if kind == "card" and same_sentence(v.get("quote", "") or B[j].get("quote", ""), B[j].get("quote", "")):
                second[(two, j)] = {**v, "quote": v.get("quote") or B[j].get("quote", "")}
            elif kind == "not_stated":
                not_stated.add((two, j))
    at: dict[tuple[str, int], int] = {}          # each card read -> the output card it is in
    for i, j, how in pairs:
        card = merged([(one, A[i]), (two, B[j])], context)
        card["paired_by"] = how
        at[(one, i)] = at[(two, j)] = len(out)
        out.append(card)
    one_way: list[tuple[str, int, tuple[str, int]]] = []
    for who, cards, used, said, other in ((one, A, used_a, said_b if ask is not None and (left_a or left_b) else {},
                                           two),
                                          (two, B, used_b, said_a if ask is not None and (left_a or left_b) else {},
                                           one)):
        for k in range(len(cards)):
            if k in used:
                continue
            if (who, k) in second:
                card = merged([(who, cards[k]), (other, second[(who, k)])], context)
                card["paired_by"] = "read_again_by_the_other"
                at[(who, k)] = len(out)
                out.append(card)
            elif said.get(k, ("", None))[0] == "same":
                one_way.append((who, k, (other, said[k][1])))
            else:
                at[(who, k)] = len(out)
                out.append(alone(who, cards[k], "not_stated_for_the_other" if (who, k) in not_stated else ""))
    # the other said "the same as mine" and did not hear it back: when its card is in a reading of two, this one
    # is another name of it (kept in ``also_named``, not a second finding); otherwise it stays one interpreter's
    for who, k, partner in one_way:
        cards = A if who == one else B
        if partner in at and out[at[partner]]["paired_by"] != "one_interpreter":
            target = out[at[partner]]
            target.setdefault("also_named", []).append({"reader": who, "name": cards[k].get("name", ""),
                                                        "quote": cards[k].get("quote", "")})
        else:
            out.append(alone(who, cards[k]))
    return out, asked


__all__ = ["agree", "crosscheck_prompt", "described", "letters", "merged", "read_crosscheck", "same_sentence"]
