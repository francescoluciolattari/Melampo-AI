"""The code of an agreed reading, given afterwards (25 November 2026).

Only the reading the interpreters agree on is coded, and coding never decides whether they agree (two coders give
the same code to three findings in four: codes that differ often mean the same thing). In order:

1. the thesaurus already holds this label with this description: its concept;
2. the name is HPO's own name or exact synonym of one term: that term;
3. otherwise the candidates sharing content words (the thesaurus's concepts and HPO's terms) are shown to every
   interpreter, on its own, with the sentence: the reading is that candidate only when every interpreter chooses
   the same one (entity linking: candidates first, then the decision in context);
4. otherwise, from a publication, a new concept of the thesaurus; from a patient's document, no concept (the
   finding keeps its words).

A publication's agreed choice becomes a label of the thesaurus, with its sentence and the answers.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from .agreement import described, letters
from .thesaurus import Concept, Thesaurus

CANDIDATES = 8


class HpoIndex:
    """HPO's terms by their content words (names and exact synonyms), for proposing candidates."""

    def __init__(self, terms: Any = None):
        self.terms = terms
        self.words: dict[str, set[str]] = {}
        if terms is None:
            return
        for term, name in terms.names.items():
            if not term.startswith("HP:"):
                continue
            for label in [name] + list(terms.synonyms.get(term, [])):
                for w in letters(label).split():
                    if len(w) > 2:
                        self.words.setdefault(w, set()).add(term)

    def find(self, name: str) -> str:
        return self.terms.find(name, prefix="HP:") if self.terms is not None and name else ""

    def candidates(self, card: dict, limit: int = CANDIDATES) -> list[tuple[str, str]]:
        if self.terms is None:
            return []
        words = {w for w in letters(f"{card.get('name', '')} {card.get('what', '')}").split() if len(w) > 2}
        count: dict[str, int] = {}
        for w in words:
            for term in self.words.get(w, ()):
                count[term] = count.get(term, 0) + 1
        def score(term: str) -> float:
            theirs = {w for w in letters(self.terms.names[term]).split() if len(w) > 2}
            return count[term] / len(words | theirs) if theirs else 0.0
        best = sorted(count, key=lambda t: (-score(t), t))[:limit]
        return [(t, self.terms.names[t]) for t in best]


def choice_prompt(context: str, items: list[tuple[dict, list[str]]]) -> str:
    """One question for a whole text: every agreed finding still without a code, with its sentence and its
    candidates (25 November 2026, declared after the sixth run: one question per finding took 148 questions)."""
    where = "a document of one patient" if context == "case" else "a medical publication"
    blocks = []
    for i, (card, options) in enumerate(items, 1):
        lines = "\n".join(f"   {k}. {o}" for k, o in enumerate(options, 1))
        blocks.append(f"{i}. Sentence: \"{card.get('quote', '')}\"\n   Finding: "
                      f"{json.dumps(described(card, context), ensure_ascii=False)}\n{lines}")
    return (f"Findings agreed by two readers in sentences of {where}. For each, which of its candidates is the same "
            "finding, the same thing with the same meaning (not a broader, narrower or related one)?\n\n"
            + "\n\n".join(blocks) + "\n\n"
            "Answer only JSON: [{\"item\": 1, \"same\": the number of the candidate, or null when none is}, ...]")


def read_choices(answer: str, sizes: list[int]) -> dict[int, int | None]:
    text = re.sub(r"<think>.*?</think>", "", answer or "", flags=re.S)
    m = re.search(r"\[.*\]", text, re.S)
    try:
        rows = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        rows = []
    out: dict[int, int | None] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        try:
            i = int(row.get("item")) - 1
        except (TypeError, ValueError):
            continue
        if not 0 <= i < len(sizes) or i in out:
            continue
        k = row.get("same")
        out[i] = int(k) if isinstance(k, int) and 1 <= k <= sizes[i] else None
    return out


def _add(thesaurus: Thesaurus, concept: Concept, card: dict, evidence: dict) -> None:
    """Every interpreter's label of the agreed reading becomes a label of the concept."""
    names = [card.get("name", "")] + list((card.get("labels") or {}).values()) + [
        x.get("name", "") for x in card.get("also_named") or []]
    for name in dict.fromkeys(names):
        if name:
            thesaurus.add(concept, {**card, "name": name}, evidence)


def normalize_all(cards: list[dict], context: str, hpo: HpoIndex, thesaurus: Thesaurus,
                  ask: Callable[[str, str], str] | None, readers: list[str], evidence: dict | None = None,
                  asked: list | None = None) -> list[dict]:
    """The concept of each agreed card of one text: ``{"concept": id or "", "code": HPO id or "", "how": ...}``.
    The thesaurus and HPO's exact names first; the rest in one question per interpreter for the whole text. Only
    a publication's reading (``context == "publication"``) writes to the thesaurus."""
    write = context == "publication"
    out: list[dict | None] = [None] * len(cards)
    pending: list[tuple[int, list[tuple[str, str, str]]]] = []
    for n, card in enumerate(cards):
        ev = {**(evidence or {}), "quote": card.get("quote", ""), "readers": readers}
        found = thesaurus.find(card)
        if found is not None:
            out[n] = {"concept": found.id, "code": found.code, "how": "thesaurus"}
            continue
        code = hpo.find(card.get("name", ""))
        if code:
            concept = thesaurus.by_code(code) or Concept(code, hpo.terms.names.get(code, card.get("name", "")), code)
            if write:
                _add(thesaurus, concept, card, {**ev, "how": "hpo_name"})
            out[n] = {"concept": concept.id, "code": code, "how": "hpo_name"}
            continue
        options: list[tuple[str, str, str]] = []
        for concept in thesaurus.candidates(card):
            options.append((concept.id, concept.code, concept.name + (f" ({concept.code})" if concept.code else "")))
        for term, name in hpo.candidates(card):
            if not any(o[1] == term for o in options):
                options.append((term, term, f"{name} ({term})"))
        if options:
            pending.append((n, options[:CANDIDATES]))
    if pending and ask is not None:
        prompt = choice_prompt(context, [(cards[n], [o[2] for o in opts]) for n, opts in pending])
        sizes = [len(opts) for _, opts in pending]
        answers = {r: read_choices(ask(r, prompt), sizes) for r in readers}
        if asked is not None:
            asked.append({"choices": len(pending), "answers": {r: len(a) for r, a in answers.items()}})
        for k, (n, opts) in enumerate(pending):
            chosen = {answers[r].get(k) for r in readers}
            if len(chosen) == 1 and None not in chosen:
                cid, code, shown = opts[chosen.pop() - 1]
                concept = thesaurus.concepts.get(cid) or Concept(cid, shown.rsplit(" (", 1)[0], code)
                if write:
                    _add(thesaurus, concept, cards[n], {**(evidence or {}), "quote": cards[n].get("quote", ""),
                                                        "readers": readers, "how": "chosen_in_context",
                                                        "answers": {r: answers[r].get(k) for r in readers}})
                out[n] = {"concept": concept.id, "code": code, "how": "chosen_in_context"}
    for n, card in enumerate(cards):
        if out[n] is None:
            if write:
                concept = thesaurus.new(card)
                _add(thesaurus, concept, card, {**(evidence or {}), "quote": card.get("quote", ""),
                                                "readers": readers, "how": "new"})
                out[n] = {"concept": concept.id, "code": "", "how": "new"}
            else:
                out[n] = {"concept": "", "code": "", "how": "none"}
    return out  # type: ignore[return-value]


def normalize(card: dict, context: str, hpo: HpoIndex, thesaurus: Thesaurus,
              ask: Callable[[str, str], str] | None, readers: list[str], evidence: dict | None = None,
              asked: list | None = None) -> dict:
    """The concept of one agreed card (``normalize_all`` of one card)."""
    return normalize_all([card], context, hpo, thesaurus, ask, readers, evidence, asked)[0]


__all__ = ["HpoIndex", "choice_prompt", "normalize", "normalize_all", "read_choices"]
