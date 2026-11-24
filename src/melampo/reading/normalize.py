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


def choice_prompt(context: str, card: dict, options: list[str]) -> str:
    where = "a document of one patient" if context == "case" else "a medical publication"
    lines = "\n".join(f"{i}. {o}" for i, o in enumerate(options, 1))
    return (f"A finding agreed by two readers in a sentence of {where}.\n"
            f"Sentence: \"{card.get('quote', '')}\"\n"
            f"Finding: {json.dumps(described(card, 'case' if context == 'case' else 'publication'), ensure_ascii=False)}\n"
            f"Which of these is the same finding, the same thing with the same meaning (not a broader, narrower or "
            f"related one)?\n{lines}\n"
            "Answer only JSON: {\"same\": the number, or null when none is}")


def read_choice(answer: str, n: int) -> int | None:
    m = re.search(r"\"same\"\s*:\s*(null|\d+)", answer or "")
    if m is None or m.group(1) == "null":
        return None
    k = int(m.group(1))
    return k if 1 <= k <= n else None


def _add(thesaurus: Thesaurus, concept: Concept, card: dict, evidence: dict) -> None:
    """Every interpreter's label of the agreed reading becomes a label of the concept."""
    for name in dict.fromkeys([card.get("name", "")] + list((card.get("labels") or {}).values())):
        if name:
            thesaurus.add(concept, {**card, "name": name}, evidence)


def normalize(card: dict, context: str, hpo: HpoIndex, thesaurus: Thesaurus,
              ask: Callable[[str, str], str] | None, readers: list[str], evidence: dict | None = None,
              asked: list | None = None) -> dict:
    """The concept of one agreed card: ``{"concept": id or "", "code": HPO id or "", "how": ...}``. Only a
    publication's reading (``context == "publication"``) writes to the thesaurus."""
    write = context == "publication"
    found = thesaurus.find(card)
    if found is not None:
        return {"concept": found.id, "code": found.code, "how": "thesaurus"}
    code = hpo.find(card.get("name", ""))
    if code:
        concept = thesaurus.by_code(code) or Concept(code, hpo.terms.names.get(code, card.get("name", "")), code)
        if write:
            _add(thesaurus, concept, card, {**(evidence or {}), "how": "hpo_name"})
        return {"concept": concept.id, "code": code, "how": "hpo_name"}
    options: list[tuple[str, str, str]] = []     # (concept id, code, shown)
    for concept in thesaurus.candidates(card):
        options.append((concept.id, concept.code, concept.name + (f" ({concept.code})" if concept.code else "")))
    for term, name in hpo.candidates(card):
        if not any(o[1] == term for o in options):
            options.append((term, term, f"{name} ({term})"))
    options = options[:CANDIDATES]
    if options and ask is not None:
        prompt = choice_prompt(context, card, [o[2] for o in options])
        answers = {r: read_choice(ask(r, prompt), len(options)) for r in readers}
        if asked is not None:
            asked.append({"quote": card.get("quote", ""), "name": card.get("name"), "options": [o[2] for o in options],
                          "answers": answers})
        chosen = set(answers.values())
        if len(chosen) == 1 and None not in chosen:
            cid, code, shown = options[chosen.pop() - 1]
            concept = thesaurus.concepts.get(cid) or Concept(cid, shown.rsplit(" (", 1)[0], code)
            if write:
                _add(thesaurus, concept, card, {**(evidence or {}), "how": "chosen_in_context", "answers": answers})
            return {"concept": concept.id, "code": code, "how": "chosen_in_context"}
    if write:
        concept = thesaurus.new(card)
        _add(thesaurus, concept, card, {**(evidence or {}), "how": "new"})
        return {"concept": concept.id, "code": "", "how": "new"}
    return {"concept": "", "code": "", "how": "none"}


__all__ = ["HpoIndex", "choice_prompt", "normalize", "read_choice"]
