"""The word that governs a mention, read by a dependency parser (13 October 2026).

Why: the chunk lattice reads a window of words, each with the kind the vocabulary gives it; it does not
know which word governs which. "liberation of the caecum and of the appendix", "the abdominal wall
incisions", "a plain radiograph of the patient's knees": the role of the structure is decided by the
word the mention depends on (liberation, incisions, radiograph), which may sit outside the block the
lattice builds, across a coordination or a preposition. A reader reads the sentence from the top down:
the clause, then which word governs which, then the word (Frank, 13 October 2026).

What it gives: the governor of the mention (its lemma, the relation, the path) as found by the parser:

* across a coordination the mention takes the governor of the first conjunct ("of the caecum and of
  the appendix" -> liberation);
* "X of <mention>" -> X (English ``prep``/``pobj``; Italian ``nmod`` with ``case`` di/del...);
* "<mention> X" (a compound modifier, "abdominal wall incisions") -> X;
* the verb of which the mention is subject or object ("the pelvic cavity was explored") -> the verb.

The kind of the governor is not decided here: the reader asks its vocabulary (``BlockMemory``, from
NCIt). A governor the vocabulary does not know is reported as such, so that the reader can say it does
not know instead of reading the mention as a structure by default.

The parser is spaCy (MIT licence) with a pipeline per language (``en_core_web_md``, ``it_core_news_md``,
both MIT); it is optional: without spaCy the reader works as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

DEFAULT_MODELS = {"en": "en_core_web_md", "it": "it_core_news_md"}

# Relations through which the governor is looked for further up (the mention is part of a larger noun).
_THROUGH = frozenset({"conj", "appos", "flat", "flat:name", "fixed"})
# A noun that modifies another noun: the other noun governs it.
_NOUN_MODIFIER = frozenset({"compound", "nmod", "nmod:poss", "poss", "npadvmod", "amod"})
# The mention is an argument of a verb.
_ARGUMENT = frozenset({"nsubj", "nsubjpass", "nsubj:pass", "dobj", "obj", "obl", "iobj", "attr", "oprd"})
_NOUNS = frozenset({"NOUN", "PROPN"})
# Prepositions that place something in the mention ("opacity in the lower lobes", "lesione nel fegato"):
# the mention is where the governor is, whatever the governor is (a finding, a density, a clip).
LOCATIVE = frozenset({
    "in", "within", "at", "near", "on", "into", "inside", "along", "over", "throughout", "beneath", "below",
    "above", "behind", "under", "around", "across", "overlying", "adjacent",
    "nel", "nello", "nella", "nei", "negli", "nelle", "a", "al", "allo", "alla", "ai", "agli", "alle",
    "su", "sul", "sullo", "sulla", "sui", "sugli", "sulle", "presso", "entro", "lungo", "sopra", "sotto",
    "dentro", "attorno", "intorno",
})


@dataclass(frozen=True)
class Governor:
    lemma: str              # the governing word, lemmatised and lower case
    relation: str           # "of", "compound", "nmod:di", "verb:nsubjpass", ...
    pos: str                # its part of speech (NOUN, VERB, ...)
    path: tuple[str, ...] = field(default=())  # how it was reached, for the trace

    @property
    def nominal(self) -> bool:
        """The mention is a modifier of this noun: the noun phrase it heads contains the mention."""
        return not self.relation.startswith("verb:") and not self.locative

    @property
    def locative(self) -> bool:
        """The governor is placed in the mention ("opacity in the lobe"): the mention is a location."""
        return self.relation.rsplit(":", 1)[-1] in LOCATIVE


class GovernorParser:
    """The governor of a mention in a sentence. One spaCy pipeline per language, loaded on first use."""

    def __init__(self, models: dict[str, str] | None = None, max_steps: int = 6):
        self.models = dict(models or DEFAULT_MODELS)
        self.max_steps = max_steps
        self._nlp: dict[str, object] = {}

    def _pipeline(self, language: str):
        if language not in self._nlp:
            import spacy

            name = self.models.get(language)
            if name is None:
                raise ValueError(f"no parser for language {language!r}")
            self._nlp[language] = spacy.load(name, exclude=["ner", "textcat"])
        return self._nlp[language]

    @lru_cache(maxsize=4096)  # noqa: B019 - one parser per process, sentences repeat across mentions
    def _parse(self, language: str, sentence: str):
        return self._pipeline(language)(sentence)

    def governor(self, sentence: str, start: int, end: int, language: str = "en") -> Governor | None:
        """The governor of the mention at ``sentence[start:end]``; ``None`` when it heads its clause."""
        doc = self._parse(language, sentence)
        inside = [t for t in doc if t.idx < end and t.idx + len(t.text) > start]
        if not inside:
            return None
        heads = [t for t in inside if not (start <= t.head.idx < end) or t.head is t]
        tok = heads[-1] if heads else inside[-1]
        path: list[str] = []
        for _ in range(self.max_steps):
            dep, head = tok.dep_, tok.head
            if head is tok or dep == "ROOT":
                return None
            if dep in _THROUGH:
                path.append(f"{tok.text}-{dep}->{head.text}")
                tok = head
                continue
            if dep == "pobj" and head.dep_ == "prep" and head.head is not head:
                # English "X of <mention>": the preposition hangs on X
                gov = head.head
                path.append(f"{tok.text}-pobj->{head.text}-prep->{gov.text}")
                rel = head.lemma_.lower()
                if gov.pos_ in _NOUNS:
                    return Governor(gov.lemma_.lower(), rel, gov.pos_, tuple(path))
                return Governor(gov.lemma_.lower(), f"verb:{rel}", gov.pos_, tuple(path))
            if dep.startswith("nmod") and head.pos_ in _NOUNS:
                case = next((c.lemma_.lower() for c in tok.children if c.dep_ == "case"), "")
                path.append(f"{tok.text}-{dep}->{head.text}")
                return Governor(head.lemma_.lower(), f"nmod:{case}" if case else "nmod", head.pos_, tuple(path))
            if dep in _NOUN_MODIFIER and head.pos_ in _NOUNS:
                path.append(f"{tok.text}-{dep}->{head.text}")
                return Governor(head.lemma_.lower(), "compound", head.pos_, tuple(path))
            if dep in _ARGUMENT or dep.startswith("obl") or dep.startswith("nsubj"):
                path.append(f"{tok.text}-{dep}->{head.text}")
                case = next((c.lemma_.lower() for c in tok.children if c.dep_ in ("case", "prep")), "")
                rel = f"verb:{dep}" + (f":{case}" if case else "")
                return Governor(head.lemma_.lower(), rel, head.pos_, tuple(path))
            return None
        return None


__all__ = ["DEFAULT_MODELS", "LOCATIVE", "Governor", "GovernorParser"]
