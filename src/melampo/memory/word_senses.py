"""One written form, several meanings: the context and the language decide, and when they do not, the linker abstains.

"GB" is the gallbladder in English radiology, the white cell count in Italian clinical text
("GB 5040/mmc") and a unit of memory. "Ponte" is the pons, a dental bridge or a bypass graft.
"Ileo" is the ileum or an ileus. A person never meets such a word alone; the rest of the sentence,
the language it is written in and the numbers and units next to it settle the meaning before the
word is "recognised". This module does the same for any listed form. It is one mechanism with the
facts kept in data (``data/linking/word_senses.json``), not one rule per word, so a new ambiguous
form is a new entry and a test, not new code.

**What it models, and what it does not.**

* All meanings are considered together, not the first that comes to mind. Psycholinguistics calls
  this reordered access (Duffy, Morris & Rayner 1988; Rodd, Gaskell & Marslen-Wilson 2002/2005):
  every meaning is activated, how common it is and what surrounds it reorder them, and strong
  context removes the meaning that does not fit. Here each sense of a form collects evidence and
  the senses compete.
* Evidence is combined by constraint satisfaction, as in Kintsch's construction-integration: the
  construction is permissive (every sense is a candidate, every cue counts), the integration keeps
  what the constraints support. A cue is a word, a pattern of numbers and units, or the language.
* The anatomical sense is accepted only on positive evidence *and* a margin over the best other
  sense. Silence about a word is not evidence for the anatomical reading: in an ambiguous form the
  default is to abstain, with the competing sense named. A reader who is unsure rereads; the
  linker sends the item to review.
* Language is a discriminant, not a filter. A sense attested in one language only (GB as
  gallbladder: English) loses weight in a text of the other language, and that can be outweighed
  by strong evidence in the sentence ("colecistectomia pregressa, GB non visualizzata").

**What is not demonstrated.** The weights and the starter inventory are design choices, tested on
constructed sentences and on the Italian clinical cases where the GB error was found. They are not
yet measured on a labelled gold set, which is the only thing that can certify the result. The
inventory covers only the forms listed in the data file; a form not listed is not checked here.
Evidence is read from the whole sentence (and an optional wider context at half weight), without
syntactic distance, negation or scope.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# What one cue is worth. A strong cue (a word or a pattern only one sense has) decides almost alone;
# a weak cue (a word that goes with one sense but also appears elsewhere) needs company.
STRONG = 3.0
WEAK = 1.0
CONTEXT_FACTOR = 0.5  # evidence from outside the sentence counts half
LANGUAGE_MATCH = 1.0
LANGUAGE_MISMATCH = -2.0
# The anatomical sense is accepted when it reaches ACCEPT and beats the best other sense by MARGIN.
ACCEPT = 2.0
MARGIN = 1.0

DEFAULT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "linking" / "word_senses.json"
)

_WORD = re.compile(r"[A-Za-zÀ-ÿ0-9]+")
# A written form is a whole word: "ileo" in "ileo-psoas" or "ileo-cecale" is part of another word.
_WHOLE_WORD = re.compile(r"[A-Za-zÀ-ÿ0-9]+(?:-[A-Za-zÀ-ÿ0-9]+)*")
# Words that belong to exactly one of the two languages. Short, frequent, and not shared
# ("a", "in", "no" and "per" are left out because both languages use them).
_ITALIAN = frozenset(
    "il lo la le gli un una di del della dei delle degli nel nella nei nelle sul sulla non e ed che con "
    "sono è presente assente regolare regolari nei limiti dimensioni sede".split()
)
_ENGLISH = frozenset(
    "the of and with is are was were there without has have shows seen noted within "
    "size patient findings".split()
)


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def words_of(text: str) -> list[str]:
    """Lower-case, accent-free words, as written."""
    return [_fold(w) for w in _WORD.findall(text)]


def detect_language(text: str) -> str | None:
    """ "it", "en" or None when the sentence does not say. Counts words only one language has.

    A heuristic, deliberately cautious: it answers only when one language has at least one marker
    and at least twice as many as the other. It is a cue, not a ruling.
    """
    words = words_of(text)
    it = sum(w in _ITALIAN for w in words)
    en = sum(w in _ENGLISH for w in words)
    if it >= 1 and it >= 2 * en:
        return "it"
    if en >= 1 and en >= 2 * it:
        return "en"
    return None


@dataclass(frozen=True)
class Sense:
    id: str
    anatomical: bool
    # What the anatomical sense may link to; empty means any class the other stages find.
    classes: frozenset[str] = frozenset()
    # Languages in which this meaning is attested. Empty means all.
    languages: frozenset[str] = frozenset()
    strong: frozenset[str] = frozenset()
    weak: frozenset[str] = frozenset()
    patterns: tuple[re.Pattern[str], ...] = ()

    def score(
        self, sentence: str, context: str = "", language: str | None = None
    ) -> tuple[float, list[str]]:
        """Evidence for this sense and the cues that gave it."""
        score = 0.0
        seen: list[str] = []
        for text, factor in ((sentence, 1.0), (context, CONTEXT_FACTOR)):
            if not text:
                continue
            words = set(words_of(text))
            for cue in sorted(words & self.strong):
                score += STRONG * factor
                seen.append(cue)
            for cue in sorted(words & self.weak):
                score += WEAK * factor
                seen.append(cue)
            for pattern in self.patterns:
                if pattern.search(text):
                    score += STRONG * factor
                    seen.append(pattern.pattern)
        if language and self.languages:
            if language in self.languages:
                score += LANGUAGE_MATCH
                seen.append(f"language:{language}")
            else:
                score += LANGUAGE_MISMATCH
                seen.append(f"not-{'|'.join(sorted(self.languages))}:{language}")
        return score, seen


@dataclass(frozen=True)
class SenseVerdict:
    accepted: bool
    reason: str = ""
    form: str = ""
    sense: str = ""
    classes: frozenset[str] = frozenset()
    scores: dict[str, float] = field(default_factory=dict)
    cues: dict[str, list[str]] = field(default_factory=dict)
    language: str | None = None


_PASS = SenseVerdict(accepted=True)


@dataclass
class SenseInventory:
    forms: dict[str, tuple[Sense, ...]] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "SenseInventory":
        return cls()

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "SenseInventory":
        forms: dict[str, tuple[Sense, ...]] = {}
        for form, senses in data["forms"].items():
            built = []
            for sid, entry in senses.items():
                built.append(
                    Sense(
                        id=sid,
                        anatomical=bool(entry.get("anatomical", False)),
                        classes=frozenset(entry.get("classes", ())),
                        languages=frozenset(entry.get("languages", ())),
                        strong=frozenset(_fold(w) for w in entry.get("strong", ())),
                        weak=frozenset(_fold(w) for w in entry.get("weak", ())),
                        patterns=tuple(
                            re.compile(p, re.IGNORECASE)
                            for p in entry.get("patterns", ())
                        ),
                    )
                )
            if not any(s.anatomical for s in built) or len(built) < 2:
                raise ValueError(
                    f"{form!r}: an ambiguous form needs an anatomical sense and at least one other"
                )
            forms[_fold(form)] = tuple(built)
        # Other written forms of the same word (plural, other language) share its senses.
        for alias, target in data.get("aliases", {}).items():
            if _fold(target) not in forms:
                raise ValueError(f"alias {alias!r}: unknown form {target!r}")
            forms[_fold(alias)] = forms[_fold(target)]
        return cls(forms)

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "SenseInventory":
        return cls.from_json(json.loads(path.read_text("utf-8")))

    def forms_in(self, mention: str) -> list[str]:
        """The listed forms the mention is written with."""
        words = (_fold(w) for w in _WHOLE_WORD.findall(mention))
        return [w for w in dict.fromkeys(words) if w in self.forms]

    def judge(self, mention: str, sentence: str, context: str = "") -> SenseVerdict:
        """Whether the mention can be read as an anatomical structure here, given everything around it."""
        language = detect_language(sentence)
        limit: set[str] = set()
        judged: dict[str, Any] = {}
        for form in self.forms_in(mention):
            senses = self.forms[form]
            scores: dict[str, float] = {}
            cues: dict[str, list[str]] = {}
            for sense in senses:
                scores[sense.id], cues[sense.id] = sense.score(
                    sentence, context, language
                )
            anatomical = [s for s in senses if s.anatomical]
            others = [s for s in senses if not s.anatomical]
            best = max(anatomical, key=lambda s: scores[s.id])
            rival = max((scores[s.id] for s in others), default=0.0)
            rival_id = max(others, key=lambda s: scores[s.id]).id if others else ""
            # Two anatomical senses (one form, two structures) also compete with each other.
            rest = [scores[s.id] for s in anatomical if s is not best]
            if rest and max(rest) > rival:
                rival = max(rest)
                rival_id = max(
                    (s for s in anatomical if s is not best), key=lambda s: scores[s.id]
                ).id
            top = scores[best.id]
            common = {
                "form": form,
                "scores": scores,
                "cues": cues,
                "language": language,
            }
            if top < ACCEPT:
                # Nothing in the sentence supports the anatomical reading; say who else could be meant.
                reason = (
                    f"sense_conflict:{rival_id}"
                    if rival >= ACCEPT
                    else "sense_unresolved"
                )
                return SenseVerdict(False, reason, sense=best.id, **common)
            if top - rival < MARGIN:
                return SenseVerdict(
                    False, f"sense_conflict:{rival_id}", sense=best.id, **common
                )
            judged = {**common, "sense": best.id}
            if best.classes:
                limit |= best.classes
        if judged:
            return SenseVerdict(True, classes=frozenset(limit), **judged)
        return _PASS


def classes_allowed(verdict: SenseVerdict, cids: Iterable[str]) -> bool:
    """False when the winning sense limits the classes and none of ``cids`` is among them."""
    return not verdict.classes or bool(verdict.classes & set(cids))
