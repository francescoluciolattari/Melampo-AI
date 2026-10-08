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

    def language_part(self, language: str | None) -> float:
        """What the language of the sentence alone adds to this sense's score."""
        if not language or not self.languages:
            return 0.0
        return LANGUAGE_MATCH if language in self.languages else LANGUAGE_MISMATCH


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


_WORD_AFTER = re.compile(r"^[\s-]*([A-Za-zÀ-ÿ]+)")
_WORD_BEFORE = re.compile(r"([A-Za-zÀ-ÿ]+)[\s-]*$")
# A head before the structure with a preposition between: "function of the liver", "toni del cuore",
# "funzione dell'utero". English and Italian, since reports mix them.
_WORD_BEFORE_OF = re.compile(
    r"([A-Za-zÀ-ÿ]+)\s+(?:of(?:\s+the)?\s+|(?:di|del|della|dello|dei|degli|delle)\s+|(?:dell|d)['’]\s*)$",
    re.IGNORECASE,
)


_LETTERS = "A-Za-zÀ-ÿ"


def locate(
    mention: str, sentence: str, start: int | None = None
) -> re.Match[str] | None:
    """Where the mention stands in the sentence, as a whole word: "thyroid" is not the "thyroid" of
    "Hypothyroidism". With ``start`` (the mention's offset in the sentence, when the caller knows
    it), the occurrence there, not the first one: "Brain volume correlates with intrinsic brain
    activity" has two. Without a whole-word occurrence, the first occurrence anywhere."""
    text = mention.strip()
    if not text:
        return None
    pattern = re.compile(
        rf"(?<![{_LETTERS}]){re.escape(text)}(?![{_LETTERS}])", re.IGNORECASE
    )
    found = list(pattern.finditer(sentence))
    if not found:
        return re.search(re.escape(text), sentence, flags=re.IGNORECASE)
    if start is not None:
        return min(found, key=lambda m: abs(m.start() - start))
    return found[0]


# One coordinated word between the structure and the head ("liver and renal function tests",
# "bladder and bowel function"), or a one- or two-character name ("Liver X receptor").
_COORDINATED = re.compile(
    r"^\s+(?:and|or|e|o|ed|/)\s+[A-Za-zÀ-ÿ]+(?:-[A-Za-zÀ-ÿ]+)?(?=[\s-])|^\s+[A-Z0-9]{1,2}(?=\s)"
)


def _head(
    mention: str,
    sentence: str,
    after: frozenset[str],
    before: frozenset[str],
    start: int | None = None,
    not_coordinated: frozenset[str] = frozenset(),
    depth: int = 0,
    molecules: bool = False,
) -> str:
    """The head word next to the mention: where the head stands follows the syntax of the language.

    An English compound has its head on the right ("liver function"): ``after``. Italian puts the
    head on the left ("funzione epatica", "ormoni tiroide"): ``before``. Both languages can put it
    on the left with a preposition ("function of the liver", "toni del cuore"): every word of
    either list, except a prefix such as "anti". A comma or a full stop ends the construction.
    """
    if not (after or before):
        return ""
    match = locate(mention, sentence, start)
    if not match:
        return ""
    rest = sentence[match.end() :]
    found = _WORD_AFTER.match(rest)
    next_word = found.group(1) if found else ""
    if found and _fold(found.group(1)) in after:
        return _fold(found.group(1))
    joined = _COORDINATED.match(rest)
    if joined:
        found = _WORD_AFTER.match(rest[joined.end() :])
        if (
            found
            and _fold(found.group(1)) in after
            and (
                _fold(found.group(1)) not in not_coordinated
                or not joined.group(0).split()[0].isalpha()
                or joined.group(0).strip().isupper()
            )
        ):
            # only a measurement shared by both ("liver and renal function"): "thyroid lobe and
            # central lymph node" names two structures
            return _fold(found.group(1))
    left = sentence[: match.start()]
    found = _WORD_BEFORE.search(left)
    if found and _fold(found.group(1)) in before:
        return _fold(found.group(1))
    if molecules and found and _fold(found.group(1)).startswith("anti"):
        # an antibody against the organ's antigen ("antisoluble liver", "antinuclear")
        return "anti"
    if molecules and next_word and _enzyme(next_word):
        # an enzyme named after the organ ("thyroid peroxidase", "liver esterase")
        return _fold(next_word)
    slashed = re.search(r"([A-Za-zÀ-ÿ]+)/$", left)
    if slashed and depth == 0:
        # "anti-SLA/LP (antisoluble liver/liver pancreas)": a word joined by a slash shares the
        # construction of the word before it
        shared = _head(
            slashed.group(1),
            sentence,
            after,
            before,
            slashed.start(1),
            not_coordinated,
            depth=1,
            molecules=molecules,
        )
        if shared:
            return shared
    found = _WORD_BEFORE_OF.search(left)
    if found and _fold(found.group(1)) in (after | before) - _PREFIXES:
        return _fold(found.group(1))
    return ""


_PREFIXES = frozenset(("anti",))
_NOT_ENZYMES = frozenset(
    "disease diseases release releases increase increases decrease decreases case cases base "
    "bases phase phases purchase showcase".split()
)


def _enzyme(word: str) -> bool:
    folded = _fold(word)
    return (
        len(folded) > 5
        and folded.endswith(("ase", "ases"))
        and folded not in _NOT_ENZYMES
    )


# Words that head the name of an institution, a study, a data set or an instrument (English and
# Italian), and the short words that may join the capitalised words of such a name.
_NAME_HEADS = frozenset(
    """association associations society foundation college institute institutes institution
    university hospital clinic centre center consortium council federation committee group network
    organization organisation academy study registry register workshop congress conference journal
    trial initiative program programme project atlas index system questionnaire inventory
    exchange guidelines guideline survey database bank cohort
    associazione societa fondazione istituto universita ospedale clinica centro consorzio
    federazione comitato gruppo rete studio registro congresso indice questionario""".split()
)
_NAME_JOINERS = frozenset(
    "of for and the on in de del della di per e degli delle dei".split()
)
_CAPITALISED = re.compile(r"[A-ZÀ-Ý][a-zà-ÿ]+(?:-[A-Za-zà-ÿ]+)*$")
_NAME_TOKEN = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]*|[^\sA-Za-zÀ-ÿ]")


def _proper_name(mention: str, sentence: str, start: int | None = None) -> str:
    match = locate(mention, sentence, start)
    if not match or not _CAPITALISED.match(
        sentence[match.start() : match.end()].split()[0]
    ):
        return ""
    tokens = [(m.group(0), m.start(), m.end()) for m in _NAME_TOKEN.finditer(sentence)]
    inside = [
        i
        for i, (_, a, b) in enumerate(tokens)
        if a >= match.start() and b <= match.end()
    ]
    if not inside:
        return ""

    def part(i: int) -> bool:
        word = tokens[i][0]
        return bool(_CAPITALISED.match(word)) or word.isupper() and len(word) > 1

    def joined(i: int, step: int) -> int:
        """How many tokens to move from ``i`` to reach the next capitalised word, through at most
        two joining words ("Association for the Study of the Liver"); 0 if none."""
        for gap in (1, 2, 3):
            j = i + step * gap
            if not 0 <= j < len(tokens):
                return 0
            if part(j):
                return gap
            if tokens[j][0].lower() not in _NAME_JOINERS:
                return 0
        return 0

    lo, hi = inside[0], inside[-1]
    while gap := joined(lo, -1):
        lo -= gap
    while gap := joined(hi, 1):
        hi += gap
    words = [t[0] for t in tokens[lo : hi + 1]]
    if sentence[tokens[hi][2] :].lstrip().startswith(":"):
        # a heading ("Liver Study: normal"), not a name
        return ""
    capitalised = [w for w in words if _CAPITALISED.match(w)]
    if len(capitalised) < 2:
        return ""
    if any(
        _fold(w) in _NAME_HEADS
        for w in capitalised
        if w.lower() not in mention.lower().split()
    ):
        return " ".join(words)
    return ""


@dataclass
class SenseInventory:
    forms: dict[str, tuple[Sense, ...]] = field(default_factory=dict)
    # Words next to a structure's name that make it the modifier of a measurement or an assay
    # ("heart rate", "liver function", "anti-thyroid"): true for every structure, so it is data
    # about the neighbour, not about the structure.
    heads_after: frozenset[str] = frozenset()
    heads_before: frozenset[str] = frozenset()
    # Words of a procedure on the structure ("liver biopsy", "biopsia del fegato"): the structure is
    # named, and its role in the sentence is the site of a procedure, not of a finding.
    procedures_after: frozenset[str] = frozenset()
    procedures_before: frozenset[str] = frozenset()
    # Heads that stop the link without making the structure the site of a measurement
    # ("para-aortic lymph nodes", "heart team", "portal phase").
    heads_not_measured: frozenset[str] = frozenset()

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
        heads = data.get("attribute_heads", {})
        procedures = data.get("procedure_heads", {})
        return cls(
            forms,
            frozenset(_fold(w) for w in heads.get("after", ())),
            frozenset(_fold(w) for w in heads.get("before", ())),
            frozenset(_fold(w) for w in procedures.get("after", ())),
            frozenset(_fold(w) for w in procedures.get("before", ())),
            frozenset(_fold(w) for w in heads.get("not_a_measurement", ())),
        )

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "SenseInventory":
        return cls.from_json(json.loads(path.read_text("utf-8")))

    def attribute_head(
        self, mention: str, sentence: str, start: int | None = None
    ) -> str:
        """The neighbouring word that makes the mention the modifier of a measurement, else "".

        "heart rate", "liver function tests", "thyroid-stimulating hormone", "anti-thyroid
        antibodies": the structure is named but not meant. Only the word right after (or right
        before) the mention counts; a comma or a full stop ends the construction.
        """
        return _head(
            mention,
            sentence,
            self.heads_after,
            self.heads_before,
            start,
            self.heads_not_measured,
            molecules=True,
        )

    def proper_name(self, mention: str, sentence: str, start: int | None = None) -> str:
        """The name of an institution, a study, a registry or an instrument the mention is a word
        of ("American Heart Association", "The Cancer Genome Atlas", "Prostate Imaging Reporting and
        Data System", "Dallas Heart Study"), else "". The structure is part of a proper name, not
        named. Read from the writing: a run of capitalised words (joined by short function words)
        that contains the mention and a word that heads such names."""
        return _proper_name(mention, sentence, start)

    def procedure_head(
        self, mention: str, sentence: str, start: int | None = None
    ) -> str:
        """The neighbouring word of a procedure done on the structure ("liver biopsy", "biopsia del
        fegato", "resezione epatica"), else "". Same positions as ``attribute_head``."""
        return _head(
            mention, sentence, self.procedures_after, self.procedures_before, start
        )

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
            # The language of the sentence can tip a choice, not make one: without it, the words of
            # the text must reach the threshold ("medio-lateral axis of the hand": one weak cue,
            # "lateral", plus English was a vertebra; MedMentions, 8 October 2026).
            from_text = top - best.language_part(language)
            common = {
                "form": form,
                "scores": scores,
                "cues": cues,
                "language": language,
            }
            if top < ACCEPT or from_text < ACCEPT:
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
