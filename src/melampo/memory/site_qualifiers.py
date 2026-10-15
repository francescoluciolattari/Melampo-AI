"""What the words around a structure say about it: how precise the site is, how marked the finding is, and
how certain (15 October 2026).

Frank: an adverb says nothing about *whether* a word is a body site, and the reader must not let it vote on that
("approximately T11 level"); but it is not empty. "Approximately T11" is T11 give or take a level; "mild
wedging" is not "marked wedging"; "lievemente iperdensa" says a degree and a density. The graph keeps them.

Three attributes, each with the words of the report and their offsets, so a reader can always go back to the
text:

* **location** -- how precisely the report places the finding on the structure: ``as_named`` (nothing
  qualifies the name), ``approximate`` ("approximately", "about", "circa", "in prossimità di"), ``range`` ("T11-T12",
  "between T10 and T12", "tra L4 e L5") or ``region`` ("the region of", "area", "regione", "zona");
* **degree** -- how marked the finding of the clause is: ``low`` ("mild", "minimal", "lieve", "modesto"),
  ``moderate`` ("moderate", "discreto"), ``high`` ("marked", "severe", "marcato", "cospicuo"); and **character**,
  the radiological quality written as a word made of a direction and a quality ("hyperdense", "ipointenso",
  "anecogeno", "opacity", "lucency"), read from the shape of the word rather than from a list of findings;
* **certainty** -- the ConText reading of the clause that already exists (``assertion.AssertionDetector``:
  factual, possible, hypothetical; affirmed or negated; current or historical).

They describe a link; they never decide one. Words of a closed class (degree adverbs, hedges, prepositions of
approximation) are a vocabulary, not a rule about a case: they are kept as written and grouped, and a word not
in the vocabulary is simply not recorded. Scope: the clause of the mention (sentence edges, ";" and the
adversative terminators of the assertion cues), as ConText does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .assertion import AssertionDetector, select_cues

AS_NAMED, APPROXIMATE, RANGE, REGION = "as_named", "approximate", "range", "region"
LOW, MODERATE, HIGH = "low", "moderate", "high"

# Words that make a place approximate, by language (several words are matched as a phrase).
_APPROXIMATE = {
    "en": ("approximately", "approx", "about", "around", "roughly", "near", "nearly", "close to", "adjacent to",
           "in the vicinity of", "at or near", "at about", "near the level of"),
    "it": ("circa", "approssimativamente", "all incirca", "in prossimità di", "in prossimità del",
           "in prossimità della", "in prossimità dello", "in prossimità dell", "nei pressi di", "nei pressi del",
           "nei pressi della", "vicino a", "vicino al", "vicino alla", "intorno a", "intorno al", "intorno alla"),
}
# Nouns that make the place a region around the structure ("the region of the hilum", "regione ilare").
_REGION = {
    "en": ("region", "regions", "area", "areas", "zone", "zones", "territory", "vicinity"),
    "it": ("regione", "regioni", "zona", "zone", "area", "aree", "territorio", "distretto", "sede"),
}
# Degree words: closed classes of adjectives and adverbs of intensity.
_DEGREE = {
    "en": {
        LOW: ("minimal", "minimally", "mild", "mildly", "slight", "slightly", "subtle", "subtly", "small", "trace",
              "tiny", "faint", "faintly", "minor", "minute"),
        MODERATE: ("moderate", "moderately"),
        HIGH: ("marked", "markedly", "severe", "severely", "significant", "significantly", "extensive",
               "extensively", "large", "massive", "prominent", "considerable", "considerably", "advanced"),
    },
    "it": {
        LOW: ("minimo", "minima", "minimi", "minime", "lieve", "lievi", "lievemente", "modesto", "modesta",
              "modesti", "modeste", "modestamente", "piccolo", "piccola", "piccoli", "piccole", "tenue", "tenui",
              "lieve-moderato", "accenno"),
        MODERATE: ("moderato", "moderata", "moderati", "moderate", "moderatamente", "discreto", "discreta",
                   "discreti", "discrete", "discretamente"),
        HIGH: ("marcato", "marcata", "marcati", "marcate", "marcatamente", "severo", "severa", "severi", "severe",
               "grave", "gravi", "importante", "importanti", "cospicuo", "cospicua", "cospicui", "cospicue",
               "notevole", "notevoli", "notevolmente", "voluminoso", "voluminosa", "voluminosi", "voluminose",
               "esteso", "estesa", "estesi", "estese", "massivo", "massiva", "ampio", "ampia", "ampi", "ampie"),
    },
}
# Radiological character: a direction prefix and a quality stem, in either language.
_DIRECTION = (("hyper", HIGH), ("iper", HIGH), ("hypo", LOW), ("ipo", LOW), ("iso", "equal"), ("an", "absent"))
_QUALITY = (
    ("density", r"dens\w*|attenua\w*"),
    ("signal", r"intens\w*"),
    ("echogenicity", r"echo\w*|ecogen\w*|ecoic\w*|echoic\w*"),
    ("perfusion", r"perfus\w*"),
    ("metabolism", r"metabol\w*|captant\w*|capta\w*"),
)
_CHARACTER = re.compile(
    r"\b(" + "|".join(d for d, _ in _DIRECTION) + r")[-\s]?(" + "|".join(q for _, q in _QUALITY) + r")\b",
    re.IGNORECASE,
)
# Qualities written as one word without a direction prefix.
_PLAIN_CHARACTER = (
    (re.compile(r"\b(opacit\w*|opacific\w*|opacament\w*|radiopac\w*|radio-opac\w*|addensament\w*)\b", re.I),
     "density", HIGH),
    (re.compile(r"\b(lucenc\w*|lucent|radiolucen\w*|radiotrasparen\w*|ipertrasparen\w*)\b", re.I), "density", LOW),
    (re.compile(r"\b(sclerotic|sclerosing|sclerotizzat\w*|addensat\w*)\b", re.I), "density", HIGH),
    (re.compile(r"\b(lytic|litic\w*|osteolitic\w*|osteolytic)\b", re.I), "density", LOW),
)
_RANGE = re.compile(
    r"\b([CTLS]\d{1,2})\s*(?:-|–|/|to|a|e|through)\s*([CTLS]?\d{1,2})\b"
    r"|\b(?:between|tra|fra)\s+(?:the\s+|il\s+|lo\s+|la\s+)?([\w-]+)\s+(?:and|e)\s+(?:the\s+|il\s+|lo\s+|la\s+)?([\w-]+)",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"[\w’']+(?:-[\w’']+)*", re.UNICODE)
# The words that may stand between an approximation and the place it qualifies: articles, "of", "level", a
# distance ("approximately at the level of", "a circa 5 cm dall'arcata"); a noun in between is another thing
# ("around the contusion in the right lobe" is around the contusion).
_LINKING = frozenset("""the a an of at to level levels the from cm mm
    il lo la i gli le l un una uno di del della dello dei degli delle dell d a al alla allo ai agli alle all da dal
    dalla dallo dai dagli dalle dall livello""".split()) | frozenset(str(n) for n in range(0, 100))
_JOINED = re.compile(r"[\w\s’'-]*")
_GOVERNS = re.compile(r"\s+(?:(?:of|in)\s+(?:the\s+)?|(?:di|del|della|dello|dei|degli|delle|dell['’]\s*|d['’]\s*)?)",
                      re.IGNORECASE)
NEAR_WORDS = 5   # words before the mention read for an approximation or a region (inside the clause)


@dataclass(frozen=True)
class Cue:
    word: str
    start: int
    end: int
    value: str

    def as_text(self) -> str:
        return f"{self.value}:{self.word}@{self.start}"


@dataclass(frozen=True)
class SiteQualifiers:
    location: str = AS_NAMED
    location_cues: tuple[Cue, ...] = ()
    degree: str = ""
    degree_cues: tuple[Cue, ...] = ()
    degree_scope: str = ""                # "phrase" (same stretch between commas) or "clause"
    character: tuple[Cue, ...] = ()       # value "<quality>:<direction>" ("density:high")
    character_scope: str = ""
    certainty: str = "factual"
    polarity: str = "affirmed"
    temporality: str = "current"
    certainty_cues: tuple[str, ...] = ()
    language: str = "en"
    clause: tuple[int, int] = (0, 0)

    def as_dict(self) -> dict[str, Any]:
        """Flat values (strings and lists of strings) for a graph edge or a JSON record."""
        return {
            "location": self.location, "location_cues": [c.as_text() for c in self.location_cues],
            "degree": self.degree, "degree_cues": [c.as_text() for c in self.degree_cues],
            "degree_scope": self.degree_scope, "character_scope": self.character_scope,
            "character": [c.value for c in self.character], "character_cues": [c.as_text() for c in self.character],
            "certainty": self.certainty, "polarity": self.polarity, "temporality": self.temporality,
            "certainty_cues": list(self.certainty_cues), "language": self.language,
        }


def _fold(text: str) -> str:
    return re.sub(r"[’']", " ", text.lower())


def _find_phrases(text: str, lo: int, hi: int, phrases, value: str) -> list[Cue]:
    """Every phrase of the list in ``text[lo:hi]`` on word boundaries, with offsets in ``text``."""
    folded = _fold(text)
    found: list[Cue] = []
    for phrase in phrases:
        for m in re.finditer(r"(?<!\w)" + re.escape(_fold(phrase)) + r"(?!\w)", folded[lo:hi]):
            found.append(Cue(text[lo + m.start(): lo + m.end()], lo + m.start(), lo + m.end(), value))
    return found


def read_site_qualifiers(sentence: str, start: int, end: int, language: str = "en") -> SiteQualifiers:
    """The qualifiers of the structure named at ``sentence[start:end]``."""
    language = language if language in _DEGREE else "en"
    detector = AssertionDetector(cues=select_cues(language))
    status = detector.detect(sentence, start, end)
    lo, hi = detector._clause_bounds(_normalise_for_bounds(sentence), start, end)

    # location: an approximation or a region in the few words before the mention (or right after it,
    # "T11 level approximately"), a range on the mention itself or around it
    tokens = [m for m in _TOKEN.finditer(sentence, lo, start)]
    near_lo = tokens[-NEAR_WORDS].start() if len(tokens) >= NEAR_WORDS else lo
    after = _TOKEN.match(sentence, min(hi, end + 1)) if end < hi else None
    near_hi = min(hi, after.end() if after else end)
    location_cues: list[Cue] = []
    for m in _RANGE.finditer(sentence, max(lo, start - 30), min(hi, end + 30)):
        if m.start() <= end and m.end() >= start:
            location_cues.append(Cue(m.group(0), m.start(), m.end(), RANGE))
    # a cue belongs to the place only inside its phrase: nothing but words between it and the mention
    # ("circa 40-50%), Arteria ..." is the approximation of a percentage, not of the artery)
    location_cues += [c for c in _find_phrases(sentence, near_lo, start, _APPROXIMATE[language], APPROXIMATE)
                      if all(w.lower() in _LINKING for w in re.findall(r"\w+", sentence[c.end:start]))
                      and _JOINED.fullmatch(sentence[c.end:start])]
    # a region noun governs the mention ("the region of the hilum", "in sede ilare", "regione glutea") or
    # follows it ("hilar region"); "aree di necrosi alle dita" is a finding in an area, not a region
    location_cues += [c for c in _find_phrases(sentence, near_lo, start, _REGION[language], REGION)
                      if _GOVERNS.fullmatch(sentence[c.end:start])]
    location_cues += [c for c in _find_phrases(sentence, end, near_hi, _REGION[language], REGION)
                      if not sentence[end:c.start].strip()]
    location = AS_NAMED
    for value in (RANGE, APPROXIMATE, REGION):
        if any(c.value == value for c in location_cues):
            location = value
            break

    # degree and character: the words of the phrase of the mention (between commas, colons, brackets) first;
    # only when the phrase has none, the words of the clause, and the scope says so: in "vie biliari non
    # dilatate, colecisti alitiasica e piccola cisti renale" the "piccola" is the cyst's, not the ducts'
    plo, phi = _phrase_bounds(sentence, lo, hi, start, end)
    degree_cues: list[Cue] = []
    for level, words in _DEGREE[language].items():
        degree_cues += _find_phrases(sentence, lo, hi, words, level)
    degree_cues.sort(key=lambda c: c.start)
    degree_cues, degree_scope = _scoped(degree_cues, plo, phi)
    degree = min(degree_cues, key=lambda c: abs(c.start - start)).value if degree_cues else ""

    character: list[Cue] = []
    for m in _CHARACTER.finditer(sentence, lo, hi):
        direction = next(v for d, v in _DIRECTION if m.group(1).lower() == d)
        quality = next(q for q, rx in _QUALITY if re.fullmatch(rx, m.group(2), re.I))
        character.append(Cue(m.group(0), m.start(), m.end(), f"{quality}:{direction}"))
    for rx, quality, direction in _PLAIN_CHARACTER:
        for m in rx.finditer(sentence, lo, hi):
            if not any(c.start <= m.start() < c.end for c in character):
                character.append(Cue(m.group(0), m.start(), m.end(), f"{quality}:{direction}"))
    character.sort(key=lambda c: c.start)
    character, character_scope = _scoped(character, plo, phi)

    return SiteQualifiers(
        location=location, location_cues=tuple(sorted(location_cues, key=lambda c: c.start)),
        degree=degree, degree_cues=tuple(degree_cues), degree_scope=degree_scope, character=tuple(character),
        character_scope=character_scope,
        certainty=status.certainty, polarity=status.polarity, temporality=status.temporality,
        certainty_cues=status.cues, language=language, clause=(lo, hi),
    )


_PHRASE_BREAK = re.compile(r"[,:()\[\]\"“”«»]")


def _phrase_bounds(text: str, lo: int, hi: int, start: int, end: int) -> tuple[int, int]:
    """The stretch of the clause around the mention between commas, colons, brackets and quotes."""
    left = max((m.end() for m in _PHRASE_BREAK.finditer(text, lo, start)), default=lo)
    right = next((m.start() for m in _PHRASE_BREAK.finditer(text, end, hi)), hi)
    return left, right


def _scoped(cues: list[Cue], plo: int, phi: int) -> tuple[list[Cue], str]:
    inside = [c for c in cues if plo <= c.start and c.end <= phi]
    if inside:
        return inside, "phrase"
    return (cues, "clause") if cues else ([], "")


def _normalise_for_bounds(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() or ch in ".;" else " " for ch in text)


__all__ = ["AS_NAMED", "APPROXIMATE", "RANGE", "REGION", "LOW", "MODERATE", "HIGH", "Cue", "SiteQualifiers",
           "read_site_qualifiers"]
