"""A small grammatical profile of a language, for reading the phrase around a mention.

Two things differ between the languages the linker reads, and the chunk lattice needs both:

* **where the head of a noun phrase is.** English compounds are right-headed ("heart rate": a rate),
  Italian ones are left-headed ("frequenza cardiaca": a frequency; "frattura del femore": a fracture);
* **what cannot be a head.** A verb form or an adverb ends a noun phrase; it must not be taken for the
  word that types it ("the heart removed from the donor": the phrase is "heart", not "removed").

This is a *filter on surface forms*, not a part-of-speech tagger: closed classes, adverb endings and
regular past participles, with lists of the common nouns that look like them. It is deliberately
conservative (a noun wrongly kept costs nothing, a noun wrongly dropped loses a head) and the lattice
asks it only about words its own memory does not know. For Italian, morphology alone cannot tell a verb
from a noun ("tessuto", "tratto", "stato" are nouns), so only closed classes and adverbs are filtered;
a tagger (``it_core_news``, Stanza) is the way to more, and ``Grammar.verbal`` is the one place to plug it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_EN_CLOSED = frozenset(
    """is are was were be been being am has have had having do does did done can could may might must
    shall should will would not no never nor i you he she it we they me him her us them my your his its
    our their who whom whose which that what whatever there here very quite rather also just only even
    still already often usually sometimes again once however thus therefore hence moreover whereas""".split()
)
# regular-looking words that are nouns or adjectives, not verbs or adverbs
_EN_NOUN_LY = frozenset(
    """family anomaly assembly ally apply reply supply fly belly bully lily italy rely multiply july
    monopoly homily tally rally jelly folly lilly sibly""".split()
)
_EN_NOUN_ED = frozenset(
    """hundred bed red need seed speed weed feed breed creed steed shed sled wed fed led bled shred
    sacred naked wicked kindred""".split()
)
_EN_IRREGULAR_PARTICIPLE = frozenset(
    """shown seen found taken given known made grown drawn broken withdrawn written chosen done gone
    begun worn torn borne sworn thrown blown flown held kept left lost met paid read said sent set
    shot sold spent stood struck taught told thought understood won""".split()
)

_IT_CLOSED = frozenset(
    """è sono era erano fu furono sarà saranno sia siano essere ha hanno ho
    aveva avevano avuto avere non né ne si ci vi mi ti lo la li le gli il un una uno che chi cui
    quale quali questo questa questi queste quello quella quelli quelle ove dove come quando
    quindi però anche ancora già solo sempre mai molto poco più meno circa
    di a da in con su per tra fra del dello della dei degli delle dell nel nello nella nei negli nelle
    sul sullo sulla sui sugli sulle al allo alla ai agli alle dal dallo dalla dai dagli dalle
    e ed o od ma se""".split()
)

_OF_EN = r"of"
_OF_IT = r"(?:di|d['’]|del|dello|della|dei|degli|delle|dell['’])"
_DET_EN = r"(?:the\s+|a\s+|an\s+|this\s+|these\s+|that\s+)?"
_DET_IT = r"(?:il\s+|lo\s+|la\s+|l['’]|i\s+|gli\s+|le\s+|un\s+|uno\s+|una\s+|un['’])?"


@dataclass(frozen=True)
class Grammar:
    language: str
    head_side: str  # "right" (English) or "left" (Italian)
    closed: frozenset[str]
    of_after: re.Pattern[str] = field(repr=False, default=None)  # type: ignore[assignment]
    of_before: re.Pattern[str] = field(repr=False, default=None)  # type: ignore[assignment]

    @classmethod
    def for_language(cls, language: str) -> Grammar:
        if language == "en":
            return cls("en", "right", _EN_CLOSED,
                       re.compile(rf"^\s+{_OF_EN}\s+{_DET_EN}", re.IGNORECASE),
                       re.compile(rf"\b{_OF_EN}\s+{_DET_EN}$", re.IGNORECASE))
        if language == "it":
            return cls("it", "left", _IT_CLOSED,
                       re.compile(rf"^\s+{_OF_IT}\s*{_DET_IT}", re.IGNORECASE),
                       re.compile(rf"\b{_OF_IT}\s*{_DET_IT}$", re.IGNORECASE))
        raise ValueError(f"no grammar for language {language!r}")

    def verbal(self, word: str) -> bool:
        """The surface of a verb form or an adverb (never true for a word in the closed nouns above)."""
        w = word.lower()
        if self.language == "en":
            if w in _EN_NOUN_LY or w in _EN_NOUN_ED:
                return False
            if w in _EN_IRREGULAR_PARTICIPLE:
                return True
            if len(w) > 4 and w.endswith("ly"):
                return True
            return len(w) > 4 and w.endswith("ed")
        if self.language == "it":
            return len(w) > 6 and w.endswith("mente")
        return False

    def ends_phrase(self, word: str) -> bool:
        """The word cannot be inside a noun phrase: a closed-class word or a verb form or adverb."""
        w = word.lower()
        return w in self.closed or self.verbal(w)

    def could_head(self, word: str) -> bool:
        w = word.lower()
        return bool(w) and any(c.isalpha() for c in w) and not self.ends_phrase(w)


__all__ = ["Grammar"]
