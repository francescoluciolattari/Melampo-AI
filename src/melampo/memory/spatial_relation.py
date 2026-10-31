"""What the sentence does with a structure it names: is the finding *in* or *of* it, or does the structure only
place something else? (31 October 2026, the "landmark" reading.)

A radiologist reading "some enlarged nodes are evident along the aortic arch" does not think the aorta is
abnormal: the arch places the nodes. Spatial role labelling of radiology reports (Rad-SpRL; Datta et al.,
*Journal of Biomedical Informatics* 2020) names the three parts: the thing placed (*trajector*: the nodes), the
place of reference (*landmark*: the aortic arch) and the word that relates them (*spatial indicator*: along).
What matters here is the kind of relation. The region connection calculus (Randell, Cui & Cohn 1992) separates
a region that is part of another ("a nodule *in* the lobe", "calcification *of* the arch") from one that is only
connected to it or apart from it ("nodes *along* the arch", "fluid *between* the loops", "the appendix *lateral
to* the cecum"). Only the first says something about the structure.

This module reads that relation from the words before the mention. It **does not decide**: the relation is one
node of the Kintsch reader's network (``ci_reader``, family ``relation``), weighed with the others, and the
reader's reading is compared with the model's (``finding_owner``). The words are the spatial indicators of
radiology reports in English and Italian, grouped by the relation they express, not by the cases met.

The scan goes left from the mention, over articles, laterality and position words and numbers; over a
coordination to the indicator of the first conjunct ("near the pole of the spleen and the splenic flexure");
and over "of" after a word that names a part or a region ("the loops of the small intestine", "the anterior
pole of the spleen"), because such a part is where the relation is anchored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CONTAINMENT = "containment"   # the finding is in or of the structure: the structure is described
PROXIMITY = "proximity"       # the structure only places something else: a landmark
DIRECTION = "direction"       # towards or into the structure: weaker evidence of a landmark
AFFECTED = "affected"         # the structure is the object of a verb ("infiltrates the duodenum")

# Spatial indicators, longest first within a language. Multi-word ones are matched as written, and an
# Italian one also with its articulated preposition ("vicino al", "in prossimità della").
_EN = {
    PROXIMITY: (
        "in the vicinity of", "in the region of", "in the area of", "in the projection of", "close to", "next to",
        "adjacent to", "lateral to", "laterally to", "medial to", "medially to", "internally to", "externally to", "dorsally to", "ventrally to", "cranially to",
        "caudally to", "anterior to", "posterior to", "superior to", "inferior to",
        "proximal to", "distal to", "outside of", "outside", "beyond", "near", "nearby", "along", "alongside",
        "around", "surrounding", "between", "among", "beside", "abutting", "above", "below", "behind",
        "in front of", "away from", "apart from",
    ),
    DIRECTION: ("into", "towards", "toward", "onto"),
    CONTAINMENT: ("within", "inside", "throughout", "involving", "affecting", "of", "in", "at", "on"),
}
_IT_ART = r"(?:a|al|allo|alla|ai|agli|alle|all'|di|del|dello|della|dei|degli|delle|dell'|da|dal|dalla|dai|dalle)?"
_IT = {
    PROXIMITY: (
        "in prossimità", "nelle vicinanze", "nell'area", "in regione", "nella regione", "vicino", "accanto",
        "adiacente", "lateralmente", "medialmente", "anteriormente", "posteriormente", "superiormente",
        "inferiormente", "dorsalmente", "ventralmente", "cranialmente", "caudalmente", "lungo", "attorno", "intorno", "circostante", "circostanti", "tra", "fra",
        "all'esterno", "al di fuori", "al di sopra", "al di sotto", "sopra", "sotto", "dietro", "davanti",
    ),
    DIRECTION: ("verso", "dentro"),
    CONTAINMENT: (
        "a carico", "all'interno", "a livello", "in corrispondenza", "al", "allo", "alla", "ai", "agli", "alle",
        "all'", "nel", "nello", "nella", "nei", "negli", "nelle", "nell'", "in", "di", "del",
        "dello", "della", "dei", "degli", "delle", "dell'", "sul", "sulla", "sui", "sulle",
    ),
}
# Words that sit between the indicator and the name without changing the relation.
_SKIP = frozenset("""
the a an this these that those its their both some
left right bilateral bilaterally anterior posterior upper lower superior inferior medial lateral proximal distal
mid middle main common internal external deep superficial central peripheral dorsal ventral apical basal
il lo la i gli le l' l un una uno un' dx sx destro destra sinistro sinistra destri sinistri bilaterale
anteriore posteriore superiore inferiore mediale laterale prossimale distale medio media principale comune
interno interna esterno esterna profondo centrale periferico periferica
""".split())
# Words that name a part or a region of what follows "of": the relation is anchored there and the scan goes on.
_PARTS = frozenset("""
pole poles flank flanks wall walls edge edges margin margins side sides aspect aspects portion portions part parts segment
segments loop loops base apex tip surface surfaces border borders area region regions zone level levels projection
course lumen floor roof neck body tail head
polo poli fianco fianchi parete pareti margine margini lato lati versante porzione porzioni parte parti segmento segmenti ansa
anse base apice superficie bordo bordi area regione zona livello proiezione decorso lume pavimento tetto collo
corpo coda testa
""".split())
_JUNCTIONS = frozenset("""
confluence junction bifurcation transition border passage
confluenza giunzione biforcazione passaggio transizione confine
""".split())
_COORD = frozenset({"and", "or", "e", "ed", "o", "nonché", ","})
_TOKEN = re.compile(r"[A-Za-zÀ-ÿ]+'?|[,;:()]|\d+(?:[.,]\d+)?")


@dataclass(frozen=True)
class Relation:
    kind: str          # CONTAINMENT, PROXIMITY, DIRECTION, AFFECTED, or "" (nothing read)
    indicator: str     # the words that relate ("along", "vicino al", "of")
    placed: str        # what the clause describes before the relation (its first noun), for the person


def _indicators(language: str) -> list[tuple[str, tuple[str, ...]]]:
    table = _IT if language == "it" else _EN
    out = []
    for kind, words in table.items():
        for w in words:
            out.append((kind, tuple(_TOKEN.findall(w.lower()))))
    return sorted(out, key=lambda kv: -len(kv[1]))


def _match_before(tokens: list[str], i: int, language: str) -> tuple[str, int, str] | None:
    """An indicator that ends at token ``i`` (Italian: possibly followed by an articulated preposition)."""
    if language == "it" and re.fullmatch(_IT_ART, tokens[i]):
        # an indicator followed by its articulated preposition ("vicino al") before the preposition alone
        for kind, words in _indicators(language):
            n = len(words)
            if i - n >= 0 and tuple(tokens[i - n: i]) == words:
                return kind, i - n, " ".join(words) + " " + tokens[i]
    for kind, words in _indicators(language):
        n = len(words)
        if i - n + 1 >= 0 and tuple(tokens[i - n + 1: i + 1]) == words:
            return kind, i - n + 1, " ".join(words)
    return None


def read_relation(sentence: str, start: int, language: str = "en", verbal=None) -> Relation:
    """The relation between what the clause describes and the structure named at ``start``.

    ``verbal``: a function word -> bool for verbs of the language (``Grammar.verbal``), to read a structure
    that is the object of a verb ("infiltrates the duodenum") as affected."""
    tokens = [t.lower() for t in _TOKEN.findall(sentence[:start])]
    cut = max((k for k, t in enumerate(tokens) if t in (";", ":", "(")), default=-1)
    tokens = tokens[cut + 1:]
    i = len(tokens) - 1
    after_coordination = False
    steps = 0
    while i >= 0 and steps < 24:
        steps += 1
        t = tokens[i]
        if t in _SKIP or t.isdigit() or re.fullmatch(r"\d+(?:[.,]\d+)?", t):
            i -= 1
            continue
        found = _match_before(tokens, i, language)
        if found:
            kind, at, words = found
            of_like = words.split()[0] in ("of", "di", "del", "dello", "della", "dei", "degli", "delle", "dell'")
            if kind == CONTAINMENT and of_like and at > 0:
                before = at - 1
                while before >= 0 and tokens[before] in _SKIP:
                    before -= 1
                if before >= 0 and tokens[before] in _PARTS:
                    i = before - 1      # "the loops of the small intestine": the relation is the loops'
                    continue
            if kind == CONTAINMENT and after_coordination and of_like:
                i = at - 1              # "of" inside the first conjunct: its own relation is further left
                continue
            if kind == PROXIMITY and words in ("between", "among", "tra", "fra") and at > 0:
                before = at - 1
                while before >= 0 and tokens[before] in _SKIP:
                    before -= 1
                if before >= 0 and tokens[before] in _JUNCTIONS:
                    # "the confluence between the cephalic and the subclavian vein": a junction is part of both
                    return Relation(CONTAINMENT, tokens[before] + " " + words, _placed(tokens[:before], language))
            return Relation(kind, words, _placed(tokens[:at], language))
        if t in _COORD:
            after_coordination = True
            i -= 1
            continue
        if after_coordination:
            i -= 1                      # the words of the first conjunct
            continue
        if verbal is not None and verbal(t):
            return Relation(AFFECTED, t, _placed(tokens[:i], language))
        return Relation("", "", "")     # a word of the name itself, or nothing that relates
    return Relation("", "", "")


def _placed(tokens: list[str], language: str) -> str:
    """The words of the clause before the relation (what is described there), for the person's queue: not
    evidence, and not a parse."""
    words = [t for t in tokens if t not in ",;:()"]
    return " ".join(words[-8:])


__all__ = ["AFFECTED", "CONTAINMENT", "DIRECTION", "PROXIMITY", "Relation", "read_relation"]
