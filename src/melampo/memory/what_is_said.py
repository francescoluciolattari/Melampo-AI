"""What the reader knows of a sentence, to check what a model says it states (2 November 2026).

The reader does not decide what a sentence says of a structure: the models do, copying the words. The reader
checks those words with what a radiologist knows besides them, and when the two contradict each other a person
decides. Its knowledge comes from the ontologies and from the grammar, never from a list written for the cases:

* **what is an abnormality.** HPO names the phenotypic abnormalities and, in their logical definitions, the PATO
  qualities they are made of ("increased size": enlarged, dilated...; "increased thickness": thickened). A word of
  a PATO quality used by HPO is a *quality*: it states a deviation wherever it stands in the phrase ("an enlarged
  liver", "the liver is enlarged"). A word of an HPO name that is neither its anatomy nor a quality is an
  *entity* ("cyst", "nodule", "effusion"): it states an abnormality only as the head of its phrase. In
  "intracranial fluid spaces" the head is "spaces" and "fluid" says which spaces; in "free fluid in the pelvis"
  the head is "fluid" (``grammar.Grammar``: English phrases are right-headed, Italian ones left-headed). The
  radiological characters and characters the site reader already knows ("hypodense", "hyperechoic") count as qualities.
* **what is negated or past**, from the assertion detector (``assertion``): "not enlarged" states no deviation;
  "history of" makes it a fact of the past.
* **what is coordinated.** "Masses in the liver ... and in the pancreas", "flow through the stomach and the
  duodenum": two conjuncts share what the sentence says of them, and neither carries the other's finding.

Without the ontologies (``MELAMPO_HPO``, ``MELAMPO_PATO``, ``MELAMPO_UBERON``) only the radiological characters
and degrees are known, and the reader checks less: it never decides more.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .grammar import Grammar
from .hpo_candidates import GENERIC_WORDS, PHENOTYPE_ROOT, read_obo, stem, stems

_TOKEN = re.compile(r"[A-Za-zÀ-ÿ]+(?:-[A-Za-zÀ-ÿ]+)*|\d+(?:[.,]\d+)?|[^\sA-Za-zÀ-ÿ\d]")
# words that end a noun phrase besides the language's closed classes: prepositions and conjunctions
_BOUNDARY = frozenset("""
of in on at to from by with without for and or but into onto over under within through along between among
across around near beside behind above below after before during per via than as
di a da in con su per tra fra e o ma oltre sotto sopra lungo verso dopo prima durante senza
""".split())
_COORDINATORS = frozenset("and or e ed o oppure nonché".split())
_PREPOSITIONS = frozenset("""
of in on at to from through into within along nel nella nello nei negli nelle del della dello dei degli delle
di in al alla allo ai agli alle sul sulla sullo sui sugli sulle
""".split())
_ARTICLES = frozenset("the a an il lo la i gli le l un uno una both".split())
# stems that name no deviation on their own ("increased" says nothing without what increases)
_EMPTY = frozenset(stem(w) for w in """
increased decreased reduced elevated raised low high lower higher abnormal abnormality normal morphology quality
present presence absent absence
level levels amount size shape structure function process feature involving affecting type
""".split())


ENTITY_TERMS = 5      # a lesion heads the names of many phenotypes (cyst, nodule, stone, polyp, effusion)


@dataclass
class AnomalyWords:
    """The words that state an abnormality: ``qualities`` (stems of the PATO qualities HPO's phenotypes are made
    of, each with the sites HPO uses it at: "thin" is an abnormality of the skin in HPO, of no bladder wall) and
    ``entities`` (the heads of HPO names made of no quality, that head no anatomical name: cyst, effusion)."""

    qualities: dict[str, set[str]] = field(default_factory=dict)   # stem -> the UBERON sites HPO uses it at
    entities: set[str] = field(default_factory=set)
    anatomy: object = None          # hpo_candidates.Anatomy (UBERON), for the classes of a structure

    def sites_of(self, cid: str) -> set[str]:
        """The UBERON classes of a linker class, up to its organ (empty without UBERON: any site counts)."""
        from .hpo_candidates import site_anchors

        anchors = list(site_anchors(cid)) if cid else []
        if self.anatomy is None or not anchors:
            return set()
        return set(self.anatomy.site_classes(anchors))

    @classmethod
    def from_files(cls, hpo: Path, pato: Path | None = None, uberon: Path | None = None) -> AnomalyWords:
        with open(hpo, encoding="utf-8") as fh:
            terms = read_obo(fh, "HP:")
        # only the phenotypic abnormalities (not the modifiers "Mild", "Progressive", the modes of inheritance)
        parents = {h: [p.split(" ")[0] for p in t.get("is_a", [])] for h, t in terms.items()}
        phenotypic: dict[str, bool] = {}

        def is_phenotypic(h: str, depth: int = 0) -> bool:
            if h == PHENOTYPE_ROOT:
                return True
            if h not in phenotypic:
                phenotypic[h] = depth < 40 and any(is_phenotypic(p, depth + 1) for p in parents.get(h, []))
            return phenotypic[h]

        pato_names: dict[str, list[str]] = {}
        pato_parents: dict[str, list[str]] = {}
        if pato is not None and Path(pato).exists():
            with open(pato, encoding="utf-8") as fh:
                for pid, t in read_obo(fh, "PATO:").items():
                    pato_names[pid] = [(t.get("name") or [""])[0]] + [_quoted(s) for s in t.get("synonym", [])
                                                                      if " EXACT" in s]
                    pato_parents[pid] = [p.split(" ")[0] for p in t.get("is_a", []) if p.startswith("PATO:")]
        anatomy_names: dict[str, list[str]] = {}
        anatomy = None
        if uberon is not None and Path(uberon).exists():
            from .hpo_candidates import Anatomy as Uberon

            with open(uberon, encoding="utf-8") as fh:
                anatomy = Uberon.from_obo(fh)
            anatomy_names = {u: [n for n in ns if " " in n or n == anatomy.labels.get(u)] or ns
                             for u, ns in anatomy.names.items()}
        # the words that head an anatomical name ("liver", "wall", "space", "fluid" of "cerebrospinal fluid")
        anatomy_heads = {_head_of_name(n) for names in anatomy_names.values() for n in names}
        qualities: dict[str, set[str]] = {}
        heads: dict[str, set[str]] = {}
        for h, t in terms.items():
            if not is_phenotypic(h):
                continue
            inter = t.get("intersection_of", [])
            site = {w for x in inter for w in x.split() if w.startswith("UBERON:")}
            used = {w for x in inter for w in x.split() if w.startswith("PATO:")}
            for pid in used if site else ():   # a quality of no structure says nothing of one
                if _descends(pid, POSITION, pato_parents):
                    continue                    # a position says where, not what
                for name in pato_names.get(pid, []):
                    words = stems(name)
                    if len(words) == 1:         # "enlarged", "dilated", "thickened": the word of the quality
                        qualities.setdefault(words[0], set()).update(site)
            if used:
                continue            # the abnormality is the quality ("Thickened gallbladder wall": thickened)
            # a term without a quality names its abnormality: its head ("Hepatic cysts": cyst)
            for name in [(t.get("name") or [""])[0]] + [_quoted(s) for s in t.get("synonym", []) if " EXACT" in s]:
                head, rest = _head_of_name(name), stems(name)[:-1]
                # "Abnormal liver morphology", "Reduced bladder capacity": the head is the attribute that deviates,
                # not a lesion; a lesion's name has only its site and its kind ("Hepatic cysts", "Renal cyst")
                if head and rest and not any(r in _EMPTY or r in GENERIC_WORDS for r in rest):
                    heads.setdefault(head, set()).add(h)
        # the attributes the deviations are of ("increased size" is a "size", "increased volume" a "volume"):
        # an attribute is measured, not abnormal, and a laterality or a position says where, not what
        used_all = {w for t in terms.values() for x in t.get("intersection_of", []) for w in x.split()
                    if w.startswith("PATO:")}
        attributes: set[str] = set()
        stack = [p for u in used_all for p in pato_parents.get(u, [])]
        seen: set[str] = set()
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            if u not in used_all:       # "cystic" is above "polycystic" and is itself a deviation HPO uses
                attributes |= set(stems((pato_names.get(u) or [""])[0]))
            stack += pato_parents.get(u, [])
        for st in _EMPTY | GENERIC_WORDS | attributes:
            qualities.pop(st, None)
        # a lesion heads the names of many phenotypes ("Hepatic cysts", "Renal cyst", "Ovarian cyst"...): the head of
        # ENTITY_TERMS names or more is an entity, a word met in one or two is a part of their names
        entities = {h for h, named in heads.items() if len(named) >= ENTITY_TERMS}
        entities -= _EMPTY | GENERIC_WORDS | anatomy_heads | attributes
        entities = {e for e in entities if len(e) > 3 and not e.isdigit()}
        if anatomy is not None:
            # each site with the classes above it up to its organ (``site_classes``, as for the HPO candidates)
            up: dict[str, set[str]] = {}
            for st, at in qualities.items():
                qualities[st] = {"*"} if "*" in at else {c for u in at for c in up.setdefault(
                    u, set(anatomy.site_classes([u])))}
        return cls(qualities, entities, anatomy)

    @classmethod
    def from_env(cls) -> AnomalyWords:
        hpo = os.environ.get("MELAMPO_HPO")
        if not hpo or not Path(hpo).exists():
            return cls()
        pato, uberon = os.environ.get("MELAMPO_PATO"), os.environ.get("MELAMPO_UBERON")
        return cls.from_files(Path(hpo), Path(pato) if pato else None, Path(uberon) if uberon else None)

    def __bool__(self) -> bool:
        return bool(self.qualities or self.entities)


POSITION = "PATO:0000140"     # PATO's position: left, right, anterior, displaced...


def _descends(pid: str, root: str, parents: dict[str, list[str]]) -> bool:
    seen, stack = set(), [pid]
    while stack:
        u = stack.pop()
        if u == root:
            return True
        if u not in seen:
            seen.add(u)
            stack += parents.get(u, [])
    return False


def _head_of_name(name: str) -> str:
    """The head of an HPO name, as English names it: the last word before "of", "in", a comma or a parenthesis
    ("Hepatic cyst": cyst; "Cyst of the liver": cyst; "Pleural effusion": effusion)."""
    first = re.split(r"\s+(?:of|in|on|at|with|due to|involving|affecting)\s+|[,(]", name.lower())[0]
    words = stems(first)
    return words[-1] if words else ""


def _quoted(text: str) -> str:
    m = re.match(r'\s*"(.*?)"', text)
    return m.group(1) if m else ""


def phrases(text: str, language: str = "en") -> list[list[tuple[str, int, int]]]:
    """The noun phrases of ``text`` as lists of (word, start, end): runs of words between punctuation, closed
    classes, prepositions, conjunctions and verb forms (``Grammar``). A filter on surface forms, not a parser."""
    grammar = Grammar.for_language(language if language in ("en", "it") else "en")
    out, cur = [], []
    for m in _TOKEN.finditer(text):
        w = m.group(0)
        low = w.lower()
        if not re.match(r"[A-Za-zÀ-ÿ]", w) or low in _BOUNDARY or low in grammar.closed:
            if cur:
                out.append(cur)
            cur = []
            continue
        cur.append((w, m.start(), m.end()))
    if cur:
        out.append(cur)
    return out


def stated_anomalies(text: str, language: str = "en", words: AnomalyWords | None = None,
                     sites: set[str] | None = None) -> list[str]:
    """The words of ``text`` that state an abnormality, affirmed: a quality anywhere in its phrase when HPO uses
    it at the structure (``sites``: its UBERON classes and their ancestors; any site when not given), an entity
    only as the head of its phrase, a radiological character; not when negated."""
    from .assertion import POLARITY_NEGATED, AssertionDetector, select_cues

    words = words or AnomalyWords()
    head_last = language != "it"
    detector = AssertionDetector(cues=select_cues(language if language in ("en", "it") else "en"))
    found: list[tuple[str, int, int]] = []
    for phrase in phrases(text, language):
        head = phrase[-1] if head_last else phrase[0]
        for w, s, e in phrase:
            st = stem(w)
            at = words.qualities.get(st)
            quality = at is not None and (not sites or "*" in at or bool(at & sites))
            if quality or (st in words.entities and (w, s, e) == head) or _radiological(w, language):
                found.append((w, s, e))
    out = []
    for w, s, e in found:
        if detector.detect(text, s, e).polarity == POLARITY_NEGATED:
            continue
        out.append(w)
    return out


# above or below the tissue's own (an "anechoic" bladder, an "isodense" lesion say nothing abnormal by themselves)
_RADIOLOGICAL = re.compile(r"^(hyper|hypo)-?(dens|echo|echogen|intens|attenuat|vascular|perfus|enhanc)\w*$|"
                           r"^(iper|ipo)-?(dens|ecogen|intens|attenuat|vascolar|perfus|captant)\w*$", re.I)


def _radiological(word: str, language: str) -> bool:
    """A radiological character: a density, echogenicity, intensity, enhancement above or below the tissue's
    (the qualities the site reader reads, ``site_qualifiers``)."""
    return bool(_RADIOLOGICAL.match(word))


def negated(text: str, language: str = "en") -> bool:
    """Does the assertion detector read the last word of ``text`` as negated ("not enlarged", "no stones")?"""
    from .assertion import POLARITY_NEGATED, AssertionDetector, select_cues

    words = list(re.finditer(r"[A-Za-zÀ-ÿ]+", text or ""))
    if len(words) < 2:
        return False
    detector = AssertionDetector(cues=select_cues(language if language in ("en", "it") else "en"))
    last = words[-1]
    return detector.detect(text, last.start(), last.end()).polarity == POLARITY_NEGATED


def past(text: str, language: str = "en") -> bool:
    """Does the assertion detector read ``text`` as a fact of the past (its historical cues)?"""
    from .assertion import TEMPORALITY_HISTORICAL, AssertionDetector, select_cues

    detector = AssertionDetector(cues=select_cues(language if language in ("en", "it") else "en"))
    return detector.detect(text, 0, len(text)).temporality == TEMPORALITY_HISTORICAL


def conjuncts(sentence: str, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Are the structures at ``a`` and ``b`` (character spans, in either order) coordinated? Either nothing but a
    coordinator and an article between them ("the stomach and the duodenum", "liver and stomach parenchyma"), or
    each opened by the same preposition and the second by a coordinator ("in the liver ... and in the
    pancreas"): what the sentence says of one it says of the other."""
    (s1, e1), (s2, e2) = sorted([a, b])
    between = [t.lower() for t in re.findall(r"[A-Za-zÀ-ÿ']+|,", sentence[e1:s2])]
    if between and between[0] in (",",) and len(between) > 1:
        between = between[1:]
    if between and between[0] in _COORDINATORS and all(t in _ARTICLES for t in between[1:]):
        return True
    before_second = [t.lower() for t in re.findall(r"[A-Za-zÀ-ÿ']+", sentence[:s2])][-4:]
    before_first = [t.lower() for t in re.findall(r"[A-Za-zÀ-ÿ']+", sentence[:s1])][-3:]
    lead = [t for t in before_second if t not in _ARTICLES]
    if len(lead) >= 2 and lead[-2] in _COORDINATORS and lead[-1] in _PREPOSITIONS:
        first = [t for t in before_first if t not in _ARTICLES]
        return bool(first) and first[-1] == lead[-1]
    return False


__all__ = ["AnomalyWords", "conjuncts", "negated", "past", "phrases", "stated_anomalies"]
