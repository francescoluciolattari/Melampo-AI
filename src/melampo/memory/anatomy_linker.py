"""Link an anatomical mention in an Italian or English report to a TotalSegmentator class -- or abstain.

**The guarantee is about silence, not about coverage.** No linker can promise
zero errors on text it has never seen, and a radiologist cannot either. What
this module promises is narrower and testable: it accepts a link automatically
only when independent checks agree, and everything else goes to the review
list with the reason. "I don't know" is a correct output; a wrong class
delivered as if it were right is the failure the design is built against.

**How the stages mirror a reader, and why each is there.**

1. *Recognition from knowledge* (`Lexicon`). An expert recognises "LID",
   "epistrofeo", "VI costa dx" instantly because they know the names, not by
   reasoning. Dual-process accounts of clinical reasoning put knowledge, not
   deliberation, at the centre of expertise (Norman et al., J Eval Clin Pract
   2024). Here that knowledge is an explicit bilingual lexicon of names,
   synonyms and standard abbreviations, matched exactly after normalisation.
   No similarity score: either the name is known or it is not.
2. *Expectation from context*. Comprehension is predictive: the region a
   sentence is about narrows what a word can mean, and a mismatch triggers
   reanalysis (the N400/P600 retrieval-integration account). "S1" is the first
   sacral vertebra in a spine report and Couinaud segment I in a liver report;
   "D2" is a vertebra or the second part of the duodenum; "T2" is a vertebra
   or an MRI sequence; "LM" is the middle lobe or the left main coronary
   artery. Such names are accepted only when the region is stated next to
   them (in the mention, a section heading, or a few words away) and no other
   region is named in the sentence. Otherwise the linker abstains.
3. *Integration check* (`verify`). Attributes a reader would never get wrong
   when reading slowly -- side, rib or vertebral number and what kind of thing
   carries the number, lobe position, internal versus common, one structure
   versus two -- are extracted deterministically and must agree with the
   candidate. A side or number the mention does not state is never supplied
   by a model when the structure exists on both sides or at several levels.
   These checks run before the language models (only compatible options are
   offered: the "restrictive decoding" idea of LLM4BioEL, EMNLP Findings
   2025) and again after.
4. *Deliberation* (injected chat models). Only for what recognition did not
   resolve: two models must independently choose the same option. Agreement
   is necessary, never sufficient: errors of different LLMs are strongly
   correlated -- when two models both err they agree about 60% of the time
   (Kim et al., ICML 2025) -- which is exactly what happened with "LID" in the
   2026-10-05 run. That is why stage 3 is deterministic and comes first.
5. *Asking*. Anything else is an abstention with a reason, for the
   end-of-ingestion list of doubtful reports.

**What the language models never do**: produce a code. They choose among
candidates the system supplies; models asked to emit ontology identifiers fail
on most terms they rarely saw in training (arXiv 2509.04458).

**Known limits, stated rather than hidden** (each is a strict xfail in the tests).
A mention with no structural word at all ("lingula", "dolore epatico") can still
be linked wrongly if retrieval ranks a wrong organ first *and* both models pick
it: no attribute shows the error. Closing it needs part-of knowledge (a curated
parts table, or the ontology's part_of relations once Italian names exist).
Parts that imply a side or a number ("processo odontoideo" -> C2) abstain.
Spaces ("loggia renale", "ipocondrio") are never linked to the organ, because
after surgery the space is empty. Bare level codes ("L5", "D2", "T2", "S1") are
accepted only with positive evidence; T1/T2 only with a vertebra word in the
mention, since in an MRI report they are sequences.
"""

import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

SIDE_RIGHT = "<dx>"
SIDE_LEFT = "<sn>"
SIDE_BOTH = "<bil>"
POS_UP = "<sup>"
POS_MID = "<mid>"
POS_DOWN = "<inf>"
COORDINATION = "<and>"

# fmt: off
_SIDE_WORDS = {
    **dict.fromkeys(("destro", "destra", "destri", "destre", "dx", "dex", "right", "rt"), SIDE_RIGHT),
    **dict.fromkeys(("sinistro", "sinistra", "sinistri", "sinistre", "sn", "sx", "left", "lt"), SIDE_LEFT),
    **dict.fromkeys(("bilaterale", "bilaterali", "bilateral", "entrambi", "entrambe", "ambedue", "both"), SIDE_BOTH),
}
_POSITION_WORDS = {
    **dict.fromkeys(("superiore", "superiori", "superior", "upper"), POS_UP),
    **dict.fromkeys(("medio", "media", "middle", "mid"), POS_MID),
    **dict.fromkeys(("inferiore", "inferiori", "inferior", "lower"), POS_DOWN),
}
# Adjective -> the noun it qualifies. Words that are also ordinary nouns or
# directions ("colica" = colic, "cranial" = towards the head) are left out.
_ADJECTIVES = {
    "splenico": "milza", "splenica": "milza", "splenici": "milza", "splenic": "spleen",
    "epatico": "fegato", "epatica": "fegato", "epatici": "fegato", "epatiche": "fegato", "hepatic": "liver",
    "renale": "rene", "renali": "rene", "renal": "kidney",
    "surrenale": "surrene", "surrenali": "surrene", "surrenalico": "surrene", "surrenalica": "surrene",
    "gastrico": "stomaco", "gastrica": "stomaco", "gastric": "stomach", "colonic": "colon",
    "vescicale": "vescica", "tracheale": "trachea", "tracheal": "trachea",
    "esofageo": "esofago", "esofagea": "esofago", "esophageal": "esophagus", "oesophageal": "esophagus",
    "oesophagus": "esophagus", "tiroideo": "tiroide", "tiroidea": "tiroide",
    "pancreatico": "pancreas", "pancreatica": "pancreas", "pancreatic": "pancreas",
    "duodenale": "duodeno", "duodenal": "duodenum",
    "polmonare": "polmone", "polmonari": "polmone", "pulmonary": "lung",
    "cerebrale": "encefalo", "encefalico": "encefalo", "cerebral": "brain", "cranico": "cranio", "cranica": "cranio",
    "costale": "costa", "costali": "costa", "costal": "rib",
    "vertebrale": "vertebra", "vertebrali": "vertebra", "vertebral": "vertebra", "vertebre": "vertebra", "vertebrae": "vertebra",
    "sternale": "sterno", "sternal": "sternum", "femorale": "femore", "femoral": "femur",
    "omerale": "omero", "humeral": "humerus", "scapolare": "scapola", "scapular": "scapula",
    "aortico": "aorta", "aortica": "aorta", "aortic": "aorta",
    "cardiaco": "cuore", "cardiaca": "cuore", "cardiac": "heart",
    "prostatico": "prostata", "prostatica": "prostata", "prostatic": "prostate",
}
_PLURALS = {
    "coste": "costa", "costole": "costola", "ribs": "rib", "reni": "rene", "kidneys": "kidney",
    "polmoni": "polmone", "lungs": "lung", "surreni": "surrene", "adrenals": "adrenal",
}
# Words naming the tissue or container of a structure, not a different one.
_NOISE = frozenset((
    "parenchima", "parete", "pareti", "lume", "tratto", "ghiandola",
    "muscolo", "muscoli", "osso", "corpo", "soma", "livello", "corrispondenza",
    "parenchyma", "wall", "lumen", "gland", "muscle", "muscles", "bone", "body", "level",
))
# A space where an organ is, or was: after nephrectomy the "loggia renale" is empty.
CONTAINER_WORDS = frozenset((
    "loggia", "logge", "sede", "regione", "area", "fossa", "letto", "spazio", "bed", "space", "region",
    # abdominal regions are places on the body, not organs
    "ipocondrio", "epigastrio", "mesogastrio", "ipogastrio", "fianco", "quadrante",
    "hypochondrium", "epigastrium", "hypogastrium", "flank", "quadrant",
))
# What kind of structure a word names. A mention of an artery is never an organ or a vein:
# "arteria femorale" is not the femur, "arteria splenica" is not the splenic vein, an "ilo" is not the organ.
_TYPE_WORDS = {
    **dict.fromkeys(("arteria", "arterie", "artery", "arteries", "arterioso", "arteriosa", "arterial"), "<art>"),
    **dict.fromkeys(("vena", "vene", "vein", "veins", "venoso", "venosa", "venous"), "<vein>"),
    **dict.fromkeys(("dotto", "dotti", "duct", "ducts", "duttale"), "<duct>"),
    **dict.fromkeys(("nervo", "nervi", "nerve", "nerves"), "<nerve>"),
    **dict.fromkeys(("linfonodo", "linfonodi", "linfonodale", "linfonodali", "lymph", "node", "nodes", "nodal"), "<node>"),
    **dict.fromkeys(("ilo", "ili", "ilare", "ilari", "hilum", "hilar", "hila"), "<hilum>"),
}
STRUCTURE_TYPES = frozenset(_TYPE_WORDS.values())
# Positive evidence that a bare level code ("L5", "D7", "S1") names a vertebra.
VERTEBRA_WORDS = frozenset((
    "vertebra", "soma", "somatico", "somatica", "somi", "peduncolo", "pedicle", "lamina", "apofisi", "spinosa",
    "spinous", "crollo", "collapse", "frattura", "fracture", "schmorl", "spondilolisi", "spondylolysis", "listesi",
    "spondilolistesi", "spondylolisthesis", "livello", "level", "metamero", "emisoma",
))
_STOP = frozenset((
    "di", "del", "della", "dello", "dei", "degli", "delle", "la", "il", "lo", "i", "gli", "le",
    "un", "una", "uno", "al", "alla", "allo", "ai", "alle", "nel", "nella", "nello", "nei", "nelle",
    "sul", "sulla", "sullo", "sui", "sulle", "da", "dal", "dalla", "con", "per", "in", "a",
    "of", "the", "an", "at", "on", "to", "from", "with",
))
_COORD = frozenset(("e", "ed", "and"))
_RIB_NOUNS = frozenset(("costa", "costola", "rib"))
_SEGMENT_NOUNS = frozenset(("segmento", "segment"))
_NUMBERED_NOUNS = _RIB_NOUNS | _SEGMENT_NOUNS | {"coste", "costole", "ribs", "vertebra", "metamero"}
# A number next to these names something no segmentation class is: a nerve
# root, a disc, a foramen. The mention is not linked.
_NON_STRUCTURE_NUMBERED = frozenset((
    "radice", "radici", "root", "roots", "nervo", "nerve", "disco", "disc", "discale", "forame", "foramen",
    "spazio", "space", "intersomatico", "intervertebrale", "intervertebral",
))
# Qualifiers that make a different structure: "carotide interna" is not "carotide comune".
# Canonical tokens so Italian and English compare; "ipogastrica" is the internal iliac.
_QUALIFIER_WORDS = {
    **dict.fromkeys(("interna", "interno", "internal", "ipogastrica", "ipogastrico", "hypogastric"), "<int>"),
    **dict.fromkeys(("esterna", "esterno", "external"), "<ext>"),
    **dict.fromkeys(("comune", "common"), "<com>"),
    **dict.fromkeys(("profonda", "profondo", "deep"), "<deep>"),
    **dict.fromkeys(("superficiale", "superficial"), "<surf>"),
}
_QUALIFIERS = frozenset(_QUALIFIER_WORDS.values())
# "terzo medio della clavicola" is a third of the bone, not the third of anything.
_FRACTION_PARTS = frozenset(("prossimale", "medio", "distale", "proximal", "middle", "distal"))
_FRACTION_WORDS = frozenset(("terzo", "third"))
# Ambiguous abbreviations never linked by any stage: LM is the middle lobe or the left main coronary.
AMBIGUOUS_ABBREVIATIONS = frozenset(("lm",))
# A "T" level in a sentence about tumour staging is a T stage, not a vertebra.
_STAGING_CUES = frozenset(("stadio", "stadiazione", "stage", "staging", "tnm", "ptnm", "ctnm"))
_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10, "xi": 11, "xii": 12}
_ORDINALS = {
    **{w: i for i, w in enumerate(("primo", "secondo", "terzo", "quarto", "quinto", "sesto", "settimo", "ottavo", "nono", "decimo", "undicesimo", "dodicesimo"), 1)},
    **{w: i for i, w in enumerate(("prima", "seconda", "terza", "quarta", "quinta", "sesta", "settima", "ottava", "nona", "decima", "undicesima", "dodicesima"), 1)},
    **{w: i for i, w in enumerate(("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth"), 1)},
}
_SPINE_REGION = {
    "cervicale": "C", "cervical": "C", "dorsale": "T", "toracica": "T", "toracico": "T", "thoracic": "T", "dorsal": "T",
    "lombare": "L", "lumbar": "L", "sacrale": "S", "sacral": "S",
}
# Words that place a sentence in a region. Used for context-dependent names.
REGION_CUES = {
    "liver": frozenset(("fegato", "liver", "couinaud", "ipocondrio")),
    "spine": frozenset(("rachide", "colonna", "vertebra", "spine", "spinal", "sacro", "sacrum", "sacrale", "sacral",
                        "lombare", "lumbar", "cervicale", "cervical", "midollo", "cord", "canale",
                        "soma", "somatico", "osseo", "ossea", "ossei", "ossee", "osseous", "bone", "litica", "litico",
                        "lytic", "sclerotica", "sclerotico", "sclerotic", "crollo", "frattura", "fracture", "spongiosa")),
    "lung": frozenset(("polmone", "lung", "bronco", "bronchus", "torace", "thorax", "chest", "pleura", "pleurico", "pleural")),
    "heart": frozenset(("cuore", "heart", "coronaria", "coronarico", "coronary", "iva", "lad", "circonflessa", "circumflex")),
    "abdomen": frozenset(("duodeno", "duodenum", "stomaco", "stomach", "addome", "abdomen", "pancreas", "papilla",
                          "diverticolo", "diverticulum", "bulbo", "bulb", "ampolla", "ampulla", "ams", "mesenterica")),
}
# Words of MRI signal: "iperintensa in T2" names a sequence, not a vertebra.
MRI_SIGNAL_CUES = frozenset((
    "segnale", "signal", "intensita", "intensity", "iperintenso", "iperintensa", "iperintensi", "iperintense",
    "ipointenso", "ipointensa", "ipointensi", "ipointense", "isointenso", "isointensa", "hyperintense", "hypointense",
    "isointense", "pesata", "pesate", "pesato", "weighted", "sequenza", "sequenze", "sequence", "sequences",
    "stir", "flair", "dwi", "rm", "mri", "risonanza",
))
# fmt: on
_CONTEXT_WINDOW = 3
_LEVEL = re.compile(r"^([cdtls])(\d{1,2})$")
_ORDINAL_NUMBER = re.compile(r"^(\d{1,2})(?:a|o|°|ª|º|st|nd|rd|th)?$")
_TOKEN = re.compile(
    r"[A-Za-z]\d{1,2}(?![\dA-Za-z])|\d+(?:st|nd|rd|th|a|o|°|ª|º)?|[A-Za-zÀ-ÿ]+\.?|[,/&+]"
)
# "dell'atrio", "L'omero": an elided Italian article or preposition is dropped
# before tokenising, so "L'" can never become "left". English "kidney's" keeps
# the word and loses only the possessive.
_POSSESSIVE = re.compile(r"(?<=\w)['’]s\b")
_ELISION = re.compile(
    r"\b(?:l|un|d|dell|nell|all|dall|sull|coll|pell|quell|quest|sant|tutt)['’]",
    re.IGNORECASE,
)
_RIB_WORDS = re.compile(r"\b(?:cost[ae]|costol[ae]|ribs?)\b", re.IGNORECASE)
# "D 12", "T 12", "Th12" -> one level token.
_SPACED_LEVEL = re.compile(r"\b(?:Th|th|[CDTLSdtls])\s*(\d{1,2})\b")


def _fold(token: str) -> str:
    token = unicodedata.normalize("NFKD", token)
    return "".join(c for c in token if not unicodedata.combining(c)).lower()


def _prepare(text: str) -> str:
    text = _ELISION.sub(" ", _POSSESSIVE.sub("", text))
    if _RIB_WORDS.search(text):
        return text  # "L 5 rib" is the left fifth rib, not L5
    return _SPACED_LEVEL.sub(lambda m: m.group(0)[0].upper() + m.group(1), text)


def _raw_tokens(text: str) -> list[str]:
    return _TOKEN.findall(_prepare(text))


def normalise(
    text: str, *, keep_noise: bool = False, map_words: bool = True
) -> tuple[str, ...]:
    """Canonical tokens of a mention or a name, in both languages.

    Side words become <dx>/<sn>/<bil>, lobe positions <sup>/<mid>/<inf>,
    numbers #n (arabic, ordinal words, roman numerals next to a numbered noun
    or in capitals of two letters or more), vertebral levels @C5 (Th12 is @T12;
    D12 stays @D12, because D1-D4 are also the parts of the duodenum), coordination <and>. ``map_words=False`` keeps adjectives and
    plurals as written, for strict name equivalence.
    """
    raw = _raw_tokens(text)
    folded = [_fold(t).rstrip(".") for t in raw]
    out: list[str] = []
    for index, (original, token) in enumerate(zip(raw, folded, strict=True)):
        neighbours = {folded[j] for j in (index - 1, index + 1) if 0 <= j < len(folded)}
        if token in (",", "/", "&", "+") or token in _COORD:
            out.append(COORDINATION)
        elif original.lower() in ("a.", "art."):
            out.append("<art>")
        elif token in _TYPE_WORDS:
            out.append(_TYPE_WORDS[token])
        elif original in ("R", "L"):
            out.append(SIDE_RIGHT if original == "R" else SIDE_LEFT)
        elif level := _LEVEL.match(token):
            # D stays D: "D2" is a vertebra in a spine report and the second part
            # of the duodenum in an abdominal one. The lexicon says which needs context.
            letter = level.group(1).upper()
            out.append(f"@{letter}{int(level.group(2))}")
        elif token in _FRACTION_WORDS and neighbours & _FRACTION_PARTS:
            continue
        elif token in _QUALIFIER_WORDS:
            out.append(_QUALIFIER_WORDS[token])
        elif number := _ORDINAL_NUMBER.match(token):
            out.append(f"#{int(number.group(1))}")
        elif token in _ROMAN and (
            neighbours & _NUMBERED_NOUNS or (original.isupper() and len(original) >= 2)
        ):
            out.append(f"#{_ROMAN[token]}")
        elif token in _ORDINALS:
            out.append(f"#{_ORDINALS[token]}")
        elif token in _SIDE_WORDS:
            out.append(_SIDE_WORDS[token])
        elif token in _POSITION_WORDS:
            out.append(_POSITION_WORDS[token])
        elif token in _STOP or (token in _NOISE and not keep_noise):
            continue
        elif map_words:
            token = _PLURALS.get(token, token)
            out.append(_ADJECTIVES.get(token, token))
        else:
            out.append(token)
    out = _spine_levels(out)
    while out and out[-1] == COORDINATION:
        out.pop()
    while out and out[0] == COORDINATION:
        out.pop(0)
    return tuple(sorted(set(out)))


def _spine_levels(tokens: list[str]) -> list[str]:
    """ "terza vertebra lombare" -> @L3. Not when a rib is named: "VII costa ... dorsale" stays a rib."""
    regions = [_SPINE_REGION[t] for t in tokens if t in _SPINE_REGION]
    numbers = [t for t in tokens if t.startswith("#")]
    if (
        len(regions) == 1
        and len(numbers) == 1
        and not set(tokens) & (_RIB_NOUNS | _SEGMENT_NOUNS)
    ):
        level = f"@{regions[0]}{numbers[0][1:]}"
        tokens = [t for t in tokens if t not in _SPINE_REGION and t != numbers[0]] + [
            level
        ]
    if any(t.startswith("@") for t in tokens):
        tokens = [t for t in tokens if t != "vertebra"]
    return tokens


def context_tokens(text: str) -> list[str]:
    """Words of a sentence for region cues: folded, adjectives mapped, nothing dropped."""
    out = []
    for token in _raw_tokens(text):
        token = _fold(token).rstrip(".")
        token = _PLURALS.get(token, token)
        out.append(_ADJECTIVES.get(token, token))
    return out


def regions_named(tokens: Iterable[str]) -> set[str]:
    tokens = set(tokens)
    return {region for region, cues in REGION_CUES.items() if cues & tokens}


def region_supported(region: str, mention: str, sentence: str) -> bool:
    """The region is stated next to the mention and no other region is named in the sentence.

    ``spine_no_signal`` is the spine with no MRI-signal word anywhere in the sentence.
    """
    words = context_tokens(sentence)
    if region == "spine_no_signal":
        if MRI_SIGNAL_CUES & set(words):
            return False
        region = "spine"
    named = regions_named(words)
    if named - {region}:
        return False
    cues = REGION_CUES[region]
    if cues & set(context_tokens(mention)):
        return True
    head, _, _ = sentence.partition(":")
    if ":" in sentence and len(head.split()) <= 3 and cues & set(context_tokens(head)):
        return True
    # Within the clause that holds the mention: "S3; fegato nei limiti" is two clauses.
    mention_words = context_tokens(mention)
    for clause in re.split(
        r"[;.,:]|\s(?:e|ed|con|and|with)\s", sentence, flags=re.IGNORECASE
    ):
        clause_words = context_tokens(clause)
        for start in range(len(clause_words) - len(mention_words) + 1):
            if clause_words[start : start + len(mention_words)] == mention_words:
                low = max(0, start - _CONTEXT_WINDOW)
                window = clause_words[
                    low : start + len(mention_words) + _CONTEXT_WINDOW
                ]
                if cues & set(window):
                    return True
    return False


_TNM = re.compile(r"\b[cpyr]?T[0-4][a-d]?\s*N[0-3x]", re.IGNORECASE)


def staging_context(sentence: str) -> bool:
    """A sentence about tumour staging: "T3" there is a T stage."""
    return bool(_STAGING_CUES & set(context_tokens(sentence)) or _TNM.search(sentence))


def level_evidence(mention: str, sentence: str) -> bool:
    """Positive evidence that a level code names a vertebra, not a sequence, stage, root or duodenum.

    T1 and T2 need a vertebra word in the mention itself ("soma di T2"): in a spine
    MRI report a nearby vertebra word does not stop "T2" from being the sequence.
    Other codes accept one within three words, in the same clause.
    """
    mention_words = context_tokens(mention)
    if VERTEBRA_WORDS & set(mention_words):
        return True
    if re.search(r"\bT\s*[12]\b", mention):
        return False
    words_split = re.split(
        r"[;.,:]|\s(?:e|ed|con|and|with)\s", sentence, flags=re.IGNORECASE
    )
    for clause in words_split:
        clause_words = context_tokens(clause)
        for start in range(len(clause_words) - len(mention_words) + 1):
            if clause_words[start : start + len(mention_words)] == mention_words:
                low = max(0, start - _CONTEXT_WINDOW)
                if VERTEBRA_WORDS & set(
                    clause_words[low : start + len(mention_words) + _CONTEXT_WINDOW]
                ):
                    return True
    return False


# --------------------------------------------------------------------------
# Attributes and the deterministic integration check
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Attributes:
    sides: frozenset[str]
    numbers: frozenset[str]
    positions: frozenset[str]
    qualifiers: frozenset[str]
    coordination: bool
    number_kind: str | None
    types: frozenset[str] = frozenset()

    @classmethod
    def of(cls, tokens: Iterable[str]) -> "Attributes":
        tokens = set(tokens)
        numbers = frozenset(t for t in tokens if t[:1] in "#@")
        if tokens & _SEGMENT_NOUNS or (
            any(t.startswith("@S") for t in numbers) and tokens & {"fegato", "liver"}
        ):
            kind = "segment"
        elif tokens & _RIB_NOUNS:
            kind = "rib"
        elif any(t.startswith("@") for t in numbers):
            kind = "vertebra"
        else:
            kind = None
        return cls(
            sides=frozenset(tokens & {SIDE_RIGHT, SIDE_LEFT, SIDE_BOTH}),
            numbers=numbers,
            positions=frozenset(tokens & {POS_UP, POS_MID, POS_DOWN}),
            qualifiers=frozenset(tokens & _QUALIFIERS),
            coordination=COORDINATION in tokens,
            number_kind=kind if numbers else None,
            types=frozenset(tokens & STRUCTURE_TYPES),
        )


def _base(tokens: Iterable[str]) -> tuple[str, ...]:
    return tuple(t for t in tokens if t not in (SIDE_RIGHT, SIDE_LEFT, SIDE_BOTH))


def _bare(tokens: Iterable[str]) -> tuple[str, ...]:
    """A name without side, numbers, positions or qualifiers: its structure words."""
    drop = {
        SIDE_RIGHT,
        SIDE_LEFT,
        SIDE_BOTH,
        POS_UP,
        POS_MID,
        POS_DOWN,
        COORDINATION,
    } | _QUALIFIERS
    return tuple(t for t in tokens if t not in drop and t[:1] not in "#@")


def _class_kind(cid: str) -> str | None:
    if cid.startswith("rib_"):
        return "rib"
    if cid.startswith("vertebrae_"):
        return "vertebra"
    if cid.startswith("liver_segment_"):
        return "segment"
    return None


@dataclass(frozen=True)
class Candidate:
    """A concept the linker may choose: a TotalSegmentator class or an ontology term."""

    cid: str
    label: str
    names: tuple[tuple[str, ...], ...]
    is_target_class: bool

    @property
    def attributes(self) -> Attributes:
        return Attributes.of(t for name in self.names for t in name)

    @property
    def number_kind(self) -> str | None:
        if self.is_target_class:
            return _class_kind(self.cid)
        return self.attributes.number_kind


def verify(
    mention_tokens: Sequence[str],
    candidate: Candidate,
    sided_bases: frozenset[tuple[str, ...]],
) -> str | None:
    """The reason the candidate contradicts the mention, or None if nothing does."""
    mention = Attributes.of(mention_tokens)
    if mention.coordination or SIDE_BOTH in mention.sides:
        return "mention_names_more_than_one_structure"
    if len(mention.sides) > 1:
        return "mention_has_two_sides"
    if len([n for n in mention.numbers if n.startswith("@")]) > 1:
        return "mention_spans_several_levels"
    if set(mention_tokens) & _NON_STRUCTURE_NUMBERED and mention.numbers:
        return "number_refers_to_a_root_disc_or_foramen"
    cand = candidate.attributes
    if mention.sides:
        if cand.sides and cand.sides != mention.sides:
            return "side_mismatch"
        if not cand.sides and any(
            _base(name) in sided_bases for name in candidate.names
        ):
            return "candidate_lacks_the_side_the_mention_states"
    elif cand.sides and any(_base(name) in sided_bases for name in candidate.names):
        return "side_not_stated_in_mention"
    if mention.numbers:
        if not cand.numbers:
            return "candidate_lacks_the_number_the_mention_states"
        if not mention.numbers <= cand.numbers:
            return "number_or_level_mismatch"
        if mention.number_kind != candidate.number_kind:
            # "quinto metatarso" is not a rib; "XII" alone names no numbered class.
            return "number_belongs_to_another_kind_of_structure"
    elif cand.numbers:
        return "number_not_stated_in_mention"
    if mention.positions and cand.positions and mention.positions != cand.positions:
        return "position_mismatch"
    if cand.positions and not mention.positions:
        return "position_not_stated_in_mention"
    if mention.qualifiers != cand.qualifiers:
        return "qualifier_names_a_different_structure"
    if mention.types and not mention.types <= cand.types:
        return "structure_type_mismatch"
    return None


# --------------------------------------------------------------------------
# Recognition: the lexicon
# --------------------------------------------------------------------------


@dataclass
class Lexicon:
    """Exact recognition of known names, with context-dependent names marked."""

    index: dict[tuple[str, ...], list[tuple[str, str | None]]] = field(
        default_factory=dict
    )
    strict: dict[tuple[str, ...], set[str]] = field(default_factory=dict)
    classes: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Lexicon":
        lexicon = cls(classes=data["classes"])
        index: dict[tuple[str, ...], list[tuple[str, str | None]]] = defaultdict(list)
        strict: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for cid, entry in data["classes"].items():
            requires = entry.get("requires_context", {})
            for name in entry["it"] + entry["en"]:
                region = requires.get(name)
                key = normalise(name)
                if (cid, region) not in index[key]:
                    index[key].append((cid, region))
                if region is None:
                    strict[normalise(name, keep_noise=True, map_words=False)].add(cid)
        lexicon.index = dict(index)
        lexicon.strict = dict(strict)
        return lexicon

    def recognise(self, mention: str, sentence: str) -> tuple[list[str], str]:
        """Classes the mention names, after context; and how they were found."""
        hits = self.index.get(normalise(mention), [])
        if not hits:
            return [], "unknown_name"
        kept = sorted(
            {
                cid
                for cid, region in hits
                if region is None or region_supported(region, mention, sentence)
            }
        )
        if not kept:
            return [], "context_does_not_support_the_name"
        return kept, "recognised" if len(kept) == 1 else "ambiguous_name"

    def candidates(self) -> list[Candidate]:
        out = []
        for cid, entry in self.classes.items():
            label = f"{entry['it'][0]} / {entry['en'][0]}"
            names = tuple(
                dict.fromkeys(normalise(n) for n in entry["it"] + entry["en"])
            )
            out.append(
                Candidate(cid=cid, label=label, names=names, is_target_class=True)
            )
        return out

    def resolve_equivalent(self, name: str) -> str | None:
        """The target class with exactly this name: noise words kept, no adjective mapping."""
        hits = self.strict.get(normalise(name, keep_noise=True, map_words=False), set())
        return next(iter(hits)) if len(hits) == 1 else None


# --------------------------------------------------------------------------
# Ontology distractors (UBERON human anatomy)
# --------------------------------------------------------------------------


def load_obo_terms(
    lines: Iterable[str], *, require_xref_prefix: str = "FMA:"
) -> list[dict[str, Any]]:
    """Non-obsolete terms with an xref of the given prefix (FMA: human anatomy)."""
    terms: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in lines:
        line = raw.rstrip("\n")
        if line == "[Term]":
            current = {"synonyms": [], "xrefs": []}
            terms.append(current)
            continue
        if line.startswith("["):
            current = None
            continue
        if current is None:
            continue
        if line.startswith("id: "):
            current["id"] = line[4:].strip()
        elif line.startswith("name: "):
            current["name"] = line[6:].strip()
        elif line.startswith("synonym: "):
            match = re.match(r'synonym: "(.*)" EXACT', line)
            if match:
                current["synonyms"].append(match.group(1))
        elif line.startswith("xref: "):
            current["xrefs"].append(line[6:].split()[0])
        elif line.startswith("is_obsolete: true"):
            current["obsolete"] = True
    return [
        t
        for t in terms
        if t.get("id")
        and t.get("name")
        and not t.get("obsolete")
        and any(x.startswith(require_xref_prefix) for x in t["xrefs"])
    ]


def build_pool(
    lexicon: Lexicon, ontology_terms: Sequence[dict[str, Any]] = ()
) -> tuple[list[Candidate], dict[str, str]]:
    """Target classes plus ontology distractors; and which distractors are a target class by name.

    Equivalence uses the term's primary name only, compared strictly, so that
    "cranial muscle" can never become the skull.
    """
    pool = lexicon.candidates()
    equivalent: dict[str, str] = {}
    for term in ontology_terms:
        names = tuple(
            dict.fromkeys(
                normalise(n) for n in [term["name"], *term.get("synonyms", ())]
            )
        )
        pool.append(
            Candidate(
                cid=term["id"], label=term["name"], names=names, is_target_class=False
            )
        )
        target = lexicon.resolve_equivalent(term["name"])
        if target:
            equivalent[term["id"]] = target
    return pool, equivalent


def sided_bases(pool: Sequence[Candidate]) -> frozenset[tuple[str, ...]]:
    """Names that exist with a side somewhere in the pool ("kidney" has "left kidney")."""
    return frozenset(
        _base(name) for c in pool for name in c.names if Attributes.of(name).sides
    )


# Kept for callers of the first version.
lateralised_bases = sided_bases


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------

ChatFn = Callable[[str], str]
Retriever = Callable[[str, str, int], list[str]]

ACCEPTED = "accepted"
ABSTAINED = "abstained"


@dataclass
class LinkResult:
    status: str
    cid: str | None
    stage: str
    reason: str
    options: list[str] = field(default_factory=list)
    votes: dict[str, str | None] = field(default_factory=dict)


def _choice_prompt(mention: str, sentence: str, labels: Sequence[str]) -> str:
    options = "\n".join(f"{i}. {label}" for i, label in enumerate(labels, start=1))
    return (
        "You are a radiologist. Link the expression to the anatomical structure it names in this "
        "report sentence. Answer with the option number only. Answer 0 if no option is exactly "
        "that structure, if the expression names more than one structure, or if you are not sure.\n\n"
        f"Sentence: {sentence}\nExpression: {mention}\n\nOptions:\n{options}\n0. none / not sure\n\nNumber:"
    )


def _parse(answer: str, n: int) -> int | None:
    match = re.search(r"-?\d+", answer or "")
    if not match:
        return None
    value = int(match.group())
    return value if 0 <= value <= n else None


@dataclass
class AnatomyLinker:
    lexicon: Lexicon
    pool: list[Candidate]
    equivalent: dict[str, str] = field(default_factory=dict)
    retriever: Retriever | None = None
    chats: dict[str, ChatFn] = field(default_factory=dict)
    max_options: int = 10
    nearest: int = 10
    closest: int = 5

    def __post_init__(self) -> None:
        self._by_id = {c.cid: c for c in self.pool}
        self._sided = sided_bases(self.pool)
        # Words that are by themselves the name of a class ("rene", "femore",
        # "kidney"). A mention containing one names that structure or a part of it.
        self._anchors = frozenset(
            bare[0] for key in self.lexicon.index if len(bare := _bare(key)) == 1
        )

    def _anchored(self, tokens: Sequence[str], candidate: Candidate) -> bool:
        """ "right kidney's upper pole" names the kidney: the right upper lung lobe is not it."""
        named = set(tokens) & self._anchors
        if not named:
            return True
        return bool(named & {t for name in candidate.names for t in name})

    def _context_allows(
        self, candidate: Candidate, mention: str, sentence: str
    ) -> bool:
        """The same expectation the lexicon applies, for candidates found by retrieval."""
        cid = self.equivalent.get(candidate.cid, candidate.cid)
        if _class_kind(cid) == "segment" and not region_supported(
            "liver", mention, sentence
        ):
            return False
        if _class_kind(cid) == "vertebra" and any(
            t.startswith("@") for t in normalise(mention)
        ):
            return level_evidence(mention, sentence)
        return True

    def _link_level(self, code: str, mention: str, sentence: str) -> LinkResult:
        """A bare level code: a vertebra only with vertebra evidence, a liver segment only with the liver named."""
        letter, number = code[1], int(code[2:])
        vertebra = f"vertebrae_{'T' if letter == 'D' else letter}{number}"
        readings = []
        if vertebra in self.lexicon.classes and level_evidence(mention, sentence):
            readings.append(vertebra)
        segment = f"liver_segment_{number}"
        if (
            letter == "S"
            and segment in self.lexicon.classes
            and region_supported("liver", mention, sentence)
        ):
            readings.append(segment)
        if len(readings) == 1:
            return LinkResult(
                ACCEPTED, readings[0], "lexicon", "level_code_with_evidence"
            )
        reason = (
            "level_code_reads_two_ways" if readings else "level_code_without_evidence"
        )
        return LinkResult(ABSTAINED, None, "lexicon", reason, options=readings)

    def link(self, mention: str, sentence: str) -> LinkResult:
        tokens = normalise(mention)
        attributes = Attributes.of(tokens)
        if attributes.coordination or SIDE_BOTH in attributes.sides:
            return LinkResult(
                ABSTAINED, None, "integration", "mention_names_more_than_one_structure"
            )
        if attributes.numbers and set(tokens) & _NON_STRUCTURE_NUMBERED:
            return LinkResult(
                ABSTAINED,
                None,
                "integration",
                "number_refers_to_a_root_disc_or_foramen",
            )
        if set(tokens) & AMBIGUOUS_ABBREVIATIONS:
            return LinkResult(ABSTAINED, None, "integration", "ambiguous_abbreviation")
        if any(t.startswith("@T") for t in tokens) and staging_context(sentence):
            return LinkResult(ABSTAINED, None, "integration", "t_stage_not_a_vertebra")
        if set(tokens) & CONTAINER_WORDS:
            return LinkResult(
                ABSTAINED, None, "integration", "mention_names_a_space_not_an_organ"
            )
        if len(tokens) == 1 and tokens[0].startswith("@"):
            return self._link_level(tokens[0], mention, sentence)

        recognised, how = self.lexicon.recognise(mention, sentence)
        if how == "recognised":
            return LinkResult(ACCEPTED, recognised[0], "lexicon", how)
        if how in ("ambiguous_name", "context_does_not_support_the_name"):
            return LinkResult(ABSTAINED, None, "lexicon", how, options=recognised)

        if self.retriever is None or len(self.chats) < 2:
            return LinkResult(
                ABSTAINED, None, "lexicon", "unknown_name_and_no_deliberation_stage"
            )

        # Only near candidates are offered. When the closest ones are all rejected,
        # what is left is far from the mention, and two models agreeing on a far
        # option is exactly the correlated error this design does not trust.
        ranked = list(dict.fromkeys(self.retriever(mention, sentence, self.nearest)))[
            : self.nearest
        ]
        options = [
            cid
            for cid in ranked
            if (candidate := self._by_id.get(cid))
            and verify(tokens, candidate, self._sided) is None
            and self._anchored(tokens, candidate)
            and self._context_allows(candidate, mention, sentence)
        ][: self.max_options]
        if not options:
            return LinkResult(
                ABSTAINED, None, "integration", "no_candidate_survives_the_checks"
            )
        if not set(options) & set(ranked[: self.closest]):
            return LinkResult(
                ABSTAINED,
                None,
                "integration",
                "closest_candidates_all_rejected",
                options,
            )

        labels = [self._by_id[c].label for c in options]
        prompt = _choice_prompt(mention, sentence, labels)
        votes: dict[str, str | None] = {}
        for name, chat in self.chats.items():
            choice = _parse(chat(prompt), len(options))
            votes[name] = options[choice - 1] if choice else None
        chosen = set(votes.values())
        if None in chosen or len(chosen) != 1:
            reason = "a_model_was_not_sure" if None in chosen else "models_disagree"
            return LinkResult(ABSTAINED, None, "deliberation", reason, options, votes)
        cid = chosen.pop()
        candidate = self._by_id[cid]
        if verify(tokens, candidate, self._sided) or not self._context_allows(
            candidate, mention, sentence
        ):
            return LinkResult(
                ABSTAINED,
                None,
                "integration",
                "choice_fails_the_checks",
                options,
                votes,
            )  # pragma: no cover
        resolved = cid if candidate.is_target_class else self.equivalent.get(cid, cid)
        return LinkResult(
            ACCEPTED,
            resolved,
            "deliberation",
            "models_agree_and_checks_pass",
            options,
            votes,
        )
