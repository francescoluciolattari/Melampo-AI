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
   or an MRI sequence. Such names are accepted only when the region is stated
   next to them (in the mention, a section heading, or a few words away) and
   no other region is named in the sentence. Otherwise the linker abstains.
   The same holds for a written form with several meanings ("GB": gallbladder
   in English, white cell count in Italian clinical text; "LM"; "ponte"):
   every meaning competes on the evidence of the sentence, its numbers and
   units and its language, and the structure is linked only when its meaning
   wins by a margin (`word_senses`, data in `data/linking/word_senses.json`).
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

**Part-of knowledge and translation (added 2026-10-05).** A reader knows that the
sigmoid is part of the colon; that is a curated table (`anatomy_parts`), and the
link says so (`relation`: equal, part_of, contour_of, approx). Tissue and space
words are never dropped silently: "parete aortica" is a part of the aorta, "lume
esofageo" and "muscolo sternale" are not linked. For Italian mentions the ontology
has no names, so the two models translate the mention to a standard English term
(the mention is marked with <tgt> tags, as in BioELX 2026); the term is trusted only if
it equals a pool name word for word, both models reach the same single concept, and
side, number, position, qualifier, part words and organ words of the mention are
all still in it.

**Known limits, stated rather than hidden.**
Presence is not linking: "assenza del rene destro" links the right kidney, and the
absence is a polarity attribute that Level 1 must carry. Two models can still
translate or choose the same wrong term for a structure outside the table and the
segmentation classes; the deterministic guards narrow that, they do not remove it.
Parts that the table does not list abstain. Spaces ("loggia renale", "ipocondrio")
are never linked to the organ, because after surgery the space is empty. Bare level
codes ("L5", "D2", "T2", "S1") are accepted only with positive evidence; T1/T2 only
with a vertebra word in the mention, since in an MRI report they are sequences.


**Streams, not a cascade (7 October 2026).** The cheap readers above (senses, integration checks,
level codes, lexicon, part table) are evaluated independently and all of them are kept in
``LinkResult.trace``; ``_decide`` weighs them in the same order of authority as before, so every
decision is unchanged (858 of 858 bench decisions identical, deterministic and with fake models).
The two models are asked in parallel. With ``graph`` (``anatomy_graph.AnatomyGraph``) the models'
choice is also checked against its graph neighbours and a finer UBERON term gets a proposed parent
class (``fallback``), applied only when ``accept_parent_fallback`` is set.

**Area, profile and roles (T3, 7 October 2026, night).** The area of the exam (``exam_area``, the
region the exam's name says it studies) is an expectation: a structure of the area supports the
link, a far one is a prediction error that sends an ambiguous form to the models (or to an
abstention without them). Every accepted link carries the independent mechanisms behind it
(``support``), the conflicts left (``conflicts``) and their difference (``convergence``), the score
the gold set will calibrate with Learn-then-Test (``evaluation.selective_calibration``). ``role``
says whether the structure is the site of a procedure ("liver biopsy") or the structure a
measurement is about ("heart rate": no link, ``about`` keeps the structure)."""

import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from melampo.memory.compounds import (
    defined_abbreviation,
    defined_short_form,
    hyphen_compound,
)
from melampo.memory.development import DevelopmentalFrame
from melampo.memory.chunk_lattice import ChunkLattice
from melampo.memory.longer_names import LongerNames
from melampo.memory.morphology import Morphology
from melampo.memory.exam_area import ADJACENT, EXPECTED, OUTSIDE, ExamAreas
from melampo.memory.exam_frame import IMAGING, ExamFrames
from melampo.memory.report_state import (
    CLINICAL,
    CONCLUSIONS,
    FINDINGS,
    TECHNIQUE,
    ReportState,
)
from melampo.memory.word_senses import SenseInventory, detect_language, locate

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
# Languages whose abbreviations a sentence of the key language accepts as its own.
_TOLERATED_LANGUAGES = {"it": frozenset({"en"})}
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
    **dict.fromkeys(("ilo", "ili", "hilum", "hila"), "<hilum>"),
    # "hilar lymph node" is a node at the hilum, not "hilum of lymph node", a part of a node.
    **dict.fromkeys(("ilare", "ilari", "hilar"), "<hilar>"),
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


# Nouns that only say "the area of": "heart region", "aortic part". A longer name that adds one of
# them is the same structure, not another.
_PLACE_NOUNS = frozenset(
    "region regions area areas part parts portion portions zone zones site sites section".split()
)


def _ordered(text: str) -> tuple[str, ...]:
    """The words of a text, each normalised on its own, in the order written (``normalise`` makes a
    bag). A word that normalises to nothing (a function word: "of", "and") stays as written, so
    "heart and stomach" is not "heart of stomach"."""
    out = []
    for word in _RAW_WORD.findall(text):
        tokens = normalise(word, keep_noise=True)
        out.append(tokens[0] if tokens else word.lower())
    return tuple(out)


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


# What a level code names when one of these is in the sentence: a root, a disc, a foramen, the
# spaces between vertebrae ("spazi intersomatici del tratto C3-C7"), a nerve. Stems, so plurals and
# compounds ("disco-artrosico", "intersomatici") are caught too.
_NOT_A_VERTEBRA_STEMS = (
    "radic", "disc", "foram", "interso", "intervert", "spazi", "space", "nerv", "gangl",
)  # fmt: skip


def level_from_report(
    code: str, sentence: str, report: ReportState | None, at: int | None
) -> bool:
    """The report, read from the top, says the spine is what is being described.

    Prediction from above: in a report whose clinical question or technique names the spine and
    no other region, a level code in a finding sentence is a vertebra ("Anterolistesi di L4 su
    L5"). Not for T1 and T2 (sequences even there), not in the header itself, and never when the
    sentence speaks of MRI signal, tumour staging, a root, a disc or a foramen, or names another
    region.
    """
    if report is None or at is None or not report.spine_scope:
        return False
    if report.section_at(at) in (CLINICAL, TECHNIQUE, "label"):
        return False
    if code in ("@T1", "@T2"):
        return False
    words = set(context_tokens(sentence))
    return not (
        words & MRI_SIGNAL_CUES
        or any(w.startswith(_NOT_A_VERTEBRA_STEMS) for w in words)
        or staging_context(sentence)
        or regions_named(words) - {"spine"}
    )


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
    # The same names with the tissue and container words kept ("wall of appendix").
    strict_names: tuple[tuple[str, ...], ...] = ()

    @property
    def attributes(self) -> Attributes:
        return Attributes.of(t for name in self.names for t in name)

    @property
    def number_kind(self) -> str | None:
        if self.is_target_class:
            return _class_kind(self.cid)
        return self.attributes.number_kind


_NOISE_CANON = {
    "parete": "wall", "pareti": "wall", "wall": "wall",
    "parenchima": "parenchyma", "parenchyma": "parenchyma",
    "lume": "lumen", "lumen": "lumen",
    "ghiandola": "gland", "gland": "gland",
    "muscolo": "muscle", "muscoli": "muscle", "muscle": "muscle", "muscles": "muscle",
    "osso": "bone", "bone": "bone",
    "corpo": "body", "soma": "body", "body": "body",
    "livello": "level", "corrispondenza": "level", "level": "level",
}  # fmt: skip


def _content(tokens: Iterable[str]) -> frozenset[str]:
    """The words that say which structure it is: no side, position, number, qualifier or type.

    Tissue and container words are kept ("wall of appendix" is not the appendix).
    """
    skip = {SIDE_RIGHT, SIDE_LEFT, SIDE_BOTH, POS_UP, POS_MID, POS_DOWN, COORDINATION}
    skip |= _QUALIFIERS | STRUCTURE_TYPES
    return frozenset(
        _NOISE_CANON.get(t, t) for t in tokens if t not in skip and t[:1] not in "#@"
    )


# Ontology homonyms: "lingula" is also a cerebellar lobule, "bulb" a part of the eye or the brain.
# A candidate from the nervous system is offered only when the sentence is about it.
_NEURO_WORDS = frozenset(("cerebellum", "cerebellar", "cerebral", "brain", "brainstem", "cortex", "cortical", "medulla", "neuron", "nucleus", "gyrus", "ganglion", "tract", "lobule", "vermis", "thalamus", "hippocampus"))  # fmt: skip
_NEURO_CUES = frozenset(("encefalo", "cervello", "cerebrale", "cerebellare", "cerebellum", "cerebral", "brain", "cranio", "cranico", "neurologico", "neuro", "sistema", "nervoso", "corteccia", "cortical", "cortex", "talamo", "ippocampo", "ventricolo", "ventricoli"))  # fmt: skip


def wrong_system(candidate: Candidate, sentence: str) -> bool:
    if candidate.is_target_class:
        return False
    words = {w for name in candidate.strict_names or candidate.names for w in name}
    if not words & _NEURO_WORDS:
        return False
    return not {_fold(t) for t in _raw_tokens(sentence)} & _NEURO_CUES


def covers(mention: str, candidate: Candidate) -> bool:
    """True when one name of the candidate says exactly what the mention says.

    Two models agreeing on a candidate is not evidence that the words match:
    "porta hepatis" is not the portal vein, "cardiac silhouette" is not a
    cardiac chamber, "right ribs" is not the true ribs. A word the candidate
    does not have, or a word only the candidate has, means the candidate is a
    different (wider or narrower) concept, and the link is not made.
    """
    wanted = _content(normalise(mention, keep_noise=True))
    names = candidate.strict_names or candidate.names
    return bool(wanted) and any(_content(name) == wanted for name in names)


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
    # Names with their tissue and container words kept: "lume esofageo" is not a name of the esophagus.
    noisy: frozenset[tuple[str, ...]] = frozenset()
    # Which language list(s) of the lexicon ("it", "en") each key comes from.
    languages: dict[tuple[str, ...], frozenset[str]] = field(default_factory=dict)
    # Keys written as names with no tissue word at all ("tiroide", "femore"). A key that exists
    # only because a name lost its head noun ("osso dell'anca" -> "anca", "left innominate bone"
    # -> "left innominate") is not a name: the head noun is what the name refers to.
    headed: frozenset[tuple[str, ...]] = frozenset()

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
        written: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for entry in data["classes"].values():
            for language in ("it", "en"):
                for name in entry[language]:
                    written[normalise(name)].add(language)
        lexicon.languages = {k: frozenset(v) for k, v in written.items()}
        lexicon.index = dict(index)
        lexicon.strict = dict(strict)
        lexicon.noisy = frozenset(
            normalise(name, keep_noise=True)
            for entry in data["classes"].values()
            for name in entry["it"] + entry["en"]
        )
        lexicon.headed = frozenset(
            normalise(name)
            for entry in data["classes"].values()
            for name in entry["it"] + entry["en"]
            if normalise(name, keep_noise=True) == normalise(name)
        )
        return lexicon

    def abbreviation_clash(self, mention: str, language: str | None) -> bool:
        """A short form written in the lexicon for the other language only, in a sentence that is
        plainly in this one: "lid" (LID, lobo inferiore destro) in "the lid fissure".

        Abbreviations collide with ordinary words of the other language far more than full names
        do, and the language of the surrounding text is what tells them apart. Only short forms
        (three letters or fewer, or capitals of two to four letters) are held to it, and only when
        the language of the sentence is known: a name spelled the same in both languages, or a
        sentence the language of which is not clear, never clashes. The rule is not symmetric:
        Italian reports use English abbreviations as a matter of course (CCA, SVC, IVC), English
        reports do not use Italian ones, so an English-only name in an Italian sentence is
        tolerated (``_TOLERATED_LANGUAGES``), an Italian-only name in an English sentence is not.
        """
        if language is None:
            return False
        words = [w for w in re.findall(r"[A-Za-zÀ-ÿ]+", mention)]
        content = [
            w
            for w in words
            if _fold(w) not in _SIDE_WORDS and _fold(w) not in _POSITION_WORDS
        ]
        if len(content) != 1:
            return False
        word = content[0]
        if not (len(word) <= 3 or (word.isupper() and len(word) <= 4)):
            return False
        written = self.languages.get(normalise(mention), frozenset())
        accepted = {language} | _TOLERATED_LANGUAGES.get(language, frozenset())
        return bool(written) and not (written & accepted)

    def name_clash(self, mention: str, language: str | None) -> bool:
        """A full name written in the lexicon for the other language only, in a sentence plainly of
        a language that does not borrow it: "sigma" (sigma, the sigmoid colon in Italian) in an
        English sentence is the Greek letter, a sigma factor or a standard deviation (85 of 85 such
        links in the CRAFT corpus, 8 October 2026). The same asymmetry as for short forms: English
        names in Italian reports are tolerated."""
        if language is None:
            return False
        written = self.languages.get(normalise(mention), frozenset())
        accepted = {language} | _TOLERATED_LANGUAGES.get(language, frozenset())
        return bool(written) and not (written & accepted)

    def recognise(
        self, mention: str, sentence: str, headless_ok: bool = False
    ) -> tuple[list[str], str]:
        """Classes the mention names, after context; and how they were found.

        ``headless_ok``: the sense inventory has read the form in its sentence (it has a sense
        profile, and the sentence chose the anatomical sense), so the context supplies the head
        noun the mention lacks ("atrophy of the left paraspinal" is the muscle)."""
        hits = self.index.get(normalise(mention), [])
        if not hits:
            return [], "unknown_name"
        with_tissue = normalise(mention, keep_noise=True)
        if with_tissue != normalise(mention) and with_tissue not in self.noisy:
            # "lume esofageo", "parete aortica", "muscolo sternale": a tissue or a space of the
            # organ, not the organ. Dropping the word would report a part as the whole.
            return [], "unknown_name"
        if (
            not headless_ok
            and with_tissue == normalise(mention)
            and normalise(mention) not in self.headed
        ):
            # The mention is a written name without its head noun: "anca" is the hip (a region,
            # UBERON:0001464), "osso dell'anca" the hip bone; "left innominate" is the vein or the
            # artery as often as the bone; "paravertebrale" is a region. The head of a noun phrase
            # says what it refers to, so the shortened form is left to the other streams.
            return [], "name_without_its_head_noun"
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
            strict = tuple(
                dict.fromkeys(
                    normalise(n, keep_noise=True) for n in entry["it"] + entry["en"]
                )
            )
            out.append(
                Candidate(
                    cid=cid,
                    label=label,
                    names=names,
                    is_target_class=True,
                    strict_names=strict,
                )
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
        strict = tuple(
            dict.fromkeys(
                normalise(n, keep_noise=True)
                for n in [term["name"], *term.get("synonyms", ())]
            )
        )
        pool.append(
            Candidate(
                cid=term["id"],
                label=term["name"],
                names=names,
                is_target_class=False,
                strict_names=strict,
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


class ModelUnavailable(RuntimeError):
    """A model or the retriever did not answer (rate limit, network). Never a guess: the link abstains."""


@dataclass(frozen=True)
class Evidence:
    """What one stream said about the mention, independently of the others.

    ``verdict`` is ``support`` (this stream links ``cid``), ``veto`` (this stream rules the mention or
    the candidate out, with ``reason``), ``abstain`` (the stream looked and could not decide) or
    ``silent`` (the stream has nothing to say about this mention).
    """

    stream: str
    verdict: str
    cid: str | None = None
    reason: str = ""
    relation: str = "equal"
    part: str = ""
    options: tuple[str, ...] = ()


SUPPORT, VETO, UNDECIDED, SILENT = "support", "veto", "abstain", "silent"
# Evidence against the hypothesis that is not a veto: a prediction error (a structure far from the
# area of the exam). It is recorded, lowers the convergence and sends an ambiguous form to the models.
AGAINST = "against"


@dataclass
class LinkResult:
    status: str
    cid: str | None
    stage: str
    reason: str
    options: list[str] = field(default_factory=list)
    votes: dict[str, str | None] = field(default_factory=dict)
    relation: str = "equal"  # equal, part_of, kind_of, contour_of, approx
    part: str = ""
    # The class the graph proposes for a link to a finer UBERON term, when it is not applied
    # (accept_parent_fallback=False): a suggestion for the review queue, not a link.
    fallback: str = ""
    # Every stream's evidence, in the order the decision considered it (audit trail).
    trace: list[Evidence] = field(default_factory=list, compare=False)
    # The independent mechanisms that support the accepted class and the conflicts found (T3,
    # 7 Oct 2026). Mechanisms: name (lexicon, part table, level code), sense (an ambiguous form read
    # in its sentence), frame (the sentence is imaging findings), area (the structure belongs to the
    # area of the exam), models (the two models' choice or their check in context; one mechanism,
    # not two: their errors are correlated). ``convergence`` = mechanisms - conflicts is the score
    # the gold set will calibrate (Learn-then-Test, ``selective_calibration``); it decides nothing yet.
    support: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    # The role of the structure in the sentence, SNOMED CT style: "procedure_site" (a biopsy, a
    # resection of it), "inherent_location" (a measurement of it: "heart rate", "funzione del
    # fegato"; the link abstains, ``about`` says which structure was measured), "" otherwise.
    role: str = ""
    about: str | None = None
    # How the chunk lattice read the phrase ("role:inherent_location:heart size"), when it is on.
    block: str = ""
    # How the construction-integration reader read it ("decision:top reading:margin"), when it is on.
    reading: str = ""
    # Structures the Greek and Latin roots of the mention point to (stage 3): a proposal for the
    # review queue on an abstention, never a link.
    proposals: tuple[str, ...] = ()

    @property
    def convergence(self) -> int:
        return len(self.support) - len(self.conflicts)


_MEASUREMENT_VETOES = (
    "frame_is_a_measurement",
    "attribute_head_names_a_measurement",
    "process_head_names_a_process",
)
_MODEL_STAGES = frozenset(("deliberation", "translation"))

_NO_OPTION_REASONS = frozenset(
    ("no_candidate_survives_the_checks", "closest_candidates_all_rejected")
)


_VERDICT_WORD = re.compile(r"\b(YES|NO|UNSURE)\b")


def _verdict(reply: str) -> str:
    """YES, NO or UNSURE from a model's reply. The first word is not enough ("Word: YES" is a yes):
    the verdict is the one of the three words the reply uses; none, or two different, is UNSURE."""
    found = set(_VERDICT_WORD.findall((reply or "").upper()))
    return found.pop() if len(found) == 1 else "UNSURE"


_CHOICE_NUMBER = re.compile(r"\b([1-5])\b")


def _choice_verdict(reply: str) -> str:
    """1 is YES, 2 to 4 are NO, 5 (or no single number) is UNSURE."""
    found = set(_CHOICE_NUMBER.findall(reply or ""))
    if len(found) != 1:
        return "UNSURE"
    number = found.pop()
    return {"1": "YES", "5": "UNSURE"}.get(number, "NO")


def _mark(sentence: str, mention: str) -> str:
    """The sentence with the mention between <tgt> tags (mention-anchored prompting, BioELX 2026)."""
    match = re.search(re.escape(mention.strip()), sentence, flags=re.IGNORECASE)
    if not match:
        return f"{sentence} <tgt>{mention}</tgt>"
    return f"{sentence[: match.start()]}<tgt>{match.group()}</tgt>{sentence[match.end() :]}"


def _translate_prompt(mention: str, sentence: str) -> str:
    return (
        "You are a radiologist and a translator. In the report sentence the expression between "
        "<tgt> tags names an anatomical structure. Give its standard English anatomical term. "
        "Keep side, number and level exactly as written. Do not make it broader or narrower, and do "
        "not add words. If it is not an anatomical structure, names more than one, or you are not "
        "sure, answer UNSURE. Answer with the term only.\n\n"
        f"Sentence: {_mark(sentence, mention)}\nTerm:"
    )


def _clean_term(answer: str) -> str | None:
    line = (answer or "").strip().splitlines()[0:1]
    term = (line[0] if line else "").strip().strip("\"'`*.").strip()
    if not term or term.upper().startswith("UNSURE") or len(term.split()) > 8:
        return None
    return term


# Part and tissue words, Italian and English, as one id: a translation must keep exactly the ones
# the mention has ("collo dell'utero" is not "body of uterus", "corpo" is not dropped).
_PART_IDS = {
    **{w: "head" for w in ("testa", "head")}, **{w: "tail" for w in ("coda", "tail")},
    **{w: "pole" for w in ("polo", "pole")}, **{w: "lobe" for w in ("lobo", "lobe")},
    **{w: "neck" for w in ("collo", "neck", "cervice", "cervix")},
    **{w: "fundus" for w in ("fondo", "fundus")},
    **{w: "shaft" for w in ("diafisi", "shaft", "diaphysis")},
    **{w: "isthmus" for w in ("istmo", "isthmus")}, **{w: "apex" for w in ("apice", "apex")},
    **{w: "dome" for w in ("cupola", "dome")}, **{w: "base" for w in ("base",)},
    **{w: "body" for w in ("corpo", "body", "soma")}, **{w: "wall" for w in ("parete", "pareti", "wall")},
    **{w: "lumen" for w in ("lume", "lumen")}, **{w: "parenchyma" for w in ("parenchima", "parenchyma")},
    **{w: "process" for w in ("processo", "process")}, **{w: "wing" for w in ("ala", "wing")},
    **{w: "trunk" for w in ("tronco", "trunk")}, **{w: "branch" for w in ("ramo", "branch")},
    **{w: "segment" for w in ("segmento", "segment")}, **{w: "root" for w in ("radice", "root")},
    **{w: "horn" for w in ("corno", "horn")}, **{w: "angle" for w in ("angolo", "angle")},
    **{w: "tuberosity" for w in ("tuberosita", "tuberosity")}, **{w: "margin" for w in ("margine", "margin")},
}  # fmt: skip
# Organs outside the segmentation classes that a translation must not swap for another one.
_EXTRA_ORGAN_GROUPS = (
    frozenset(("polmone", "lung")), frozenset(("utero", "uterus")), frozenset(("ovaio", "ovary")),
    frozenset(("mammella", "breast")), frozenset(("cervello", "encefalo", "brain")),
    frozenset(("uretere", "ureter")), frozenset(("testicolo", "testis")),
)  # fmt: skip


def _part_ids(tokens: Iterable[str]) -> frozenset[str]:
    return frozenset(_PART_IDS[t] for t in tokens if t in _PART_IDS)


def _choice_prompt(mention: str, sentence: str, labels: Sequence[str]) -> str:
    options = "\n".join(f"{i}. {label}" for i, label in enumerate(labels, start=1))
    return (
        "You are a radiologist. Link the expression to the anatomical structure it names in this "
        "report sentence. Answer with the option number only. Answer 0 if no option is exactly "
        "that structure, if the expression names more than one structure, or if you are not sure.\n\n"
        f"Sentence: {_mark(sentence, mention)}\nExpression: {mention}\n\nOptions:\n{options}\n0. none / not sure\n\nNumber:"
    )


def _parse(answer: str, n: int) -> int | None:
    match = re.search(r"-?\d+", answer or "")
    if not match:
        return None
    value = int(match.group())
    return value if 0 <= value <= n else None


_COMPOUND_AFTER = re.compile(r"-[A-Za-zÀ-ÿ]")


def _first_part_of_a_compound(
    mention: str, sentence: str, start: int | None = None
) -> bool:
    """A single word joined to the next by a hyphen ("cranio-facial", "cranio-caudally",
    "ileo-psoas") is a combining form, not the structure. Level codes (C5-6, L4-L5) are not words."""
    text = mention.strip()
    if " " in text or any(c.isdigit() for c in text) or not text.isalpha():
        return False
    match = locate(text, sentence, start)
    if match and match.start() > 0 and sentence[match.start() - 1] in "-_":
        return False
    return bool(match and _COMPOUND_AFTER.match(sentence[match.end() :]))


def mention_offset(
    mention: str, sentence: str, report: ReportState | None, at: int | None
) -> int | None:
    """Where the mention starts in ``sentence``, from its offset ``at`` in the report text, so that
    the words next to *this* occurrence are read when the sentence names the structure twice. None
    when the caller gave no position, or the sentence is not a stretch of the report text."""
    if report is None or at is None or not sentence:
        return None
    begin = report.text.rfind(sentence, 0, at + len(sentence))
    if begin < 0 or at - begin > len(sentence):
        return None
    local = at - begin
    if (
        sentence[local : local + len(mention.strip())].lower()
        != mention.strip().lower()
    ):
        return None
    return local


def _findings_of_an_imaging_report(report: ReportState | None, at: int | None) -> bool:
    """The sentence sits in the findings of a report whose header names an imaging exam: what it
    says is about the images, whatever words of laboratory or vital signs it carries."""
    if report is None or at is None or not report.modalities:
        return False
    return report.section_at(at) in (FINDINGS, CONCLUSIONS)


def _a_written_name(mention: str) -> bool:
    """A structure written out as a word ("liver", "tiroide"), not a code or a short form: "C3" in
    laboratory results is the complement protein, not the third cervical vertebra measured."""
    tokens = normalise(mention)
    if any(t.startswith("@") for t in tokens):
        return False
    words = re.findall(r"[A-Za-zÀ-ÿ]+", mention)
    return any(len(w) >= 4 and not w.isupper() for w in words)


# Conflicts that come from a stream reading the mention against the link (not from a check that was
# not run): the ones the conflict monitor may act on.
_READ_AGAINST = frozenset(
    ("blind_reader_disagrees", "outside_the_exam_area", "side_differs_from_the_exam")
)
# A short form: two to four capitals, possibly with a digit ("SVC", "IVC", "LAA", "RML"); level
# codes (L4, C5-6) are read by their own stream.
_ABBREVIATION = re.compile(r"(?=[A-Z0-9]*[A-Z][A-Z0-9]*[A-Z])[A-Z][A-Z0-9]{1,3}")
_SIDE_TOKENS = frozenset((SIDE_RIGHT, SIDE_LEFT, SIDE_BOTH))
_RAW_WORD = re.compile(r"[A-Za-zÀ-ÿ0-9]+")
_STRONG_BREAK = re.compile(r"[.;:!?()\[\]]")
_SIDE_IN_CLASS = re.compile(r"_(left|right)(?:_|$)")
_ANY_SIDE_WORD = re.compile(
    r"\b(?:destr[oaie]|sinistr[oaie]|dx|sx|sn|right|left|bilateral\w*|entramb\w*|both)\b",
    re.IGNORECASE,
)
_OTHER_SIDE_WORDS = re.compile(
    r"\b(?:controlateral\w*|contralateral\w*|bilateral\w*|entramb\w*|both|confronto|"
    r"rispetto|compared?|comparison|simmetric\w*|symmetric\w*)\b",
    re.IGNORECASE,
)


def _profile(
    trace: Sequence[Evidence], result: LinkResult, flagged: bool
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The independent mechanisms behind an accepted link, and the conflicts left.

    Each mechanism counts once whatever the number of streams in it: the two models are one, the
    lexicon and the part table are one (curated names). Nothing here changes the decision; it is the
    second-order reading (how well supported the link is), kept for calibration on the gold set.
    """
    support: list[str] = []
    if result.stage in ("lexicon", "parts"):
        support.append("name")
    by = {(e.stream, e.verdict) for e in trace}
    if any(e.stream == "senses" and e.verdict == SUPPORT and e.options for e in trace):
        support.append("sense")
    if ("frame", SUPPORT) in by:
        support.append("frame")
    if ("area", SUPPORT) in by:
        support.append("area")
    if ("blind", SUPPORT) in by:
        support.append("blind")
    if ("exam_side", SUPPORT) in by:
        support.append("exam_side")
    if ("discourse", SUPPORT) in by:
        support.append("discourse")
    if ("morphology", SUPPORT) in by:
        support.append("morphology")
    if result.stage in _MODEL_STAGES or ("verify", SUPPORT) in by:
        support.append("models")
    conflicts: list[str] = []
    if ("area", AGAINST) in by:
        conflicts.append("outside_the_exam_area")
    if ("blind", AGAINST) in by:
        conflicts.append("blind_reader_disagrees")
    if ("exam_side", AGAINST) in by:
        conflicts.append("side_differs_from_the_exam")
    if ("development", AGAINST) in by:
        conflicts.append("name_shared_with_a_developing_structure_in_a_developmental_text")
    if flagged and ("verify", SUPPORT) not in by:
        conflicts.append("ambiguous_form_not_checked")
    return tuple(support), tuple(conflicts)


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
    parts: Any = None  # anatomy_parts.PartTable
    translate: bool = True
    # Which meaning of an ambiguous written form is meant (GB, LM, ponte...). Loaded from
    # data/linking/word_senses.json; pass SenseInventory.empty() only to measure without it.
    senses: SenseInventory = field(default_factory=SenseInventory.load)
    # The frame a sentence is written in (exam_frame): laboratory results and vital signs name a
    # structure only as the modifier of a measurement. ExamFrames.empty() measures without it.
    frames: ExamFrames = field(default_factory=ExamFrames.load)
    # The area of the exam (exam_area): the region the exam studies, read from the exam's name in
    # the sentence or in the report header. An expectation, never a veto on its own.
    areas: ExamAreas = field(default_factory=ExamAreas.load)
    # Stage 7, the blind reader (blind_reader.BlindReader): derives a concept from the mention alone,
    # by character n-gram similarity over every name of the ontology, without an LLM and without
    # seeing the answer, and is compared with the link afterwards. Its disagreement is a conflict;
    # with ``blind_veto`` it is an abstention (off until the gold set says what it costs).
    blind: Any = None
    blind_veto: bool = False
    morphology: Any = field(default_factory=Morphology.load)
    # Does the text talk about a developing organism, and does the name also name a developing
    # structure of the ontology? Recorded as evidence (a conflict in the profile), never a veto.
    development: DevelopmentalFrame = field(default_factory=DevelopmentalFrame.load)
    # Known names of other kinds of things (a scale, a protein, a gene, a substance) that contain
    # the mention: data/linking/longer_names.json, kind taken from the ontology that lists the name.
    longer_names: LongerNames = field(default_factory=LongerNames.load)
    # The phrase around the mention read as blocks (chunk_lattice.py). Off by default: when given, the
    # reading is written in the trace (stream "blocks") and in ``LinkResult.block``; it decides nothing
    # until its roles have been checked on the radiologists' gold set (roles right in 60-80% of a
    # reading of real reports, docs/linker/lettura_del_sintagma_cervello_e_modelli_2026-10-09.md §8).
    chunk_lattice: ChunkLattice | None = None
    # The construction-integration reader (ci_reader.CIReader, a Kintsch-style integrator over the
    # readings of the mention). Off by default and, when given, trace only: its reading goes to the
    # stream "reader" and to ``LinkResult.reading``, silent, and decides nothing. It did not meet the
    # pre-registered criteria on the external check (docs/linker/perche_il_medico_legge_e_noi_sbagliamo_
    # 2026-10-09.md §6.1); it is here so the radiologists' gold set can measure it on real reports.
    ci_reader: Any = None
    _ci_documents: dict = field(default_factory=dict, repr=False, compare=False)
    # Stage 8: what a conflict between independent readings does to an accepted link. "record"
    # keeps the link and writes the conflict in the profile; "review" sends the link to the review
    # queue when an independent stream read against it (blind reader, exam area, exam side) and no
    # more than ``review_below`` mechanisms support it.
    # What kind of text the linker reads: "radiology_report" (the default, the deployment), or
    # another kind ("case_report", "literature", "clinical_note"). The kind decides what an
    # abbreviation may mean: in a radiology report SVC is the superior vena cava; elsewhere it is
    # accepted by name only with signs of imaging or anatomy in the text.
    document_type: str = "radiology_report"
    # Stage 1: a paired structure named without a side takes the side of the exam name ("RM
    # ginocchio destro: ... femore"). Off until the labelling protocol says the same; off, the
    # side of the exam is written as a proposal on the abstention.
    inherit_exam_side: bool = False
    conflict_policy: str = "record"
    review_below: int = 2
    # UBERON is-a/part-of anchored to the classes (anatomy_graph.AnatomyGraph): the neighbour check
    # on the models' choice and the fallback to the parent. None keeps the linker as it was.
    graph: Any = None
    # Keys of the forms that can mean more than one thing (``form_ambiguity.audit``). With
    # ``verify`` and at least one model, a link made from the name alone to one of these is read
    # in its sentence by every model and kept only if all of them say it is that structure.
    ambiguous: frozenset[tuple[str, ...]] = frozenset()
    verify: bool = False
    # Measurement option: read every link made from the name alone, not only the flagged forms.
    # The 2026-10-07 reading of 600 case-report mentions found misses the structural flags do not
    # mark ("heart rate", "short axis view"); this shows how many such links the models stop.
    verify_all: bool = False
    # How the check in context asks. "yes_no": "is the marked text the <structure>?" (the question
    # names the answer, and models lean towards yes). "choice": the same sentence with five balanced
    # options and an explicit "cannot be determined" (chain-of-verification style questions, and an
    # explicit abstain option, which raised safe abstention more than anything else in MedAbstain,
    # EACL 2026). Measured against each other with the verify-probe before either becomes the default.
    verify_style: str = "yes_no"
    # Whether the graph's parent becomes the link (True) or only a proposal (False, the default).
    # Three blind reviews of 100 random lifts each found 3%, 1% and 5% of them wrong before the
    # rules they suggested; the rate after them is not measured on fresh data. Until the gold set
    # certifies the fallback (step 9), it proposes and does not decide.
    accept_parent_fallback: bool = False

    def __post_init__(self) -> None:
        self._by_id = {c.cid: c for c in self.pool}
        # Ontology names of two words or more ("secondary heart field", "blood brain barrier"),
        # to see whether a mention is only a word inside a longer anatomical name.
        self._longer: dict[tuple[str, ...], set[str]] = defaultdict(set)
        self._longer_names: dict[str, str] = {}
        for candidate in self.pool:
            if candidate.is_target_class:
                continue
            self._longer_names[candidate.cid] = candidate.label
            for name in {*candidate.names, *candidate.strict_names}:
                if len(name) >= 2:
                    self._longer[tuple(name)].add(candidate.cid)
        # Every UBERON name, also of terms outside the human pool (embryonic and developmental
        # structures: "secondary heart field", "blood brain barrier", "pharyngeal arch artery").
        for node, name in (getattr(self.graph, "names", None) or {}).items():
            if len(name.split()) < 2:
                continue
            self._longer_names.setdefault(node, name)
            for key in (normalise(name), normalise(name, keep_noise=True)):
                if len(key) >= 2:
                    self._longer[tuple(key)].add(node)
        # Their other written names too (EXACT and RELATED synonyms): "aortic arch artery" is a
        # name of the pharyngeal arch artery, not of the aorta. Read as an ordered run of words
        # with a head of its own: "embryonic brain" (a brain) and "adult heart" (a heart) are
        # the structure with a qualifier, "aortic arch artery" is another structure.
        self._longer_syn: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for node, names in (getattr(self.graph, "synonyms", None) or {}).items():
            for name in names:
                key = _ordered(name)
                if len(key) >= 2:
                    self._longer_syn[key].add(node)
        # Names shared by a developing structure and an adult one ("aortic arch": the pharyngeal
        # arch artery of the embryo, the arch of the aorta).
        self._developing_names: dict[tuple[str, ...], str] = {}
        if self.graph is not None and getattr(self.graph, "names", None):
            developing = self.graph.developing()
            young: dict[tuple[str, ...], str] = {}
            adult: set[tuple[str, ...]] = set()
            for node, name in self.graph.names.items():
                for text in (name, *(self.graph.synonyms or {}).get(node, ())):
                    key = tuple(normalise(text))
                    if not key:
                        continue
                    if node in developing:
                        young.setdefault(key, name)
                    else:
                        adult.add(key)
            self._developing_names = {k: v for k, v in young.items() if k in adult}
        self._sided = sided_bases(self.pool)
        # Words that are by themselves the name of a class ("rene", "femore",
        # "kidney"). A mention containing one names that structure or a part of it.
        self._by_content: dict[frozenset[str], list[Candidate]] = defaultdict(list)
        for candidate in self.pool:
            for name in candidate.strict_names or candidate.names:
                key = _content(name)
                if key and candidate not in self._by_content[key]:
                    self._by_content[key].append(candidate)
        groups: list[frozenset[str]] = list(_EXTRA_ORGAN_GROUPS)
        for entry in self.lexicon.classes.values():
            words = set()
            for name in entry["it"] + entry["en"]:
                bare = _bare(normalise(name))
                if len(bare) == 1:
                    words.add(bare[0])
            if words:
                groups.append(frozenset(words))
        self._organ_groups = groups
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

    def _link_level(
        self,
        code: str,
        mention: str,
        sentence: str,
        report: ReportState | None = None,
        at: int | None = None,
    ) -> LinkResult:
        """A bare level code: a vertebra only with vertebra evidence, a liver segment only with the liver named."""
        letter, number = code[1], int(code[2:])
        vertebra = f"vertebrae_{'T' if letter == 'D' else letter}{number}"
        readings = []
        from_report = False
        if vertebra in self.lexicon.classes:
            if level_evidence(mention, sentence):
                readings.append(vertebra)
            elif self.document_type == "radiology_report" and level_from_report(
                code, sentence, report, at
            ):
                # the spine as the subject of the text is a property of a report; an article
                # that speaks of lumbar ganglia is not a spine report ("Figure S1", "plasmid
                # L3" in CRAFT, 8 October 2026)
                readings.append(vertebra)
                from_report = True
        segment = f"liver_segment_{number}"
        if (
            letter == "S"
            and segment in self.lexicon.classes
            and region_supported("liver", mention, sentence)
        ):
            readings.append(segment)
        if len(readings) == 1:
            return LinkResult(
                ACCEPTED,
                readings[0],
                "lexicon",
                "level_code_from_the_report_state"
                if from_report and readings[0] == vertebra
                else "level_code_with_evidence",
            )
        reason = (
            "level_code_reads_two_ways" if readings else "level_code_without_evidence"
        )
        return LinkResult(ABSTAINED, None, "lexicon", reason, options=readings)

    def link(
        self,
        mention: str,
        sentence: str,
        context: str = "",
        report: ReportState | None = None,
        at: int | None = None,
    ) -> LinkResult:
        """Link ``mention``, or abstain with the reason.

        The cheap streams (senses, integration checks, level codes, lexicon, part table) each read the
        mention on their own and are all evaluated; none sees another's answer. ``_decide`` then weighs
        them in a fixed order of authority. The expensive streams (retrieval + two models, translation)
        run only when the cheap ones leave the mention undecided. ``context`` is optional text around
        the sentence (the rest of the report, a heading); it counts half in choosing between the
        meanings of an ambiguous form. ``report`` (a ``ReportState``) and ``at`` (where the mention
        starts in the report text) give the expectation read from the top; without them nothing
        changes, and the header becomes the context when none is given.
        """
        if report is not None and not context:
            context = report.header()
        trace = self.read(mention, sentence, context, report, at)
        where = mention_offset(mention, sentence, report, at)
        result = self._decide(trace, mention, sentence)
        senses = next(e for e in trace if e.stream == "senses")
        if (
            result.status == ACCEPTED
            and senses.options
            and self.equivalent.get(result.cid, result.cid) not in senses.options
            and result.cid not in senses.options
        ):
            result = LinkResult(
                ABSTAINED, None, "senses", f"sense_{senses.cid}_names_another_class"
            )
        longer = self._inside_a_longer_name(result, mention, sentence, where)
        if longer is not None:
            result = longer
        inherited = self._side_from_the_exam(result, mention, sentence, context, report)
        if inherited is not None and self.inherit_exam_side:
            result = inherited
        elif inherited is not None:
            # Until the radiologists agree that the side of the exam name is the side of the text
            # (the protocol says today: the side is read from the sentence), it is a proposal.
            result = LinkResult(
                ABSTAINED,
                result.cid,
                result.stage,
                result.reason,
                options=[inherited.cid],
            )
            result.cid = None
            result.proposals = (inherited.cid,)
        trace = trace + [
            Evidence(
                result.stage,
                SUPPORT if result.status == ACCEPTED else UNDECIDED,
                result.cid,
                result.reason,
                result.relation,
                result.part,
                tuple(result.options),
            )
        ]
        expectation = self._expectation(result, sentence, report)
        trace.append(expectation)
        side_check = self._exam_side(result, sentence, report)
        trace.append(side_check)
        trace.append(self._discourse(result, report, at))
        if side_check.verdict == AGAINST and side_check.reason.startswith("veto:"):
            # The mention says one side, the exam is of the other: a laterality error is the
            # critical error of radiology; two readings of the side disagree, so a person decides.
            result = LinkResult(
                ABSTAINED, None, "exam_side", side_check.reason.split(":", 1)[1]
            )
        by_name = result.status == ACCEPTED and result.stage in ("lexicon", "parts")
        flagged = normalise(mention) in self.ambiguous and not senses.options
        surprise = by_name and flagged and expectation.verdict == AGAINST
        # An abbreviation means what its domain makes it mean: SVC is the superior vena cava in a
        # radiology report and stromal vascular cells in a cell-biology abstract (MedMentions,
        # 8 October 2026). Outside a radiology report (``document_type``), an abbreviation is
        # accepted by name only with signs of imaging or anatomy in the text; otherwise the models
        # read it, or the linker abstains.
        if (
            by_name
            and not senses.options
            and _ABBREVIATION.fullmatch(mention.strip())
            and not self._imaging_context(trace, sentence, report, at)
        ):
            surprise = True
            flagged = True
        if surprise and not self.chats and expectation.verdict != AGAINST:
            result = LinkResult(
                ABSTAINED, None, "domain", "abbreviation_outside_an_imaging_context"
            )
        elif surprise and not self.chats:
            # A form that can mean several things, far from the area of the exam: the prediction
            # error is what a reader would stop at, and there is no one to ask.
            result = LinkResult(
                ABSTAINED,
                None,
                "area",
                f"ambiguous_form_outside_the_exam_area:{expectation.reason.split(':', 1)[-1]}",
            )
        elif (
            self.chats
            and by_name
            and (
                surprise
                or (
                    self.verify
                    and (self.verify_all or normalise(mention) in self.ambiguous)
                    and not senses.options  # a form with a sense profile was read in context already
                )
            )
        ):
            # Deliberate reading (System 2) when the fast reading meets a conflict, or by request.
            result, check = self._verify_in_context(result, mention, sentence)
            trace.append(check)
        if self.graph is not None and result.status == ACCEPTED:
            result, graph_evidence = self._converge_on_graph(result, mention)
            trace.append(graph_evidence)
        if by_name and result.status == ACCEPTED:
            young = self._developing_names.get(tuple(normalise(mention)))
            if young:
                document = report.text if report is not None else context
                developmental, said = self.development.developmental(sentence, document)
                trace.append(
                    Evidence(
                        "development",
                        AGAINST if developmental else SILENT,
                        None,
                        f"name_shared_with_the_developing:{young}:{said}",
                    )
                )
        if self.blind is not None and result.status == ACCEPTED and result.cid:
            reading = self.blind.read(mention, result.cid)
            verdict = {"support": SUPPORT, "against": AGAINST}.get(
                reading.verdict, SILENT
            )
            trace.append(Evidence("blind", verdict, reading.best, reading.reason))
            if verdict == AGAINST and self.blind_veto:
                result = LinkResult(
                    ABSTAINED, None, "blind", f"blind_reader_disagrees:{reading.label}"
                )
        roots = self.morphology.proposes(mention) if self.morphology else ()
        if result.status == ACCEPTED and result.cid:
            same = [r for r in roots if self.morphology.names(r, result.cid)]
            trace.append(
                Evidence(
                    "morphology",
                    SUPPORT if same else SILENT,
                    result.cid,
                    f"roots_point_to:{','.join(roots)}" if roots else "",
                )
            )
        elif roots:
            result.proposals = tuple(dict.fromkeys((*result.proposals, *roots)))
            trace.append(
                Evidence(
                    "morphology",
                    SILENT,
                    None,
                    f"proposes:{','.join(roots)}",
                    options=tuple(roots),
                )
            )
        if result.status == ACCEPTED:
            result.support, result.conflicts = _profile(trace, result, flagged)
            read_against = [c for c in result.conflicts if c in _READ_AGAINST]
            if (
                self.conflict_policy == "review"
                and read_against
                and len(result.support) <= self.review_below
            ):
                held = result
                result = LinkResult(
                    ABSTAINED,
                    None,
                    "conflict",
                    f"streams_disagree:{','.join(read_against)}",
                    options=[held.cid],
                )
                result.support, result.conflicts = held.support, held.conflicts
            if self.senses.procedure_head(mention, sentence, where):
                result.role = "procedure_site"
        result.trace = trace
        result.block = next((e.reason for e in trace if e.stream == "blocks"), "")
        result.reading = next((e.reason for e in trace if e.stream == "reader"), "")
        return result

    def _inside_a_longer_name(
        self,
        result: LinkResult,
        mention: str,
        sentence: str,
        where: int | None = None,
    ) -> LinkResult | None:
        """The mention is a word inside a longer anatomical name that is not a part or a kind of the
        linked class: "heart" in "secondary heart field" (an embryonic field), "brain" in
        "blood-brain barrier", "hip" in "hip joint". Annotators of anatomy corpora label the longest
        name (CRAFT, MedMentions); a reader does the same. A longer name that is a part of the class
        ("liver parenchyma", "apex of the lung") keeps the link."""
        if (
            result.status != ACCEPTED
            or not result.cid
            or result.stage not in ("lexicon", "parts")
            or not self._longer
        ):
            return None
        # Words as written (normalised names are bags, not sequences): windows of the sentence
        # that contain the mention, one to three words wider on either side.
        spans = [m.span() for m in _RAW_WORD.finditer(sentence)]
        found = locate(mention, sentence, where)
        if not found or not spans:
            return None
        inside = [
            k
            for k, (a, b) in enumerate(spans)
            if a >= found.start() and b <= found.end()
        ]
        if not inside:
            return None
        first, last = inside[0], inside[-1]
        own = tuple(
            t for t in normalise(mention, keep_noise=True) if t not in _SIDE_TOKENS
        )
        for before in range(0, 4):
            for after in range(0, 4):
                if not before and not after:
                    continue
                lo, hi = first - before, last + after
                if lo < 0 or hi >= len(spans):
                    continue
                window = sentence[spans[lo][0] : spans[hi][1]]
                if _STRONG_BREAK.search(window):
                    continue
                keys = {
                    tuple(t for t in key if t not in _SIDE_TOKENS)
                    for key in (normalise(window), normalise(window, keep_noise=True))
                }
                # the wider window must add a word of content to what the mention says
                keys = {k for k in keys if len(k) > len(own) and not set(k) <= set(own)}
                for key in keys:
                    for term in self._longer.get(tuple(key), ()):
                        if not self._part_or_kind_of(term, result.cid):
                            name = self._longer_names.get(term, term)
                            return LinkResult(
                                ABSTAINED,
                                None,
                                "integration",
                                f"mention_is_inside_a_longer_name:{name}",
                                options=[term],
                            )
                ordered = _ordered(window)
                if ordered and ordered[-1] not in own and ordered[-1] not in _PLACE_NOUNS:
                    for term in self._longer_syn.get(ordered, ()):
                        if not self._part_or_kind_of(term, result.cid):
                            name = (self.graph.names or {}).get(term, term)
                            return LinkResult(
                                ABSTAINED,
                                None,
                                "integration",
                                f"mention_is_inside_a_longer_name:{name}",
                                options=[term],
                            )
        return None

    def _part_or_kind_of(self, term: str, cid: str) -> bool:
        """``term`` (an UBERON id) is the class, or a part or a kind of it, in the graph."""
        if self.equivalent.get(term) == cid or term == cid:
            return True
        if self.graph is None:
            return False
        nodes = self.graph.nodes_of(cid)
        if term in nodes:
            return True
        frontier, seen = {term}, {term}
        for _ in range(8):
            frontier = {
                parent
                for node in frontier
                for parent, _ in self.graph.parents.get(node, ())
                if parent not in seen
            }
            if frontier & nodes:
                return True
            seen |= frontier
            if not frontier:
                break
        return False

    def _imaging_context(
        self,
        trace: Sequence[Evidence],
        sentence: str,
        report: ReportState | None,
        at: int | None = None,
    ) -> bool:
        if self.document_type == "radiology_report":
            # the caller says the text is a radiology report: the abbreviations are radiology's
            return True
        if report is None:
            return False
        if report.modalities or report.labelled:
            # a report that names a modality or has the sections of a report (findings,
            # impression, referto, conclusioni)
            return True
        if self.frames.of(report.text).frame == IMAGING:
            # a text that speaks of images somewhere
            return True
        if at is not None and len(self._named_elsewhere(report, at)) >= 2:
            # a text about anatomy: two other structures named by curated names elsewhere in it
            # (a findings-only chest X-ray report: "heart", "lungs", "mediastinum")
            return True
        if any(e.stream == "frame" and e.verdict == SUPPORT for e in trace):
            return True
        return bool(self.areas.of(sentence))

    def _discourse(
        self, result: LinkResult, report: ReportState | None, at: int | None
    ) -> Evidence:
        """The structure is named, by an unambiguous curated name, in another sentence of the same
        report (stage 1, what the reader already knows from the report). It supports; its absence
        says nothing (most structures are named once)."""
        if result.status != ACCEPTED or not result.cid or report is None or at is None:
            return Evidence("discourse", SILENT)
        named = self._named_elsewhere(report, at)
        cid = self.equivalent.get(result.cid, result.cid)
        if cid in named or result.cid in named:
            return Evidence(
                "discourse",
                SUPPORT,
                result.cid,
                "named_in_another_sentence_of_the_report",
            )
        return Evidence("discourse", SILENT, result.cid)

    def _named_elsewhere(self, report: ReportState, at: int) -> frozenset[str]:
        key = (id(report), report.text[:64], at)
        cache = self.__dict__.setdefault("_discourse_cache", {})
        if key in cache:
            return cache[key]
        if len(cache) > 4096:
            cache.clear()
        found: set[str] = set()
        for span in report.sentences:
            if span.start <= at < span.end:
                continue
            tokens = normalise(report.text[span.start : span.end])
            for size in range(1, 7):
                for i in range(len(tokens) - size + 1):
                    entries = self.lexicon.index.get(tuple(tokens[i : i + size]))
                    if entries and len({c for c, _ in entries}) == 1:
                        found.add(entries[0][0])
        cache[key] = frozenset(found)
        return cache[key]

    def _exam_sides(self, sentence: str, report: ReportState | None) -> frozenset[str]:
        return self.areas.sides_of(sentence) or (
            report.exam_sides if report is not None else frozenset()
        )

    def _exam_side(
        self, result: LinkResult, sentence: str, report: ReportState | None
    ) -> Evidence:
        """The side of the linked class against the side the exam name states (stage 1).

        Same side: support. The other side: a conflict; it stops the link (``veto:``) unless the
        sentence itself speaks of the other side or of a comparison ("controlaterale", "rispetto al
        sinistro"), where the other side is expected."""
        if result.status != ACCEPTED or not result.cid:
            return Evidence("exam_side", SILENT)
        found = _SIDE_IN_CLASS.search(result.cid)
        sides = self._exam_sides(sentence, report)
        if not found or len(sides) != 1 or "both" in sides:
            return Evidence("exam_side", SILENT, result.cid)
        side = next(iter(sides))
        if result.reason.startswith("side_from_the_exam"):
            # the side came from the exam: it cannot also confirm itself
            return Evidence("exam_side", SILENT, result.cid, "side_taken_from_the_exam")
        if found.group(1) == side:
            return Evidence(
                "exam_side", SUPPORT, result.cid, f"exam_of_the_{side}_side"
            )
        if _OTHER_SIDE_WORDS.search(self.areas.without_exam_names(sentence)):
            return Evidence(
                "exam_side", AGAINST, result.cid, f"other_side_named_in_a_{side}_exam"
            )
        return Evidence(
            "exam_side",
            AGAINST,
            result.cid,
            f"veto:side_contradicts_the_exam:{side}",
        )

    def _side_from_the_exam(
        self,
        result: LinkResult,
        mention: str,
        sentence: str,
        context: str,
        report: ReportState | None,
    ) -> LinkResult | None:
        """A paired structure named without a side, in an exam of one side ("RM ginocchio destro:
        menisco mediale ...", "femore"): the reader takes the side of the exam. Only when the
        mention and the sentence say no side and speak of no comparison, and only for a link that the
        curated names make once the side is added; the result says where the side came from."""
        if result.status == ACCEPTED or report is None:
            return None
        sides = self._exam_sides(sentence, report)
        if len(sides) != 1 or "both" in sides:
            return None
        rest = self.areas.without_exam_names(sentence)
        if _ANY_SIDE_WORD.search(rest) or _OTHER_SIDE_WORDS.search(rest):
            # the sentence names a side, or a comparison, besides the exam name
            return None
        side = next(iter(sides))
        word = {"right": "right", "left": "left"}[side]
        inner = self.link(f"{mention} {word}", sentence, context)
        if (
            inner.status != ACCEPTED
            or inner.stage not in ("lexicon", "parts")
            or not inner.cid
        ):
            return None
        found = _SIDE_IN_CLASS.search(inner.cid)
        if not found or found.group(1) != side:
            return None
        inner.reason = f"side_from_the_exam:{side}"
        inner.trace = []
        return inner

    def _expectation(
        self, result: LinkResult, sentence: str, report: ReportState | None
    ) -> Evidence:
        """Does the linked structure belong to the area of the exam? The exam named in the sentence
        ("RM pelvi: ...") comes first, then the one in the report header."""
        if result.status != ACCEPTED or not result.cid:
            return Evidence("area", SILENT)
        areas = self.areas.of(sentence) or (
            report.areas if report is not None else frozenset()
        )
        cid = self.equivalent.get(result.cid, result.cid)
        region = self.lexicon.classes.get(cid, {}).get("region")
        relation = self.areas.relation(region, areas)
        named = ",".join(sorted(areas))
        if relation == EXPECTED:
            return Evidence(
                "area", SUPPORT, result.cid, f"expected_in_the_exam_area:{named}"
            )
        if relation == OUTSIDE:
            return Evidence(
                "area", AGAINST, result.cid, f"outside_the_exam_area:{named}"
            )
        if relation == ADJACENT:
            return Evidence(
                "area", SILENT, result.cid, f"next_to_the_exam_area:{named}"
            )
        return Evidence("area", SILENT, result.cid, "exam_area_unknown")

    def _converge_on_graph(
        self, result: LinkResult, mention: str
    ) -> tuple[LinkResult, Evidence]:
        """Steps 5 and 9 on the anatomical graph, for a link the other streams accepted.

        * A choice made by the models (deliberation) whose graph neighbour -- the other side, a
          sister, a parent or child, a structure declared disjoint -- is also among the options that
          passed every deterministic check: only the models' agreement separates the two, and two
          LLMs agreeing is not independent evidence. Abstain.
        * A link to an UBERON term finer than any class: climb to the nearest class and say how
          (``part_of``); refuse when the climb is ambiguous, needs a side the mention does not state,
          or passes through a space, a vessel, a ligament, a mesentery or an embryonic structure.
          A refusal keeps the UBERON link as it was, flagged in the trace.
        """
        if result.stage == "deliberation":
            chosen = {v for v in result.votes.values() if v}
            for raw in chosen:
                for other in result.options:
                    how = self.graph.related(raw, other) if other != raw else None
                    if how:
                        return (
                            LinkResult(
                                ABSTAINED,
                                None,
                                "graph",
                                f"neighbour_passes_the_same_checks:{how}",
                                result.options,
                                result.votes,
                            ),
                            Evidence(
                                "graph",
                                VETO,
                                raw,
                                how,
                                options=(other,),
                            ),
                        )
        if result.cid in self.lexicon.classes:
            return result, Evidence("graph", SILENT, result.cid, "already_a_class")
        from .anatomy_graph import Lift

        sides = Attributes.of(normalise(mention)).sides
        lifted = self.graph.lift(result.cid, sides)
        if isinstance(lifted, Lift):
            name = self.graph.names.get(result.cid, result.cid)
            relation = "equal" if lifted.relation == "equal" else "part_of"
            evidence = Evidence(
                "graph",
                SUPPORT,
                lifted.cid,
                f"lift_depth_{lifted.depth}",
                relation,
                name,
                lifted.path,
            )
            if not self.accept_parent_fallback:
                result.fallback = lifted.cid
                return result, evidence
            return (
                LinkResult(
                    ACCEPTED,
                    lifted.cid,
                    result.stage,
                    f"{result.reason}+graph_{relation}",
                    result.options,
                    result.votes,
                    relation=relation,
                    part="" if relation == "equal" else name,
                ),
                evidence,
            )
        return result, Evidence(
            "graph", UNDECIDED, result.cid, lifted or "no_class_near"
        )

    def read(
        self,
        mention: str,
        sentence: str,
        context: str = "",
        report: ReportState | None = None,
        at: int | None = None,
    ) -> list[Evidence]:
        """Every cheap stream's reading of the mention. Pure: no stream depends on another."""
        evidence: list[Evidence] = []
        where = mention_offset(mention, sentence, report, at)
        verdict = self.senses.judge(mention, sentence, context)
        evidence.append(
            Evidence(
                "senses",
                (SUPPORT if verdict.form else SILENT) if verdict.accepted else VETO,
                verdict.sense or None,
                verdict.reason,
                options=tuple(sorted(verdict.classes)),
            )
        )
        tokens = normalise(mention)
        attributes = Attributes.of(tokens)
        checks = (
            (
                attributes.coordination or SIDE_BOTH in attributes.sides,
                "mention_names_more_than_one_structure",
            ),
            (
                bool(attributes.numbers and set(tokens) & _NON_STRUCTURE_NUMBERED),
                "number_refers_to_a_root_disc_or_foramen",
            ),
            (
                any(t.startswith("@T") for t in tokens) and staging_context(sentence),
                "t_stage_not_a_vertebra",
            ),
            (bool(set(tokens) & CONTAINER_WORDS), "mention_names_a_space_not_an_organ"),
        )
        for failed, reason in checks:
            if failed:
                evidence.append(Evidence("integration", VETO, None, reason))
        if len(tokens) == 1 and tokens[0].startswith("@"):
            level = self._link_level(tokens[0], mention, sentence, report, at)
            evidence.append(
                Evidence(
                    "levels",
                    SUPPORT if level.status == ACCEPTED else UNDECIDED,
                    level.cid,
                    level.reason,
                    options=tuple(level.options),
                )
            )
        content = [t for t in tokens if t not in (SIDE_RIGHT, SIDE_LEFT, SIDE_BOTH)]
        raw = [
            t
            for t in normalise(mention, keep_noise=True, map_words=False)
            if t not in (SIDE_RIGHT, SIDE_LEFT, SIDE_BOTH)
        ]
        frame = self.frames.of(sentence)
        if frame.measurement and not _findings_of_an_imaging_report(report, at):
            evidence.append(
                Evidence("frame", VETO, None, f"frame_is_a_measurement:{frame.frame}")
            )
        elif frame.frame == IMAGING or _findings_of_an_imaging_report(report, at):
            evidence.append(Evidence("frame", SUPPORT, None, "frame_is_imaging"))
        if _first_part_of_a_compound(mention, sentence, where):
            evidence.append(
                Evidence(
                    "integration", VETO, None, "mention_is_the_first_part_of_a_compound"
                )
            )
        head = self.senses.attribute_head(mention, sentence, where)
        if head:
            evidence.append(
                Evidence(
                    "integration",
                    VETO,
                    None,
                    f"attribute_head_names_a_measurement:{head}",
                )
            )
        process = self.senses.process_head(mention, sentence, where)
        if process:
            # The structure is the bearer of the process (SNOMED CT "inheres in", GO "has
            # participant"), not the process: the link is not made, the structure is recorded.
            evidence.append(
                Evidence(
                    "integration",
                    VETO,
                    None,
                    f"process_head_names_a_process:{process}",
                )
            )
        name = self.senses.proper_name(mention, sentence, where)
        if name:
            evidence.append(
                Evidence(
                    "integration",
                    VETO,
                    None,
                    f"mention_is_a_word_of_a_proper_name:{name}",
                )
            )
        joined = hyphen_compound(mention, sentence, where)
        another = self.senses.non_site_head(mention, sentence, where) if joined else ""
        if joined and another:
            # "gut-brain axis", "pro-brain natriuretic peptide": a piece of a compound whose head
            # is another thing. "neck-liver", "fetal-liver-derived macrophages" keep the link.
            evidence.append(
                Evidence(
                    "integration",
                    VETO,
                    None,
                    f"mention_is_joined_by_a_hyphen_to:{joined}:{another}",
                )
            )
        heads = self.senses.non_site_heads
        defined = defined_abbreviation(mention, sentence, where, heads)
        if defined:
            evidence.append(
                Evidence(
                    "integration",
                    VETO,
                    None,
                    f"mention_is_a_word_of_the_name_defined_as:{defined}",
                )
            )
        meaning = defined_short_form(mention, sentence, where, heads)
        if meaning:
            evidence.append(
                Evidence(
                    "integration",
                    VETO,
                    None,
                    f"abbreviation_defined_in_the_text_as:{meaning}",
                )
            )
        longer_name, longer_kind = self.longer_names.containing(mention, sentence, where)
        if longer_name:
            evidence.append(
                Evidence(
                    "integration",
                    VETO,
                    None,
                    f"mention_is_inside_the_name_of_another_thing:{longer_kind}:{longer_name}",
                )
            )
        if self.chunk_lattice is not None:
            reading = self.chunk_lattice.read(mention, sentence, where)
            evidence.append(
                Evidence(
                    "blocks",
                    SILENT,
                    None,
                    f"{reading.outcome}:{reading.role or '-'}:{reading.block or '-'}",
                )
            )
        if self.ci_reader is not None:
            vector, key = (None, "")
            if report is not None and report.text:
                if report.text not in self._ci_documents:
                    if len(self._ci_documents) >= 8:
                        self._ci_documents.pop(next(iter(self._ci_documents)))
                    self._ci_documents[report.text] = self.ci_reader.document(report.text)
                vector, key = self._ci_documents[report.text]
            got = self.ci_reader.read(mention, sentence, where, None, vector, key)
            evidence.append(
                Evidence(
                    "reader",
                    SILENT,
                    None,
                    f"{got.decision}:{got.top}:{got.margin:.2f}",
                )
            )
        material = self.senses.material_qualifier(mention, sentence, where)
        if material:
            evidence.append(
                Evidence(
                    "integration",
                    VETO,
                    None,
                    f"tissue_taken_as_graft_material:{material}",
                )
            )
        if raw and all(t in _ADJECTIVES for t in raw) and content:
            evidence.append(
                Evidence("adjective", VETO, None, "bare_adjective_names_no_structure")
            )
        if self.parts is not None:
            from .anatomy_parts import strip_wrapper

            mention = strip_wrapper(mention)
        recognised, how = self.lexicon.recognise(
            mention, sentence, headless_ok=bool(verdict.form and verdict.accepted)
        )
        language = detect_language(sentence) or (report.language if report else None)
        if (
            how == "recognised"
            and not verdict.form  # a form with a sense profile has language among its evidence
            and self.lexicon.abbreviation_clash(mention, language)
        ):
            evidence.append(
                Evidence("integration", VETO, None, "abbreviation_of_another_language")
            )
        elif (
            how == "recognised"
            and not verdict.form
            and self.lexicon.name_clash(mention, language)
        ):
            evidence.append(
                Evidence("integration", VETO, None, "name_of_another_language")
            )
        if how == "recognised":
            evidence.append(Evidence("lexicon", SUPPORT, recognised[0], how))
        elif how in ("ambiguous_name", "context_does_not_support_the_name"):
            evidence.append(
                Evidence("lexicon", UNDECIDED, None, how, options=tuple(recognised))
            )
        else:
            evidence.append(Evidence("lexicon", SILENT, None, how))
        if self.parts is not None:
            from .anatomy_parts import PartLink

            known = self.parts.resolve(mention, sentence)
            bare = frozenset(
                t
                for t in normalise(mention, keep_noise=True)
                if t not in (SIDE_RIGHT, SIDE_LEFT, SIDE_BOTH)
            )
            if (
                isinstance(known, PartLink)
                and bare in self.parts.italian_only
                and language is not None
                and language != "it"
            ):
                evidence.append(
                    Evidence("integration", VETO, None, "name_of_another_language")
                )
            if isinstance(known, PartLink):
                evidence.append(
                    Evidence(
                        "parts",
                        SUPPORT,
                        known.cid,
                        known.reason,
                        relation=known.relation,
                        part=known.part,
                    )
                )
            elif isinstance(known, str):
                evidence.append(Evidence("parts", UNDECIDED, None, known))
            else:
                evidence.append(Evidence("parts", SILENT))
        return evidence

    def _decide(
        self, trace: Sequence[Evidence], mention: str, sentence: str
    ) -> LinkResult:
        """Weigh the streams. A veto from the senses or the integration checks wins over any support;
        then a level code; then a bare adjective; then the lexicon; then the part table; and only when
        none of them decides, the expensive streams."""
        by_stream: dict[str, list[Evidence]] = defaultdict(list)
        for item in trace:
            by_stream[item.stream].append(item)
        for name in ("senses", "frame", "integration"):
            for item in by_stream.get(name, ()):
                if item.verdict == VETO:
                    result = LinkResult(ABSTAINED, None, name, item.reason)
                    if item.reason.startswith(_MEASUREMENT_VETOES):
                        # Not a link, but not lost either: the structure whose property is measured
                        # (SNOMED CT "has inherent location"), for whoever reads measurements.
                        named = [
                            e.cid
                            for e in (
                                *by_stream.get("lexicon", ()),
                                *by_stream.get("parts", ()),
                            )
                            if e.verdict == SUPPORT and e.cid
                        ]
                        head = item.reason.rpartition(":")[2]
                        if (
                            named
                            and _a_written_name(mention)
                            and head not in self.senses.heads_not_measured
                        ):
                            result.role, result.about = "inherent_location", named[0]
                    return result
        for item in by_stream.get("levels", ()):
            status = ACCEPTED if item.verdict == SUPPORT else ABSTAINED
            return LinkResult(
                status, item.cid, "lexicon", item.reason, list(item.options)
            )
        for item in by_stream.get("adjective", ()):
            return LinkResult(ABSTAINED, None, "lexicon", item.reason)
        for name in ("lexicon", "parts"):
            for item in by_stream.get(name, ()):
                if item.verdict == SUPPORT:
                    return LinkResult(
                        ACCEPTED,
                        item.cid,
                        name,
                        item.reason,
                        relation=item.relation,
                        part=item.part,
                    )
                if item.verdict == UNDECIDED:
                    return LinkResult(
                        ABSTAINED, None, name, item.reason, list(item.options)
                    )
        if self.parts is not None:
            from .anatomy_parts import strip_wrapper

            mention = strip_wrapper(mention)
        return self._deliberate_or_translate(mention, sentence, normalise(mention))

    def _deliberate_or_translate(
        self, mention: str, sentence: str, tokens: tuple[str, ...]
    ) -> LinkResult:
        try:
            return self._deliberate_or_translate_models(mention, sentence, tokens)
        except ModelUnavailable as error:
            return LinkResult(
                ABSTAINED,
                None,
                "models",
                "model_unavailable",
                votes={"error": str(error)[:200]},
            )

    def _deliberate_or_translate_models(
        self, mention: str, sentence: str, tokens: tuple[str, ...]
    ) -> LinkResult:
        if len(self.chats) < 2 or (self.retriever is None and not self.translate):
            return LinkResult(
                ABSTAINED, None, "lexicon", "unknown_name_and_no_deliberation_stage"
            )
        outcome = (
            self._deliberate(mention, sentence, tokens)
            if self.retriever is not None
            else None
        )
        if outcome is not None and outcome.reason not in _NO_OPTION_REASONS:
            return outcome
        if self.translate:
            translated = self._translate(mention, sentence, tokens)
            if translated.status == ACCEPTED or outcome is None:
                return translated
        return outcome or LinkResult(
            ABSTAINED, None, "integration", "no_candidate_survives_the_checks"
        )

    def _verify_in_context(
        self, result: LinkResult, mention: str, sentence: str
    ) -> tuple[LinkResult, Evidence]:
        """Every model reads the sentence and says whether the marked text is the linked structure.

        The check for the forms that can mean more than one thing and have no sense profile
        (``ambiguous``): a form that is a structure name in one sentence is an abbreviation, a
        region, a test or a symptom in another, and the sentence is what says which. All models
        must say YES; NO or UNSURE from any of them is an abstention with the answer in the reason.
        """
        entry = self.lexicon.classes.get(result.cid or "", {})
        label = (
            f"{entry['it'][0]} / {entry['en'][0]}"
            if entry.get("it") and entry.get("en")
            else str(result.cid)
        )
        what = {
            "part_of": f"a part of the {label}",
            "contour_of": f"the outline or silhouette of the {label}",
            "approx": f"approximately the {label}",
        }.get(result.relation, f"the {label} itself")
        if self.verify_style == "choice":
            return self._verify_by_choice(result, sentence, mention, what)
        prompt = (
            "You check one automatic link in a radiology or clinical text.\n"
            f"Text: {_mark(sentence, mention)}\n"
            f"The text between <tgt> tags was linked to: {what}.\n"
            "Does the marked text, in this sentence, refer to that? Answer YES only if it names "
            "that anatomical structure here. Answer NO if it means something else (another "
            "structure, a region beside it, a test or a function, an abbreviation of another "
            "word, a word that is not anatomy). Answer UNSURE if the sentence does not settle it.\n"
            "Reply with one word: YES, NO or UNSURE."
        )
        try:
            replies = self._ask(prompt)
        except ModelUnavailable as error:
            reason = "model_unavailable"
            return (
                LinkResult(
                    ABSTAINED, None, "verify", reason, votes={"error": str(error)[:200]}
                ),
                Evidence("verify", VETO, None, reason),
            )
        answers = {name: _verdict(reply) for name, reply in replies.items()}
        verdicts = set(answers.values())
        if verdicts == {"YES"}:
            return result, Evidence(
                "verify", SUPPORT, result.cid, "context_confirmed_by_every_model"
            )
        reason = "context_check_failed:" + ("no" if "NO" in verdicts else "unsure")
        return (
            LinkResult(ABSTAINED, None, "verify", reason, votes=dict(answers)),
            Evidence("verify", VETO, None, reason),
        )

    def _verify_by_choice(
        self, result: LinkResult, sentence: str, mention: str, what: str
    ) -> tuple[LinkResult, Evidence]:
        prompt = (
            "Read one sentence from a radiology or clinical text.\n"
            f"Text: {_mark(sentence, mention)}\n"
            "In this sentence, what does the text between <tgt> tags refer to?\n"
            f"1. {what}\n"
            "2. another anatomical structure, or a region or space next to it\n"
            "3. a test, a measurement, a function, a hormone or an antibody: the structure only says "
            "what is measured\n"
            "4. an abbreviation of something else, or a word that is not anatomy\n"
            "5. the sentence does not settle it\n"
            "Reply with the number only."
        )
        try:
            replies = self._ask(prompt)
        except ModelUnavailable as error:
            reason = "model_unavailable"
            return (
                LinkResult(
                    ABSTAINED, None, "verify", reason, votes={"error": str(error)[:200]}
                ),
                Evidence("verify", VETO, None, reason),
            )
        answers = {name: _choice_verdict(reply) for name, reply in replies.items()}
        verdicts = set(answers.values())
        if verdicts == {"YES"}:
            return result, Evidence(
                "verify", SUPPORT, result.cid, "context_confirmed_by_every_model"
            )
        reason = "context_check_failed:" + ("no" if "NO" in verdicts else "unsure")
        return (
            LinkResult(ABSTAINED, None, "verify", reason, votes=dict(answers)),
            Evidence("verify", VETO, None, reason),
        )

    def _ask(self, prompt: str) -> dict[str, str]:
        """The same prompt to every model at once: the calls are independent, so they run in parallel."""
        names = list(self.chats)

        def one(name: str) -> str:
            try:
                return self.chats[name](prompt)
            except Exception as error:  # a model that does not answer is not a vote
                raise ModelUnavailable(f"{name}: {error}") from error

        if len(names) < 2:
            return {name: one(name) for name in names}
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            answers = list(pool.map(one, names))
        return dict(zip(names, answers, strict=True))

    def _deliberate(
        self, mention: str, sentence: str, tokens: tuple[str, ...]
    ) -> LinkResult:
        # Only near candidates are offered. When the closest ones are all rejected,
        # what is left is far from the mention, and two models agreeing on a far
        # option is exactly the correlated error this design does not trust.
        try:
            found = self.retriever(mention, sentence, self.nearest)
        except Exception as error:  # the encoder or the describing model did not answer
            raise ModelUnavailable(f"retriever: {error}") from error
        ranked = list(dict.fromkeys(found))[: self.nearest]
        options = [
            cid
            for cid in ranked
            if (candidate := self._by_id.get(cid))
            and verify(tokens, candidate, self._sided) is None
            and covers(mention, candidate)
            and not wrong_system(candidate, sentence)
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
        for name, answer in self._ask(prompt).items():
            choice = _parse(answer, len(options))
            votes[name] = options[choice - 1] if choice else None
        chosen = set(votes.values())
        if None in chosen or len(chosen) != 1:
            reason = "a_model_was_not_sure" if None in chosen else "models_disagree"
            return LinkResult(ABSTAINED, None, "deliberation", reason, options, votes)
        cid = chosen.pop()
        candidate = self._by_id[cid]
        if (
            verify(tokens, candidate, self._sided)
            or not covers(mention, candidate)
            or wrong_system(candidate, sentence)
            or not self._context_allows(candidate, mention, sentence)
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

    def _organs_agree(self, said: Sequence[str], wrote: Sequence[str]) -> bool:
        """Every organ the mention names appears in the translation, and no other organ does."""
        a, b = set(said), set(wrote)
        for source, target in ((a, b), (b, a)):
            for group in self._organ_groups:
                if source & group and not target & group:
                    return False
        return True

    def _translate(
        self, mention: str, sentence: str, tokens: tuple[str, ...]
    ) -> LinkResult:
        """Italian (or any) mention -> standard English term, then an exact lookup.

        Translation is an easier task than choosing among look-alike options, and the
        answer is not trusted: it must equal, word for word, a name in the pool, both
        models must reach the same single concept, and the side, number and type of
        the original mention are checked against it.
        """
        prompt = _translate_prompt(mention, sentence)
        terms: dict[str, str | None] = {
            name: _clean_term(answer) for name, answer in self._ask(prompt).items()
        }
        if None in terms.values():
            return LinkResult(
                ABSTAINED, None, "translation", "a_model_did_not_translate", votes=terms
            )
        found: dict[str, frozenset[str]] = {}
        for name, term in terms.items():
            term_tokens = normalise(str(term), keep_noise=True)
            key = _content(term_tokens)
            if _part_ids(normalise(mention, keep_noise=True)) != _part_ids(term_tokens):
                key = frozenset()  # a part or tissue word was added, dropped or swapped
            if not self._organs_agree(normalise(mention), normalise(str(term))):
                key = frozenset()  # the translation names another organ
            said, wrote = Attributes.of(tokens), Attributes.of(term_tokens)
            if (said.sides, said.numbers, said.positions, said.qualifiers) != (
                wrote.sides,
                wrote.numbers,
                wrote.positions,
                wrote.qualifiers,
            ):
                key = (
                    frozenset()
                )  # the translation changed side, number, position or qualifier
            found[name] = (
                frozenset()
                if not key
                else frozenset(
                    c.cid
                    for c in self._by_content.get(key, ())
                    if verify(tokens, c, self._sided) is None
                    and self._anchored(normalise(str(term)), c)
                    and not wrong_system(c, sentence)
                    and self._context_allows(c, mention, sentence)
                )
            )
        options = sorted(set().union(*found.values()))
        if len(set(found.values())) != 1 or len(options) != 1:
            return LinkResult(
                ABSTAINED,
                None,
                "translation",
                "translations_do_not_agree_on_one_concept",
                options,
                terms,
            )
        cid = options[0]
        candidate = self._by_id[cid]
        resolved = cid if candidate.is_target_class else self.equivalent.get(cid, cid)
        return LinkResult(
            ACCEPTED,
            resolved,
            "translation",
            "models_translate_to_the_same_pool_name",
            options,
            terms,
        )
