"""Knowledge for reading a phrase the way an expert does: E2 (informed reader) and E4a (memory arm).

Why this exists (``docs/linker/perche_il_medico_legge_e_noi_sbagliamo_2026-10-09.md``): of the 33
errors of the external check, the largest group are compound names that a bigger vocabulary knows
("inferior vena cava filter placement", "Living-Related Liver Donation") and the LLM segmenter,
asked without help, found 10 of 22 MedMentions errors while flagging 29 % of the right links. The
reading theory behind the two experiments is Kintsch's construction-integration model: a *dumb,
exhaustive* construction phase (every sense and every known name of the words is activated,
context-free) followed by an integration phase in which the context keeps what fits (Kintsch 1998,
ch. 4-5; Ericsson & Kintsch 1995 for the expert's long-term working memory). So:

* the knowledge given to the reader lists **every** reading the vocabularies have for the words
  around the mention -- the anatomical one included -- not only the ones that would make the link
  wrong (giving only those would be a hint, a "seductive detail" in the sense of Sanchez & Wiley
  2006);
* the worked examples are annotated sentences from **other** documents (MedMentions training
  split, never the case's own document), chosen by similarity of the sentence, as a long-term
  memory of how the annotators read the same word (worked examples: Sweller 2024).

Three pieces, each usable alone:

``Examples``        k nearest annotated MedMentions training sentences whose annotation covers the
                    same word;
``Definitions``     NCIt names (and EXACT synonyms) with kind and definition; UMLS exact names with
                    semantic types and a definition when ``UMLS_API_KEY`` is set;
``umls_arm``        E4a, deterministic: the longest word group around the mention that UMLS names
                    exactly and whose types are not anatomy or disease ("is it memory?").
"""

from __future__ import annotations

import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ncit_kinds import WORD, read_obo  # noqa: E402

from melampo.memory.head_typing import (  # noqa: E402,F401
    ANATOMY, CHEMICAL, DEVICE, DISEASE, FOOD, ORGANISM, PROCEDURE, PROCESS, PROPERTY, PROTEIN,
)
from melampo.memory.word_senses import _fold, locate  # noqa: E402

# -- UMLS semantic types -> the reader's types (the LLM_TYPES of phrase_probe) -------------------

_ORDER = (
    ("anatomical structure", ANATOMY),
    ("disease or finding", DISEASE),
    ("procedure", PROCEDURE),
    ("device or object", DEVICE),
    ("food", FOOD),
    ("protein or gene", PROTEIN),
    ("chemical or drug", CHEMICAL),
    ("organism", ORGANISM),
    ("process or function", PROCESS),
    ("measurement or property", PROPERTY),
)
NCIT_TYPE = {
    "anatomy": "anatomical structure", "disease": "disease or finding", "procedure": "procedure",
    "device": "device or object", "food": "food", "protein": "protein or gene", "gene": "protein or gene",
    "chemical": "chemical or drug", "organism": "organism", "process": "process or function",
    "property": "measurement or property", "activity": "process or function", "conceptual": "other",
}
SITE_TYPES = ("anatomical structure", "disease or finding")
_FUNCTION = frozenset(
    "a an the of in on at to for with by from and or but as is are was were be been this that these "
    "those its their his her our we they it which who whom whose than then into onto over under there "
    "here also not no".split()
)


def type_of_tuis(tuis) -> str:
    """The reader's type for a set of UMLS semantic types (first matching group wins)."""
    have = set(tuis)
    for name, group in _ORDER:
        if have & group:
            return name
    return "other"


# -- where the mention is, and the word groups around it ----------------------------------------


def where(case):
    """The occurrence the case is about (its offset ``at`` when known, else the first one)."""
    return locate(case["mention"], case["sentence"], case.get("at"))


def word_groups(case, left: int = 3, right: int = 4, max_words: int = 6) -> list[str]:
    """Word groups containing the mention and longer than it, inside its clause, that neither start
    nor end with a function word ("the heart" is not a name)."""
    found = where(case)
    if not found:
        return []
    s = case["sentence"]
    before = re.split(r"[,;:()\[\]]", s[: found.start()])[-1].split()[-left:]
    after = re.split(r"[,;:()\[\].]", s[found.end() :])[0].split()[:right]
    mention = s[found.start() : found.end()]
    out = []
    for a in range(len(before) + 1):
        for b in range(len(after) + 1):
            if a == 0 and b == 0:
                continue
            words = before[len(before) - a :] + [mention] + after[:b]
            words = [w.strip("\"'.,;:") for w in words]
            if len(" ".join(words).split()) > max_words:
                continue
            if words[0].lower() in _FUNCTION or words[-1].lower() in _FUNCTION:
                continue
            if not (words[0][:1].isalnum() and words[-1][-1:].isalnum()):
                continue  # "-based inferior vena cava": a broken word, not a name
            out.append(" ".join(words))
    return list(dict.fromkeys(out))


# -- worked examples ------------------------------------------------------------------------------


def medmentions_splits(root: Path) -> dict[str, str]:
    """PMID -> trng / dev / test (the corpus's own split files)."""
    out = {}
    for split in ("trng", "dev", "test"):
        path = root / "full" / "data" / f"corpus_pubtator_pmids_{split}.txt"
        if path.exists():
            for line in path.read_text("utf-8").split():
                out[line.strip()] = split
    return out


class Examples:
    """Annotated MedMentions training sentences whose annotation covers the same word."""

    def __init__(self, texts: dict, labels: dict, splits: dict, k: int = 5, cap: int = 3000, lookup=None,
                 frozen=None):
        from external_check import sentence_and_offset

        self.k, self.cap, self.lookup = k, cap, lookup
        self.texts = texts
        self.sentence_and_offset = sentence_and_offset
        self.by_word: dict[str, list] = defaultdict(list)
        for doc, labs in labels.items():
            if splits.get(doc) != "trng":
                continue
            if frozen is not None:
                frozen.refuse([doc], "worked examples", "medmentions")
            text = texts.get(("medmentions", doc), "")
            for start, end, (cui, types) in labs:
                for w in {_fold(w) for w in WORD.findall(text[start:end]) if len(w) >= 3}:
                    self.by_word[w].append((doc, start, end, cui, tuple(sorted(types))))

    def __call__(self, case) -> list[dict]:
        words = [_fold(w) for w in WORD.findall(case["mention"]) if len(w) >= 3]
        if not words:
            return []
        key = _fold(case["mention"])
        pool = [e for e in self.by_word.get(words[-1], ()) if e[0] != case.get("doc")]
        if len(pool) > self.cap:
            pool = random.Random(f"{case.get('doc')}:{key}").sample(pool, self.cap)
        target = set(_content(case["sentence"]))
        scored, seen = [], set()
        for doc, start, end, cui, types in pool:
            text = self.texts[("medmentions", doc)]
            span = text[start:end]
            if key not in _fold(span):
                continue
            sentence, at = self.sentence_and_offset(text, start, end)
            if (doc, sentence) in seen or not (0 <= at < len(sentence)):
                continue
            seen.add((doc, sentence))
            words_there = set(_content(sentence))
            score = len(target & words_there) / (len(target | words_there) or 1)
            scored.append((score, doc, sentence, at, span, cui, types))
        scored.sort(key=lambda x: (-x[0], x[1], x[3]))
        out = []
        for score, _doc, sentence, at, span, cui, types in scored[: self.k]:
            if "UnknownType" in types and self.lookup is not None and self.lookup.available:
                info = self.lookup.concept(cui) or {}
                types = tuple(info.get("types", ())) or types
            marked = f"{sentence[:at]}[[{sentence[at : at + len(span)]}]]{sentence[at + len(span):]}"
            out.append({"sentence": marked[:400], "phrase": span, "type": type_of_tuis(types), "score": round(score, 3)})
        return out


def _content(text: str) -> list[str]:
    return [w for w in (_fold(x) for x in WORD.findall(text)) if len(w) >= 4 and w not in _FUNCTION]


# -- definitions ----------------------------------------------------------------------------------


class Definitions:
    """What the vocabularies say about the word groups around the mention (all readings)."""

    def __init__(self, ncit_path: Path | None, kinds=None, lookup=None, per_case: int = 8):
        self.lookup, self.per_case = lookup, per_case
        self.ncit: dict[str, tuple[str, str, str]] = {}
        if ncit_path and Path(ncit_path).exists() and kinds is not None:
            definitions = _ncit_definitions(Path(ncit_path))
            with open(ncit_path, encoding="utf-8") as handle:
                for t in read_obo(handle):
                    kind = kinds.kind_of_id.get(t["id"])
                    if not kind or t["obs"]:
                        continue
                    for name in [t["name"], *t["syn"]]:
                        key = " ".join(_fold(w) for w in WORD.findall(name))
                        if key and key not in self.ncit:
                            self.ncit[key] = (t["name"], NCIT_TYPE.get(kind, "other"), definitions.get(t["id"], ""))

    def __call__(self, case) -> list[dict]:
        found = where(case)
        mention = case["sentence"][found.start() : found.end()] if found else case["mention"]
        groups = sorted(word_groups(case), key=lambda g: -len(g.split()))
        out, seen = [], set()
        for phrase in [*groups, mention]:
            key = " ".join(_fold(w) for w in WORD.findall(phrase))
            if key in self.ncit:
                name, kind, definition = self.ncit[key]
                if (name, kind) not in seen:
                    seen.add((name, kind))
                    out.append({"phrase": phrase, "name": name, "type": kind, "source": "NCIt", "definition": definition[:300]})
            if self.lookup is not None and self.lookup.available:
                for c in (self.lookup.exact(phrase) or [])[:3]:
                    kind = type_of_tuis(c["types"])
                    if (c["name"].lower(), kind) in {(n.lower(), k) for n, k in seen}:
                        continue
                    seen.add((c["name"], kind))
                    definition = self.lookup.definition(c["cui"]) or ""
                    out.append({"phrase": phrase, "name": c["name"], "type": kind, "source": "UMLS", "definition": definition[:300]})
        # longest phrases first, but always keep the readings of the mention itself
        own = [o for o in out if _fold(o["phrase"]) == _fold(mention)]
        longer = [o for o in out if o not in own]
        return (longer[: max(0, self.per_case - len(own[:2]))] + own[:2])


def _ncit_definitions(path: Path) -> dict[str, str]:
    out, current = {}, None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("id: "):
                current = line[4:].strip()
            elif current and line.startswith("def: "):
                found = re.match(r'def: "(.*?)(?<!\\)"', line)
                if found:
                    out[current] = found.group(1).replace('\\"', '"')
    return out


# -- E4a: the memory arm --------------------------------------------------------------------------


def umls_arm(case, lookup) -> None:
    """The longest word group around the mention that UMLS names exactly. If its type is not a body
    site (anatomy, disease), the phrase names another kind of thing (``umls_other``); if it is a
    longer anatomical name, ``umls_longer_anatomy``. A failed lookup is ``umls_lost``."""
    case["umls_other"] = 0.0
    case["umls_longer_anatomy"] = 0.0
    case["umls_kind"] = None
    for phrase in sorted(word_groups(case), key=lambda g: (-len(g.split()), g)):
        found = lookup.exact(phrase)
        if found is None:
            case["umls_lost"] = 1.0
            continue
        if not found:
            continue
        kinds = Counter(type_of_tuis(c["types"]) for c in found)
        kind = kinds.most_common(1)[0][0]
        case["umls_phrase"], case["umls_name"] = phrase, found[0]["name"]
        case["umls_kind"] = NCIT_BACK.get(kind)
        if all(type_of_tuis(c["types"]) not in SITE_TYPES for c in found):
            case["umls_other"] = 1.0
        elif kind == "anatomical structure":
            case["umls_longer_anatomy"] = 1.0
        return


NCIT_BACK = {
    "anatomical structure": "anatomy", "disease or finding": "disease", "procedure": "procedure",
    "device or object": "device", "food": "food", "protein or gene": "protein", "chemical or drug": "chemical",
    "organism": "organism", "process or function": "process", "measurement or property": "property",
    "other": "conceptual",
}


def compound_candidates(cases) -> list[dict]:
    """E4: the non-anatomical UMLS names found around anatomical mentions, for curation by
    radiologists (name, type, how often, two sentences, how many were judged errors)."""
    groups: dict[tuple, dict] = {}
    for c in cases:
        if not c.get("umls_other"):
            continue
        key = (c["umls_name"], c.get("umls_kind"))
        g = groups.setdefault(key, {"name": key[0], "kind": key[1], "count": 0, "errors": 0, "examples": []})
        g["count"] += 1
        g["errors"] += int(c.get("label") == 1)
        if len(g["examples"]) < 2:
            g["examples"].append(c["sentence"][:200])
    return sorted(groups.values(), key=lambda g: (-g["count"], g["name"]))
