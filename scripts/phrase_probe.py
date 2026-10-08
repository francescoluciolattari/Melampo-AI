#!/usr/bin/env python
"""Experiment `phrase-probe`: read the whole phrase a mention stands in, and say what kind of thing it names.

On MedMentions the linker's errors have the same evidence as its right links (name, blind reader,
discourse, no conflict): every stream reads the word, none reads the phrase ("inferior vena cava
*filter placement*", "*heart of Maroilles cheese*"). This script measures, on the links the external
check judged, several independent ways of reading the phrase. Each one answers the same question
-- *does the phrase that contains the mention name another kind of thing than a body site, and which
kind?* -- with a different mechanism and different failure modes, so that their agreement means
something. It changes nothing in the linker.

Arms (the deterministic ones need only ``--ncit``; the others are switched on by their option):

* ``head``   the head of the phrase (scan to the right, measurement nominalisations dropped) typed by
             NCIt: the kind of the NCIt classes whose names end with that word (procedure, device,
             food, protein, process, ...). Deterministic.
* ``object`` the construction "<mention> of <phrase>" whose head NCIt types as food or a manufactured
             object ("heart of Maroilles cheese"): the word names the inside of an object.
* ``discourse`` one sense per discourse (Gale, Church and Yarowsky 1992): another occurrence of the
             same word in the same document was read as the inside of an object.
* ``theme``  share of the document's words that NCIt names as food or manufactured objects.
* ``typo``   a word of the phrase is not in the vocabulary (NCIt, UBERON) but is within a small edit
             distance of a vocabulary word, and the corrected word changes the head's kind. The
             correction is a hypothesis, recorded with the original.
* ``parser`` (``--spacy``) the phrase read by a dependency parser: the mention's noun chunk, the
             head of a prepositional phrase ("development of pancreas helps"), the mention as subject.
* ``gliner`` (``--gliner``) a typed span recogniser (GLiNER-BioMed): the longest predicted entity that
             covers the mention, its type and score.
* ``retrieval`` (``--encoder``) the word groups containing the mention, embedded and searched among NCIt
             preferred names (all kinds) with a biomedical name encoder (SapBERT): best non-anatomical
             match minus best anatomical match.
* ``llm``    (``--llm``, needs OPENROUTER_API_KEY) two models are asked to split the sentence into
             concepts and say which concept contains the word, its type, and whether the word names
             where something is in a body. This is a segmentation question, not "is the link right?".

The report gives, for each signal, AUC (error against right link) and operating points (errors
caught for 0, 5, 20, 50 right links lost); the *gate* analysis (would a gate on conflict or low
convergence ever have opened on the errors?); and the *role* analysis: for each error, whether the
kind an arm reads matches the kind of the gold label (procedure, device, process...), i.e. whether
recording the structure with a role instead of linking it would agree with the annotator's purpose.

    python scripts/phrase_probe.py --craft ext/craft --medmentions ext/mm --ncit ncit.obo \
        --spacy en_core_web_sm --gliner Ihor/gliner-biomed-bi-small-v1.0 \
        --encoder cambridgeltl/SapBERT-from-PubMedBERT-fulltext --llm sample
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from head_probe import auc, operating_points  # noqa: E402

from melampo.memory.word_senses import _TAIL, _fold, locate, noun_phrase_after  # noqa: E402

DATA = ROOT / "data" / "linking"
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")

# NCIt classes that decide the kind of a name; the first ancestor in this order wins.
KIND_ROOTS = (
    ("anatomy", "NCIT:C12219"),
    ("disease", "NCIT:C7057"),
    ("procedure", "NCIT:C25218"),
    ("device", "NCIT:C97325"),
    ("food", "NCIT:C1949"),
    ("protein", "NCIT:C17021"),
    ("gene", "NCIT:C16612"),
    ("chemical", "NCIT:C1908"),
    ("organism", "NCIT:C14250"),
    ("process", "NCIT:C17828"),
    ("property", "NCIT:C20189"),
    ("activity", "NCIT:C43431"),
    ("conceptual", "NCIT:C20181"),
)
RETIRED = "NCIT:C28428"
# Kinds for which a structure in the phrase is still the site the text speaks of.
SITE_KINDS = frozenset({"anatomy", "disease"})
OBJECT_KINDS = frozenset({"food", "device"})
# The role the structure would have if the phrase names a thing of this kind.
ROLE_OF_KIND = {
    "procedure": "procedure_site",
    "device": "device_site",
    "process": "inherent_location",
    "property": "inherent_location",
    "activity": "inherent_location",
    "protein": "inside_a_name",
    "gene": "inside_a_name",
    "chemical": "inside_a_name",
    "food": "not_a_body_site",
    "organism": "not_a_body_site",
    "conceptual": "not_a_body_site",
}
# The same roles from the semantic types of a MedMentions label (UMLS semantic network).
ROLE_OF_TYPE = {
    **dict.fromkeys(("T058", "T059", "T060", "T061", "T063"), "procedure_site"),
    **dict.fromkeys(("T073", "T074", "T075", "T203"), "device_site"),
    **dict.fromkeys(
        ("T032", "T038", "T039", "T040", "T041", "T042", "T043", "T044", "T045", "T067", "T068",
         "T069", "T070", "T169", "T201", "T081", "T080"),
        "inherent_location",
    ),
    **dict.fromkeys(("T028", "T087", "T114", "T116", "T121", "T123", "T126", "T129", "T109",
                     "T130", "T131", "T125", "T127", "T167", "T104", "T120"), "inside_a_name"),
    **dict.fromkeys(("T082", "T098", "T099", "T100", "T101", "T097", "T168", "T170", "T071",
                     "T072", "T204", "T007", "T005", "T004", "T001"), "not_a_body_site"),
}
GLINER_LABELS = {
    "anatomical structure": "anatomy",
    "disease or finding": "disease",
    "medical or laboratory procedure": "procedure",
    "medical device or object": "device",
    "food": "food",
    "protein or gene": "protein",
    "chemical or drug": "chemical",
    "organism or microbiome": "organism",
    "biological process or function": "process",
    "measurement or property": "property",
    "assessment scale or score": "conceptual",
}
LLM_TYPES = (
    "anatomical structure", "disease or finding", "procedure", "device or object", "food",
    "protein or gene", "chemical or drug", "organism", "process or function",
    "measurement or property", "other",
)
LLM_KIND = dict(zip(LLM_TYPES, (
    "anatomy", "disease", "procedure", "device", "food", "protein", "chemical", "organism",
    "process", "property", "conceptual",
), strict=True))


def singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


# -- NCIt kinds ----------------------------------------------------------------------------------


def read_obo(lines):
    """(id, name, synonyms, parents, obsolete) for each [Term]."""
    term = None
    for line in lines:
        line = line.rstrip("\n")
        if line == "[Term]":
            if term:
                yield term
            term = {"id": "", "name": "", "syn": [], "isa": [], "obs": False}
        elif line.startswith("["):
            if term:
                yield term
            term = None
        elif term is not None:
            if line.startswith("id: "):
                term["id"] = line[4:]
            elif line.startswith("name: "):
                term["name"] = line[6:]
            elif line.startswith("is_obsolete: true"):
                term["obs"] = True
            elif line.startswith("is_a: "):
                term["isa"].append(line[6:].split(" ")[0])
            elif line.startswith("synonym: "):
                found = re.match(r'synonym: "(.*)" (\w+)', line)
                if found:
                    term["syn"].append(found.group(1))
    if term:
        yield term


class Kinds:
    """Kind of an NCIt name, of a single word, and of the words that end NCIt names."""

    def __init__(self, terms):
        terms = [t for t in terms if t["id"] and not t["obs"]]
        parents = {t["id"]: t["isa"] for t in terms}
        memo: dict[str, frozenset[str]] = {}

        def ancestors(node: str) -> frozenset[str]:
            if node in memo:
                return memo[node]
            memo[node] = frozenset()
            seen, stack = set(), list(parents.get(node, ()))
            while stack:
                p = stack.pop()
                if p not in seen:
                    seen.add(p)
                    stack.extend(parents.get(p, ()))
            memo[node] = frozenset(seen)
            return memo[node]

        self.kind_of_id: dict[str, str] = {}
        self.names: dict[str, str] = {}
        self.preferred: list[tuple[str, str]] = []
        last = defaultdict(Counter)
        self.vocabulary: set[str] = set()
        for t in terms:
            up = ancestors(t["id"]) | {t["id"]}
            if RETIRED in up or t["name"].lower().startswith("obsolete"):
                continue
            kind = next((k for k, root in KIND_ROOTS if root in up), None)
            if kind is None:
                continue
            self.kind_of_id[t["id"]] = kind
            for i, name in enumerate([t["name"], *t["syn"]]):
                words = [_fold(w) for w in WORD.findall(name)]
                if not words or len(words) > 6:
                    continue
                self.vocabulary.update(w for w in words if len(w) >= 4)
                key = " ".join(words)
                self.names.setdefault(key, kind)
                last[singular(words[-1])][kind] += 1
                if i == 0 and len(words) <= 4:
                    self.preferred.append((name, kind))
        self.last = last

    @classmethod
    def from_obo(cls, path: Path) -> Kinds:
        with open(path, encoding="utf-8") as handle:
            return cls(list(read_obo(handle)))

    def word_kind(self, word: str) -> tuple[str | None, float]:
        """Kind of a head word: the NCIt class named by the word itself, else the dominant kind of
        the names ending with it (share of those names), else None."""
        w = _fold(word)
        for form in (w, singular(w)):
            if form in self.names:
                return self.names[form], 1.0
        counts = self.last.get(singular(w))
        if not counts:
            return None, 0.0
        kind, n = counts.most_common(1)[0]
        total = sum(counts.values())
        if total < 3:
            return None, 0.0
        return kind, n / total


# -- cases --------------------------------------------------------------------------------------


def collect(args):
    """All rows of the external check (every status) and the judged accepted links as cases."""
    import external_check as ec

    obo = Path(args.uberon)
    lexicon, parts, graph, linker = ec.build_linker(obo)
    umls = ec.umls_of_nodes(obo)
    rows, texts = [], {}
    for name, root, reader in (
        ("craft", args.craft, ec.craft_documents),
        ("medmentions", args.medmentions, ec.medmentions_documents),
    ):
        if not root:
            continue
        documents = []
        for count, (doc, text, labels) in enumerate(reader(Path(root))):
            if args.limit and count >= args.limit:
                break
            texts[(name, doc)] = text
            documents.append((doc, text, labels))
        rows += ec.run_corpus(name, documents, lexicon, parts, graph, linker, umls, None, args.workers)
    cases = [
        dict(r, label=1 if r["by_project_rule"] == "error" else 0)
        for r in rows
        if r["status"] == "accepted" and r.get("by_project_rule") in ("agrees", "error")
    ]
    return rows, texts, cases


# -- deterministic arms -------------------------------------------------------------------------

_OF = re.compile(r"^\s+of\s+(?:the\s+|a\s+|an\s+|this\s+|these\s+)?")


def phrase_words(case) -> tuple[list[str], re.Match | None]:
    found = locate(case["mention"], case["sentence"])
    return (noun_phrase_after(case["sentence"], found.end()) if found else []), found


def head_of(words: list[str]) -> str:
    words = list(words)
    while words and words[-1] in _TAIL:
        words.pop()
    return words[-1] if words else ""


def arm_head(case, kinds: Kinds):
    words, _ = phrase_words(case)
    head = head_of(words)
    kind, share = kinds.word_kind(head) if head else (None, 0.0)
    case["head_word"], case["head_kind"] = head, kind
    case["head_other"] = share if kind and kind not in SITE_KINDS else 0.0


def object_of(sentence: str, end: int, kinds: Kinds) -> tuple[str, str | None]:
    after = _OF.match(sentence[end:])
    if not after:
        return "", None
    words = noun_phrase_after(sentence, end + after.end() - 1)
    head = head_of(words)
    kind, _ = kinds.word_kind(head) if head else (None, 0.0)
    return head, kind


def arm_object(case, kinds: Kinds):
    _, found = phrase_words(case)
    head, kind = object_of(case["sentence"], found.end(), kinds) if found else ("", None)
    case["object_head"], case["object_kind"] = head, kind
    case["object"] = 1.0 if kind in OBJECT_KINDS else 0.0


def arm_discourse(cases, rows, kinds: Kinds):
    """One sense per discourse: a word read as the inside of an object anywhere in the document."""
    object_sense = set()
    for r in rows:
        found = locate(r["mention"], r["sentence"])
        if found and object_of(r["sentence"], found.end(), kinds)[1] in OBJECT_KINDS:
            object_sense.add((r["corpus"], r["doc"], _fold(r["mention"])))
    for c in cases:
        c["discourse"] = 1.0 if (c["corpus"], c["doc"], _fold(c["mention"])) in object_sense else 0.0


def arm_theme(cases, texts, kinds: Kinds):
    share = {}
    for key, text in texts.items():
        words = [_fold(w) for w in WORD.findall(text)]
        typed = [kinds.names.get(w) for w in words if len(w) >= 4]
        known = [k for k in typed if k]
        share[key] = (sum(k in OBJECT_KINDS for k in known) / len(known)) if known else 0.0
    for c in cases:
        c["theme"] = share.get((c["corpus"], c["doc"]), 0.0)


def damerau(a: str, b: str, cap: int) -> int:
    """Optimal string alignment distance, stopping early above ``cap``."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > cap:
            return cap + 1
        prev2, prev = prev, cur
    return prev[-1]


class Speller:
    """Nearest vocabulary word for an unknown word (same first letter, small edit distance)."""

    def __init__(self, vocabulary):
        self.vocabulary = set(vocabulary)
        self.buckets = defaultdict(list)
        for w in self.vocabulary:
            if w.isalpha():
                self.buckets[(w[0], len(w))].append(w)

    def correct(self, word: str) -> str | None:
        w = _fold(word)
        if len(w) < 5 or not w.isalpha() or w in self.vocabulary or singular(w) in self.vocabulary:
            return None
        cap = 1 if len(w) < 7 else 2
        best = None
        for n in range(len(w) - cap, len(w) + cap + 1):
            for v in self.buckets.get((w[0], n), ()):
                d = damerau(w, v, cap)
                if d <= cap and (best is None or d < best[0]):
                    best = (d, v)
        return best[1] if best else None


def arm_typo(case, kinds: Kinds, speller: Speller):
    words, _ = phrase_words(case)
    case["typo"] = 0.0
    for i, raw in enumerate(words):
        stem = raw.split("-")[0]
        fixed = speller.correct(stem)
        if not fixed:
            continue
        corrected = words[:i] + [fixed] + words[i + 1 :]
        kind, share = kinds.word_kind(head_of(corrected))
        case["typo_hypothesis"] = {"written": raw, "read_as": fixed, "head_kind": kind}
        if kind and kind not in SITE_KINDS and kind != case.get("head_kind"):
            case["typo"] = share
        break


# -- model arms ----------------------------------------------------------------------------------


class Parser:
    def __init__(self, model: str, kinds: Kinds):
        import spacy

        self.nlp = spacy.load(model, disable=["ner"])
        self.kinds = kinds

    def __call__(self, case):
        found = locate(case["mention"], case["sentence"])
        if not found:
            return
        doc = self.nlp(case["sentence"])
        ids = [t.i for t in doc if t.idx >= found.start() and t.idx + len(t.text) <= found.end()]
        if not ids:
            return
        last = doc[max(ids)]
        chunk = next((c for c in doc.noun_chunks if min(ids) >= c.start and max(ids) < c.end), None)
        root = chunk.root if chunk is not None else last
        kind = None
        if root.i not in ids:
            kind, _ = self.kinds.word_kind(root.text)
        # "development of (the) pancreas": the structure is the object of "of" under a noun
        governor = last.head if last.dep_ == "pobj" else None
        of_head = governor.head if governor is not None and governor.lower_ == "of" else None
        of_kind = self.kinds.word_kind(of_head.text)[0] if of_head is not None and of_head.pos_ == "NOUN" else None
        case["parser_kind"] = kind
        case["parser_other"] = 1.0 if kind and kind not in SITE_KINDS else 0.0
        case["parser_of_kind"] = of_kind
        case["parser_of_process"] = 1.0 if of_kind in ("process", "property", "activity") else 0.0
        case["parser_subject"] = 1.0 if root.dep_ in ("nsubj", "nsubjpass") and root.i in ids else 0.0


class Gliner:
    def __init__(self, model: str, threshold: float = 0.3):
        from gliner import GLiNER

        self.model = GLiNER.from_pretrained(model)
        self.threshold = threshold

    def __call__(self, case):
        found = locate(case["mention"], case["sentence"])
        if not found:
            return
        entities = self.model.predict_entities(
            case["sentence"], list(GLINER_LABELS), threshold=self.threshold
        )
        apply_gliner(case, found.start(), found.end(), entities)


def apply_gliner(case, start: int, end: int, entities):
    covering = [e for e in entities if e["start"] <= start and e["end"] >= end]
    case["gliner_other"] = 0.0
    case["gliner_kind"] = None
    if not covering:
        case["gliner_none"] = 1.0
        return
    case["gliner_none"] = 0.0
    best = max(covering, key=lambda e: (e["end"] - e["start"], e["score"]))
    kind = GLINER_LABELS.get(best["label"])
    case["gliner_kind"] = kind
    case["gliner_span"] = case["sentence"][best["start"] : best["end"]]
    longer = (best["end"] - best["start"]) > (end - start)
    if kind not in SITE_KINDS and longer:
        case["gliner_other"] = float(best["score"])


class Retrieval:
    """Name encoder (SapBERT style: [CLS] of the name) over NCIt preferred names."""

    def __init__(self, model: str, kinds: Kinds, batch: int = 256):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).eval()
        self.batch = batch
        names = sorted(set(kinds.preferred))
        self.kinds = np.array([k for _, k in names])
        self.names = [n for n, _ in names]
        self.index = self.embed(self.names).astype(np.float16)

    def embed(self, texts):
        out = []
        for i in range(0, len(texts), self.batch):
            batch = self.tok(texts[i : i + self.batch], padding=True, truncation=True,
                             max_length=25, return_tensors="pt")
            with self.torch.no_grad():
                cls = self.model(**batch).last_hidden_state[:, 0].numpy()
            out.append(cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-9))
        return np.concatenate(out) if out else np.zeros((0, 768))

    def __call__(self, case):
        groups = word_groups(case)
        if not groups:
            case["retrieval_margin"] = -1.0
            return
        q = self.embed(groups).astype(np.float32)
        sims = np.zeros((len(groups), len(self.names)), dtype=np.float32)
        step = 50000
        for i in range(0, len(self.names), step):
            sims[:, i : i + step] = q @ self.index[i : i + step].astype(np.float32).T
        site = np.isin(self.kinds, list(SITE_KINDS))
        best = (-2.0, None, None)
        for g in range(len(groups)):
            other = np.where(~site, sims[g], -1.0)
            j = int(other.argmax())
            margin = float(other[j] - np.where(site, sims[g], -1.0).max())
            if margin > best[0]:
                best = (margin, groups[g], j)
        case["retrieval_margin"] = best[0]
        case["retrieval_group"] = best[1]
        case["retrieval_name"] = self.names[best[2]]
        case["retrieval_kind"] = str(self.kinds[best[2]])


def word_groups(case, left: int = 3, right: int = 4) -> list[str]:
    """Word groups that contain the mention and are longer than it, inside its clause."""
    found = locate(case["mention"], case["sentence"])
    if not found:
        return []
    before = re.split(r"[,;:()\[\]]", case["sentence"][: found.start()])[-1].split()[-left:]
    after = re.split(r"[,;:()\[\].]", case["sentence"][found.end() :])[0].split()[:right]
    mention = case["sentence"][found.start() : found.end()]
    groups = []
    for a in range(len(before) + 1):
        for b in range(len(after) + 1):
            if a == 0 and b == 0:
                continue
            groups.append(" ".join(before[len(before) - a :] + [mention] + after[:b]))
    return groups


PROMPT = """Read the sentence. Split it into the concepts it names, each as the longest phrase that names one concept.
Then answer about the concept whose phrase contains the word «{mention}» (the occurrence marked [[ ]]).

Sentence: {marked}

Answer with JSON only, no other text:
{{"phrase": "<the phrase of that concept>", "type": "<one of: {types}>", "body_site": "<yes if «{mention}» there names a place in a body where something is, else no>"}}"""


def ask_llm(chats: dict, case) -> dict:
    found = locate(case["mention"], case["sentence"])
    if not found:
        return {}
    s = case["sentence"]
    marked = f"{s[: found.start()]}[[{s[found.start() : found.end()]}]]{s[found.end() :]}"
    prompt = PROMPT.format(mention=case["mention"], marked=marked, types=", ".join(LLM_TYPES))
    answers = {}
    for name, chat in chats.items():
        try:
            text = chat(prompt)
            body = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            answers[name] = body
        except Exception as error:  # noqa: BLE001 - a lost answer is counted, not fatal
            answers[name] = {"error": str(error)[:120]}
    return answers


def apply_llm(case, answers: dict):
    good = [a for a in answers.values() if "error" not in a]
    case["llm_answers"] = answers
    if not good:
        case["llm_lost"] = 1.0
        return
    mention = _fold(case["mention"])
    other = not_site = 0
    kinds = []
    for a in good:
        kind = LLM_KIND.get(str(a.get("type", "")).strip().lower(), "conceptual")
        kinds.append(kind)
        longer = len(_fold(str(a.get("phrase", ""))).split()) > len(mention.split())
        other += int(longer and kind not in SITE_KINDS)
        not_site += int(str(a.get("body_site", "")).strip().lower().startswith("no"))
    case["llm_other"] = other / len(good)
    case["llm_not_site"] = not_site / len(good)
    case["llm_kind"] = Counter(kinds).most_common(1)[0][0]


# -- run and report ------------------------------------------------------------------------------

SIGNALS = (
    "head_other", "object", "discourse", "theme", "typo", "parser_other", "parser_of_process",
    "parser_subject", "gliner_other", "gliner_none", "retrieval_margin", "llm_other", "llm_not_site",
)
KIND_FIELDS = ("head_kind", "object_kind", "parser_kind", "gliner_kind", "retrieval_kind", "llm_kind")


def gold_role(case) -> str | None:
    if case["corpus"] != "medmentions":
        return None
    types = {t for label in case.get("labels", ()) for t in label.split("|", 1)[-1].split(",")}
    roles = {ROLE_OF_TYPE[t] for t in types if t in ROLE_OF_TYPE}
    return sorted(roles)[0] if roles else None


def run(args, rows=None, texts=None, cases=None, kinds=None, parser=None, gliner=None,
        retrieval=None, chats=None) -> dict:
    if cases is None:
        rows, texts, cases = collect(args)
    kinds = kinds or Kinds.from_obo(Path(args.ncit))
    failures = {}
    # Words the corpus itself writes three times or more are words, not typos ("underwent").
    frequent = Counter(_fold(w) for t in (texts or {}).values() for w in WORD.findall(t))
    speller = Speller(kinds.vocabulary | {w for w, n in frequent.items() if n >= 3})
    for c in cases:
        arm_head(c, kinds)
        arm_object(c, kinds)
        arm_typo(c, kinds, speller)
    arm_discourse(cases, rows, kinds)
    arm_theme(cases, texts, kinds)
    for tag, model in (("parser", parser), ("gliner", gliner), ("retrieval", retrieval)):
        if model is None:
            continue
        for c in cases:
            try:
                model(c)
            except Exception as error:  # noqa: BLE001 - an arm that fails is reported, not fatal
                failures[tag] = f"{type(error).__name__}: {str(error)[:200]}"
                break
    if chats:
        chosen = llm_cases(cases, args.llm, args.llm_sample)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for c, answers in zip(chosen, pool.map(lambda c: ask_llm(chats, c), chosen), strict=True):
                apply_llm(c, answers)
    return summarise(cases, failures, args)


def llm_cases(cases, scope: str, sample: int):
    if scope == "all":
        return list(cases)
    right = [c for c in cases if c["label"] == 0]
    random.Random(11).shuffle(right)
    return [c for c in cases if c["label"] == 1] + right[:sample]


def votes(c) -> float:
    """Independent cheap readings that say 'another kind of thing' (each counts once)."""
    return float(
        (c.get("head_other", 0) >= 0.6)
        + bool(c.get("object") or c.get("discourse"))
        + (c.get("gliner_other", 0) >= 0.5)
        + (c.get("retrieval_margin", -1) > 0.05)
        + bool(c.get("parser_other") or c.get("parser_of_process"))
    )


def summarise(cases, failures, args) -> dict:
    for c in cases:
        c["votes_cheap"] = votes(c)
    labels = np.array([c["label"] for c in cases])
    report = {
        "n": len(cases),
        "errors": int(labels.sum()),
        "by_corpus": dict(Counter((c["corpus"], c["label"]) for c in cases).most_common()),
        "failures": failures,
        "llm_scope": getattr(args, "llm", None),
        "features": {},
    }
    report["by_corpus"] = {f"{k[0]}:{'error' if k[1] else 'right'}": v for k, v in report["by_corpus"].items()}
    for name in (*SIGNALS, "votes_cheap"):
        have = [c for c in cases if isinstance(c.get(name), float)]
        if not have:
            continue
        s = np.array([c[name] for c in have])
        y = np.array([c["label"] for c in have])
        report["features"][name] = {
            "n": len(have),
            "errors": int(y.sum()),
            "auc": auc(s, y),
            "auc_craft": auc(*_split(have, name, "craft")),
            "auc_medmentions": auc(*_split(have, name, "medmentions")),
            "points": operating_points(s, y),
        }
    # Gate: would "only when uncertain" have opened on the errors?
    def gated(c):
        return bool(c.get("conflicts")) or (c.get("convergence") or 0) < 3

    def phrase_gate(c):
        words, found = phrase_words(c)
        return bool(words) or bool(found and _OF.match(c["sentence"][found.end() :]))

    report["gate"] = {
        "uncertain_errors": sum(gated(c) for c in cases if c["label"]),
        "uncertain_right": sum(gated(c) for c in cases if not c["label"]),
        "phrase_errors": sum(phrase_gate(c) for c in cases if c["label"]),
        "phrase_right": sum(phrase_gate(c) for c in cases if not c["label"]),
        "errors": int(labels.sum()),
        "right": int((labels == 0).sum()),
    }
    # Role: for MedMentions errors, does the kind an arm reads give the gold label's role?
    roles = {}
    for field in KIND_FIELDS:
        match = total = 0
        for c in cases:
            g = gold_role(c)
            if not c["label"] or g is None or not c.get(field):
                continue
            total += 1
            match += int(ROLE_OF_KIND.get(c[field]) == g)
        roles[field] = {"read": total, "same_role_as_gold": match}
    report["roles"] = roles
    keep = ("corpus", "mention", "sentence", "labels", "label", "support", "convergence",
            *SIGNALS, *KIND_FIELDS, "head_word", "object_head", "gliner_span", "retrieval_group",
            "retrieval_name", "typo_hypothesis", "llm_answers", "votes_cheap")
    report["errors_read"] = [{k: c.get(k) for k in keep if k in c} | {"gold_role": gold_role(c)}
                             for c in cases if c["label"]]
    report["right_flagged"] = [
        {k: c.get(k) for k in keep if k in c}
        for c in cases
        if not c["label"] and c["votes_cheap"] >= 2
    ][:100]
    report["cases"] = [{k: c.get(k) for k in keep if k in c} for c in cases]
    return report


def _split(have, name, corpus):
    sub = [c for c in have if c["corpus"] == corpus]
    return np.array([c[name] for c in sub]), np.array([c["label"] for c in sub])


def markdown(report: dict) -> str:
    def f(x):
        return "-" if x is None else f"{x:.3f}"

    g = report["gate"]
    lines = [
        "# Experiment phrase-probe: reading the whole phrase",
        "",
        f"{report['n']} judged links, {report['errors']} errors ({json.dumps(report['by_corpus'])}).",
        f"LLM scope: {report['llm_scope']}. Arms that failed: {json.dumps(report['failures']) or 'none'}.",
        "",
        "## Signals",
        "",
        "| signal | n | errors | AUC | AUC CRAFT | AUC MedMentions | errors caught / right lost at 0, 5, 20, 50 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, v in sorted(report["features"].items(), key=lambda kv: -(kv[1]["auc"] or 0)):
        points = "; ".join("-" if p["best"] is None else f"{p['best'][1]}/{p['best'][2]}" for p in v["points"])
        lines.append(f"| {name} | {v['n']} | {v['errors']} | {f(v['auc'])} | {f(v['auc_craft'])} | {f(v['auc_medmentions'])} | {points} |")
    lines += [
        "",
        "## Gate: only when uncertain?",
        "",
        f"A gate on conflict or convergence < 3 opens on {g['uncertain_errors']} of {g['errors']} errors "
        f"and {g['uncertain_right']} of {g['right']} right links. A gate on 'there is a phrase around the "
        f"mention' opens on {g['phrase_errors']} errors and {g['phrase_right']} right links.",
        "",
        "## Role: would recording the structure with a role agree with the annotator?",
        "",
        "| arm | MedMentions errors it types | kind gives the gold label's role |",
        "|---|---|---|",
    ]
    for field, v in report["roles"].items():
        lines.append(f"| {field} | {v['read']} | {v['same_role_as_gold']} |")
    lines += ["", "## Errors, as each arm reads them", ""]
    for e in report["errors_read"]:
        kinds = ", ".join(f"{k.split('_')[0]}={e.get(k)}" for k in KIND_FIELDS if e.get(k))
        lines.append(f"- {e['corpus']} `{e['mention']}` gold role {e['gold_role']}; {kinds}; votes {e.get('votes_cheap')}: {e['sentence'][:160]}")
    lines += ["", "## Right links with two or more cheap votes (would be lost)", ""]
    for e in report["right_flagged"][:60]:
        kinds = ", ".join(f"{k.split('_')[0]}={e.get(k)}" for k in KIND_FIELDS if e.get(k))
        lines.append(f"- {e['corpus']} `{e['mention']}`; {kinds}: {e['sentence'][:160]}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--craft")
    ap.add_argument("--medmentions")
    ap.add_argument("--ncit", required=True)
    ap.add_argument("--uberon", default=str(DATA / "uberon-basic.obo"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--spacy", help="spaCy English model, e.g. en_core_web_sm")
    ap.add_argument("--gliner", help="GLiNER model id")
    ap.add_argument("--encoder", help="name encoder id for retrieval (SapBERT)")
    ap.add_argument("--llm", choices=("none", "sample", "all"), default="none")
    ap.add_argument("--llm-sample", type=int, default=300, help="right links asked with --llm sample")
    ap.add_argument("--out", default="phrase_probe.json")
    ap.add_argument("--markdown", default="phrase_probe.md")
    args = ap.parse_args(argv)
    kinds = Kinds.from_obo(Path(args.ncit))
    failures_at_load = {}

    def load(tag, factory):
        try:
            return factory()
        except Exception as error:  # noqa: BLE001
            failures_at_load[tag] = f"{type(error).__name__}: {str(error)[:200]}"
            return None

    parser = load("parser", lambda: Parser(args.spacy, kinds)) if args.spacy else None
    gliner = load("gliner", lambda: Gliner(args.gliner)) if args.gliner else None
    retrieval = load("retrieval", lambda: Retrieval(args.encoder, kinds)) if args.encoder else None
    chats = None
    if args.llm != "none":
        key = os.environ.get("OPENROUTER_API_KEY")
        if key:
            from melampo.evaluation import linking_bench as lb
            from verify_probe import CHAT_MODELS, CHAT_PACING

            chats = {n: lb.OpenRouterChat(s, key, max_tokens=300, **CHAT_PACING) for n, s in CHAT_MODELS.items()}
        else:
            failures_at_load["llm"] = "OPENROUTER_API_KEY not set"
    report = run(args, kinds=kinds, parser=parser, gliner=gliner, retrieval=retrieval, chats=chats)
    report["failures"] = {**failures_at_load, **report["failures"]}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), "utf-8")
    Path(args.markdown).write_text(markdown(report), "utf-8")
    print(markdown(report)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
