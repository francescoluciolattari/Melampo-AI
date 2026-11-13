"""Of what is the finding said? The reader and the model read the sentence, each on its own (31 October 2026;
asked from the finding, with the report as context, 1 November 2026).

A report names a structure to describe it ("the liver is enlarged") or to place something else ("enlarged nodes
along the aortic arch"). Before a finding is attached to the structure the linker found, two readers say whether
the sentence is about that structure.

**The reader** (no model) brings what a radiologist knows besides the words, each as evidence that the finding is
*not* of the linked structure:

* the relation the sentence puts the structure in (``spatial_relation``), integrated by the Kintsch reader with
  the lexical evidence (``ci_reader``, second stage: ``landmark``);
* a hypothesis: the structure is named only after a hedge or in a differential diagnosis ("differential
  diagnosis: ... shadow of the left subclavian artery");
* the exam: a structure outside the exam's field of view ("middle lobe" of the lung in a prostate MRI) cannot be
  the site of what the exam shows (``exam_area``: the area read from the exam's name);
* anatomy: the clause describes another body structure first (its *bearer*: "the ureters are dilated up to their
  entry into the bladder", "thickening of the parietal pleura in the lower lobe"), and that structure is neither
  a part of the linked one nor the other way round (UBERON ``part_of``/``is_a``, when UBERON is given).

**The model** is asked the outcome from the finding (``llm_reading.owner_prompt``): what the sentence describes,
the structure it belongs to, how the linked structure stands to it (same, part, reference, wrong, hypothesis),
and yes or no; it is told the exam, the section and the structure the words were linked to, never the reader's
answer. Two models are one stream: they must agree with each other first.

The decision is on the outcome (option b, 14 October 2026). Both say the finding is of the structure: it stays.
Both say it is not: no finding at this site, the clause goes to a person with what it describes. They contradict
each other or one cannot say: the finding stays and goes to a person. **The reader alone never removes a
finding.** Patients' reports are read only by local models (``report_router.local_chat``).
"""

from __future__ import annotations

import contextvars
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE, ELSEWHERE, UNSURE = "here", "elsewhere", "unsure"
# the principle of doubt (7 November 2026): a reading that is not certain is not a question for a person, who knows
# no more than the report says; it stays in the diagnostic path as a finding in doubt, that the conclusion, the
# reader's unambiguous knowledge, other reports, the history or a new exam may close. A person sees only what is
# discordant: the context and the sentence, the sentence and the conclusion, a measure and its cited reference
DOUBT = "doubt"
# what the reader finds that is a discordance of the document, not a doubt of reading
# (9 November 2026, ctrate2: a model citing a row it was not given, or comparing a lesion's size with the organ's
# reference, is a doubt of reading, not a discordance of the document: only the sentence's own judgement against
# the reference it cites is)
DISCORDANCES = ("the_finding_is_in_the_indication", "the_conclusion_gives_it_to_another_structure",
                "its_refers_to_another_structure", "measure_the_reference_says_")
LEXICON = Path(__file__).resolve().parents[3] / "data" / "linking" / "anatomy_lexicon.json"
# Names too general to say which structure a clause describes ("wall", "body", "area", "lobe" alone).
_GENERIC = frozenset("""
body wall part region zone side tissue organ structure system area portion segment lobe cavity space line surface
layer cell cells node nodes margin edge apex base tip lumen content contents floor roof level course branch
branches vessel vessels duct ducts gland glands muscle muscles bone bones joint joints fat skin soft blood mantle
projection chest thorax abdomen pelvis head neck trunk limb
segmento lobo parete corpo area regione zona porzione parte polo base apice sede livello torace addome bacino
""".split())
# UBERON's organ subsets also hold kinds of organ ("compound organ", "trunk region element"): two structures that
# share only such a kind are not parts of one organ.
_KIND_OF_ORGAN = re.compile(r"\b(element|organ|viscus|genitalia)\b|^(exocrine|endocrine|reproductive|digestive system) gland$"
                            r"|^skeletal joint$|^skin of body$")
_WORD = re.compile(r"[A-Za-zÀ-ÿ']+")
EMBRYONIC = "UBERON:0002050"          # embryonic structure
# A report about development (9 November 2026): an obstetric or fetal exam, a pregnancy, a gestational age. Only
# then does a name with a sense of the body formed and one of the embryo ("aortic arch": the arch of the aorta, or
# the pharyngeal arch artery of the embryo) recall the embryo's too. Words of the exam, not of a finding
_DEVELOPMENT = re.compile(
    r"\b(obstetric\w*|ostetric\w*|fetal|fetus|foetal|foetus|fetale|fetali|feto|embryo\w*|embrion\w*|"
    r"pregnan\w*|gravidanz\w*|gestation\w*|gestazion\w*|crown[- ]rump|CRL|"
    r"\d+\s*(?:\+\s*\d+\s*)?(?:weeks?|settimane|wk)\s*(?:of\s+gestation|di\s+gestazione|gestational)|"
    r"(?:settimana|week)\s+\d+\s*(?:di\s+gestazione|of\s+gestation))\b", re.IGNORECASE)
_DEVELOPING: contextvars.ContextVar[bool] = contextvars.ContextVar("melampo_developing", default=False)
# the words that point back to something said before: closed-class words of the two languages
_POINTER = re.compile(r"\b(its|it|they|them|their|this|these|those|the same|same|sua|suo|sue|suoi|esso|essa|"
                      r"essi|esse|stesso|stessa|stessi|stesse|questo|questa|tale|tali)\b", re.IGNORECASE)
_SECTION = re.compile(r"(?:^|[\n.;])\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ /\-]{1,40}):")


@dataclass(frozen=True)
class Ownership:
    verdict: str                                   # HERE, ELSEWHERE, DOUBT, UNSURE
    reader: str = ""                               # the reader's reading and the evidence against the link
    models: dict[str, dict] = field(default_factory=dict)   # what each model answered
    placed: str = ""                               # what the clause describes instead, for the person
    reason: str = ""
    statement: str = ""                            # what the sentence says of the structure (STATEMENT_GROUPS)
    piece: str = ""                                # the words that say it, copied from the sentence
    claims: dict = field(default_factory=dict)     # the claims the readers agree on (question="claims")
    review: str = ""                               # what a person should look at even when the readers agree
    doubt: dict = field(default_factory=dict)      # the doubt the finding keeps, or the one that was closed

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "reader": self.reader, "models": self.models, "placed": self.placed,
                "reason": self.reason, "statement": self.statement, "piece": self.piece, "claims": self.claims,
                "review": self.review, "doubt": self.doubt}


_COORDINATORS = frozenset("and or e ed o oppure".split())


def _find_name(text: str, name: str) -> tuple[int, int] | None:
    """Where a name found by ``Anatomy.find`` (singular, lower case) is in ``text``: its first word to its last."""
    words = name.split()
    spans = [(m.start(), m.end(), _singular(m.group(0))) for m in _WORD.finditer(text)]
    for i in range(len(spans) - len(words) + 1):
        if [w for _, _, w in spans[i:i + len(words)]] == words:
            return spans[i][0], spans[i + len(words) - 1][1]
    return None


def _singular(word: str) -> str:
    w = word.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("s") and not w.endswith(("ss", "is", "us")):
        return w[:-1]
    return w


class Anatomy:
    """The names of body structures, from the linker's lexicon (English and Italian, TotalSegmentator classes)
    and, when given, UBERON (English), with UBERON's ``part_of``/``is_a`` ancestors."""

    def __init__(self, uberon: Any = None, lexicon_path: Path = LEXICON, uberon_names: dict | None = None):
        self.uberon = uberon            # hpo_candidates.Anatomy, or None
        self.names: dict[tuple[str, ...], set[str]] = {}
        self.short: dict[tuple[str, ...], set[str]] = {}
        classes = json.loads(Path(lexicon_path).read_text("utf-8"))["classes"]
        self.regions = {cid: c.get("region", "") for cid, c in classes.items()}
        from .hpo_candidates import site_anchors

        for cid, c in classes.items():
            ids = set(site_anchors(cid)) or {f"class:{cid}"}
            for name in c.get("en", []) + c.get("it", []):
                self._add(name, ids)
        self._up: dict[str, set[str]] = {}
        # UBERON's labels and exact synonyms, not its relational adjectives ("cortical", "dorsal" name no
        # structure in a clause). Only the senses a report of a person can mean (7 November 2026): a reader that
        # takes any sense of a name read "gallbladder corpus" as the corpus of an amphibian's septomaxilla
        # (UBERON:3000645). A term of another taxon (UBERON's identifiers from 2000000 and 3000000 come from the
        # fish and amphibian ontologies) and a name whose number the words would drop ("aortic arch 1") are not
        # read. The structures of the embryo are read (9 November 2026: excluding them all was a boundary fixed
        # beforehand, and it took away "trophoblast", and adult structures UBERON files under the embryo); where a
        # name has a sense of the body formed and one of the embryo, the context chooses (``find``)
        self.developing: set[str] = set()
        self._uberon_names = {u: n for u, n in (uberon_names or {}).items() if self._of_a_person(u)}
        self._heads: set[str] | None = None
        self._part_heads: dict[frozenset, set[str]] = {}
        for uid, names in (uberon_names or {}).items():
            if not self._of_a_person(uid):
                continue
            if self.uberon is not None and EMBRYONIC in self._closure(uid, False):
                self.developing.add(uid)
            for name in names:
                if not re.search(r"\d", name):
                    self._add(name, {uid})
        self.organs = {o for o in getattr(uberon, "organs", set())
                       if not _KIND_OF_ORGAN.search(uberon.labels.get(o, ""))} if uberon is not None else set()

    @staticmethod
    def _of_a_person(uid: str) -> bool:
        return not re.match(r"UBERON:[23]\d{6}$", uid)

    def _in_context(self, ids: set[str], developing: bool | None) -> set[str]:
        """The senses of a name the context recalls: a name with a sense of the body formed and one of the embryo
        recalls the embryo's only in a report about development; a name with one sense keeps it."""
        if developing is None:
            developing = _DEVELOPING.get()
        embryo = ids & self.developing
        if developing or not embryo or embryo == ids:
            return ids
        return ids - embryo

    @classmethod
    def from_uberon(cls, path: Path) -> Anatomy:
        """With UBERON (``uberon.obo``): its names and its ``part_of``/``is_a``."""
        from . import hpo_candidates as hc

        with open(path, encoding="utf-8") as fh:
            uberon = hc.Anatomy.from_obo(fh)
        with open(path, encoding="utf-8") as fh:
            terms = hc.read_obo(fh, "UBERON:")
        names = {uid: [(t.get("name") or [""])[0]] + [hc._quoted(x) for x in t.get("synonym", []) if " EXACT" in x]
                 for uid, t in terms.items()}
        return cls(uberon, uberon_names=names)

    def _add(self, name: str, ids: set[str]) -> None:
        key = tuple(_singular(w) for w in _WORD.findall(name.lower()))
        if not key or (len(key) == 1 and key[0] in _GENERIC):
            return
        if len(key) == 1 and len(key[0]) < 5:
            # a short single name ("pons", "arch") is too often a word of something else to say what a clause
            # describes; it is used only to read a name a model gives (``find(..., short=True)``)
            if len(key[0]) >= 4:
                self.short.setdefault(key, set()).update(ids)
            return
        self.names.setdefault(key, set()).update(ids)

    def find(self, text: str, short: bool = False, developing: bool | None = None) -> list[tuple[str, set[str]]]:
        """Every body structure named in ``text``, longest names first, in order (``short``: also the short
        single names, for a name a model gives), with the senses the context recalls (``developing``: a report
        about development; by default the one being read)."""
        words = [_singular(w) for w in _WORD.findall(text.lower())]
        out, i = [], 0
        while i < len(words):
            for n in range(min(5, len(words) - i), 0, -1):
                ids = self.names.get(tuple(words[i:i + n])) or (
                    self.short.get(tuple(words[i:i + n])) if short and n == 1 else None)
                if ids:
                    out.append((" ".join(words[i:i + n]), self._in_context(ids, developing)))
                    i += n
                    break
            else:
                i += 1
        return out

    def _closure(self, uid: str, kinds_only: bool) -> set[str]:
        key = (uid, kinds_only)
        if key not in self._up:
            seen, stack = {uid}, [uid]
            while stack:
                u = stack.pop()
                wholes = set(self.uberon.part_of.get(u, ())) if kinds_only else set()
                for p in self.uberon.parents.get(u, []):
                    if p not in seen and p not in wholes:
                        seen.add(p)
                        stack.append(p)
            self._up[key] = seen
        return self._up[key]

    def up(self, uid: str) -> set[str]:
        """Every class above ``uid`` (``part_of`` and ``is_a``)."""
        if self.uberon is None or not uid.startswith("UBERON:"):
            return {uid}
        return self._closure(uid, False)

    @staticmethod
    def _head(name: str) -> str:
        """The noun a name of UBERON is a name of: the word before "of" ("left lobe of liver": lobe), else its
        last word ("aortic arch": arch)."""
        words = [_singular(w) for w in _WORD.findall(name.lower())]
        if "of" in words[1:]:
            return words[words.index("of", 1) - 1]
        return words[-1] if words else ""

    def part_of_linked(self, words: str, linked: set[str]) -> bool | None:
        """Do the words name a part the linked structure has (13 November 2026)? Read from UBERON, not from a
        list of words: the parts of a structure are the terms below it by ``part_of``/``is_a``, and what each is
        a part of its kind is the noun its name is a name of ("wall of aorta": a wall). The words name such a
        part when one of their nouns is the noun of a part of the linked structure: "the wall of the arch" of the
        aorta (a wall of the aorta), never "the left lobe medial segment" of the portal vein (UBERON gives the
        portal vein branches, no lobe and no segment). None when UBERON is not at hand, or when the words have no
        noun that names a part of anything: then nothing is said."""
        if self.uberon is None or not self._uberon_names:
            return None
        if self._heads is None:
            self._heads = {self._head(n) for names in self._uberon_names.values() for n in names if n}
            self._heads.discard("")
        nouns = {_singular(w) for w in _WORD.findall(str(words).lower())} & self._heads
        if not nouns:
            return None
        roots = frozenset(x for x in linked if x.startswith("UBERON:"))
        if not roots:
            return None
        if roots not in self._part_heads:
            self._part_heads[roots] = self._heads_of_parts(roots)
        return bool(nouns & self._part_heads[roots])

    def _heads_of_parts(self, roots: frozenset) -> set[str]:
        """The nouns of the parts a structure has: the terms below it (``part_of``/``is_a``), the parts every
        kind it is has (a vein has a wall: ``is_a`` up, then ``part_of`` one step down), and the parts of the
        whole it is a subdivision of, which a subdivision has as well (a segment of the colon has the colon's
        wall: ``part_of`` one step up, then the same)."""
        if not hasattr(self, "_parts_of"):
            self._parts_of: dict[str, set[str]] = {}
            for u, wholes in self.uberon.part_of.items():
                for w in wholes:
                    self._parts_of.setdefault(w, set()).add(u)
        wholes = {w for r in roots for w in self.uberon.part_of.get(r, ())}
        kinds = {k for r in roots | wholes for k in self._closure(r, True)}
        parts = {u for u in self._uberon_names if u not in roots and self.up(u) & roots}
        parts |= {u for k in kinds for u in self._parts_of.get(k, ())}
        parts -= roots
        return {self._head(n) for u in parts for n in self._uberon_names.get(u, []) if n} - {""}

    def organs_of(self, ids: set[str]) -> set[str]:
        """The organs a structure is, or is a part of (UBERON's organ subsets, without the kinds of organ)."""
        return {o for u in ids if u.startswith("UBERON:") for o in self.up(u) | {u} if o in self.organs}

    def other_organ(self, bearer: set[str], linked: set[str]) -> bool:
        """Does the clause describe a structure of another organ than the linked one? The question a radiologist
        asks before coding a finding: to which organ does it belong. True only when both have an organ and the
        organs differ (the ureter and the bladder, a main bronchus and a lobe of the lung, the liver and a lobe of
        the lung); a structure UBERON places in no organ (a vertebral body, a wall, an artery) says nothing."""
        if self.uberon is None or bearer & linked:
            return False
        ob, ol = self.organs_of(bearer), self.organs_of(linked)
        return bool(ob) and bool(ol) and not (ob & ol)


_DIFFERENTIAL = re.compile(r"\b(differential(?: diagnosis| diagnoses)?|ddx|d\.d\.|diagnosi differenziale|"
                           r"diagnostica differenziale)\b", re.I)


def _unflagged_outcome(answer: dict) -> str:
    """What a reading would attach without its "I cannot tell it from the usual": the doubt is kept apart."""
    return claims_outcome({**answer, "state_unsure": False})


_RANK = {"finding": 3, "keep": 2, "aside": 1, "unsure": 0}   # FINDING, KEEP, ASIDE (defined further on)


def _leading_first(said: dict[str, dict]) -> dict[str, dict]:
    """The readings in order: first the one that keeps the most at the structure (a finding, then other claims,
    then another structure's), so that nothing a model read of the structure is lost while in doubt."""
    order = sorted(said, key=lambda m: -_RANK.get(_unflagged_outcome(said[m]), 0))
    return {m: said[m] for m in order}


def _reading_doubt(said: dict[str, dict], valid: dict[str, dict], outcomes: list[str], closed: str) -> dict:
    """The doubt of reading: of what kind, each reading beside the others, and what closed it if anything."""
    unflagged = [_unflagged_outcome(a) for a in valid.values()]
    if len(valid) < len(said):
        kind = "one_reading"
    elif ASIDE in unflagged or any(a.get("same") != "yes" for a in valid.values()):
        kind = "site"
    elif any(a.get("state_unsure") for a in valid.values()) and len(set(unflagged)) == 1:
        kind = "state"
    else:
        kind = "reading"
    out = {"kind": kind, "why": "the_models_read_differently", "readings": _readings(said),
           "closes_with": ["conclusion", "reader", "other_reports", "history", "new_exam"]}
    if closed:
        out["closed_by"] = "conclusion" if "conclusion" in closed else "reader"
    return out


def _readings(said: dict[str, dict]) -> dict:
    """Each model's reading, as the doubt keeps it: none of them is the doubt's own."""
    return {m: {"outcome": claims_outcome(a) if a.get("valid") else "none",
                "observed": [o.get("words", "") for o in a.get("observed", [])],
                "same": a.get("same", "")} for m, a in said.items()}


# what the reader's contradiction says the sentence attaches
_READER_SAYS = {"the_normal_words_state_an_abnormality": "finding", "the_reader_reads_another_owner": "aside",
                "nothing_said_of_it_and_the_reader_reads_another_owner": "aside",
                "the_reader_reads_a_differential": "differential",
                "the_relation_words_only_place_what_is_said_its_own": "aside",
                "the_relation_words_say_near_not_reaching": "aside", "the_finding_is_only_a_measure": "keep",
                "the_finding_is_a_past_fact": "past"}


def _clinical_doubt(claims: dict) -> dict:
    """The doubt the report itself states ("may", "cannot be excluded", a differential): a finding with its
    certainty, kept in the path; only new clinical information closes it, never our reading."""
    words = list(claims.get("uncertainty", []))
    if not claims.get("findings") or not words:
        return {}
    return {"kind": "clinical", "words": words, "about": list(claims.get("uncertain_about", [])),
            "closes_with": ["other_reports", "history", "new_exam"]}


def _one_kind_per_piece(answer: dict, mention: str = "") -> dict:
    """The same words cannot be two claims (7 November 2026: "cannot be evaluated optimally" listed as not
    assessable and also as observed): a piece a model gives another kind is not also observed. And a piece that is
    only a measure, numbers and units with the words of a measure ("measures 7 mm in diameter", "volume: 109 cm3"),
    is a measure, as the question itself says ("a measure goes in measure", 8 November 2026): it is compared with
    the reference values, not read as an observation."""
    other = {w for k in ("normal", "not_assessable", "indication", "past", "device", "procedure", "other_person",
                         "reported") for w in answer.get(k, [])}
    observed = answer.get("observed", [])
    kept = [o for o in observed if o.get("words") not in other
            and not (o.get("of") == "it" and _pure_measure(o.get("words", ""), mention))]
    if len(kept) == len(observed):
        return answer
    moved = [o["words"] for o in observed if o not in kept and o.get("words") not in other]
    have = {(m.get("words") if isinstance(m, dict) else m) for m in answer.get("measure", [])}
    measure = list(answer.get("measure", [])) + [w for w in moved if w not in have]
    return {**answer, "observed": kept, "measure": measure}


def _parallel_places(clause: str, other_at: int, at: int, language: str) -> bool:
    """Are both structures places of what is observed: each introduced by its own relation of containment
    that is not "of" ("in the gallbladder lumen … in the middle zone of the left kidney")? A structure the
    observation is "of" ("thickening of the parietal pleura in the lower lobe") is its bearer, not a place."""
    from .spatial_relation import CONTAINMENT, read_relation

    mine, theirs = read_relation(clause, at, language), read_relation(clause, other_at, language)
    of = {"of", "di", "del", "dello", "della", "dei", "degli", "delle", "dell'"}
    return (mine.kind == CONTAINMENT and theirs.kind == CONTAINMENT
            and (theirs.indicator.split() or [""])[0].lower() not in of)


def _in_a_differential(clause: str, at: int) -> bool:
    """Is the structure named only as one of the explanations a differential diagnosis lists? The scope of the
    differential is its parenthesis, or the rest of the sentence. A hedge on the finding itself ("suspected
    liver lesion") still speaks of that structure, and is not this."""
    for m in _DIFFERENTIAL.finditer(clause[:at]):
        opened = clause.rfind("(", 0, m.start())
        closed = clause.find(")", m.end())
        if opened >= 0 and clause.find(")", opened, m.start()) < 0:
            if closed < 0 or at < closed:
                return True
        else:
            return True
    return False


def report_context(report: str, clause_start: int) -> dict[str, str]:
    """What a radiologist knows of the report before the sentence: the exam (its first sentence that names a
    modality) and the section the sentence is in ("TZ:", "Bladder:").

    A table of measures is neither (2 November 2026: an echocardiogram's "LA: 38 mm; Ao: 33 mm; ..." was given
    to the models as the exam, and "Ao" as the section, and a model read the mitral valve as the aorta): a
    sentence is split at ";" too, one with three numbers or more is a table, not the name of an exam, and a
    label followed by a number ("Ao: 33 mm") is a measure, not a section."""
    from .exam_area import ExamAreas

    areas = ExamAreas.load()
    exam = ""
    if areas.modality is not None:
        # the exam's name: the first sentence that names a modality and a region ("CT of the chest"), else the
        # first that names a modality
        sentences = [x.strip() for x in re.split(r"(?<=[.\n;])\s+", report[:3000]) if x.strip()]
        sentences = [x for x in sentences if len(_NUMBER.findall(x)) < 3]
        named = [x for x in sentences if areas.modality.search(x)]
        exam = next((x for x in named if areas.of(x)), named[0] if named else "")
        exam = " ".join(exam.split())[:120]
    sections = [m for m in _SECTION.finditer(report[:clause_start + 1])
                if not re.match(r"\s*[\d<>~≈]", report[m.end():m.end() + 6])]
    section = sections[-1].group(1).strip() if sections and clause_start - sections[-1].end() < 400 else ""
    # the document around the sentence, as a radiologist reads it (6 November 2026): the kind of section it is in
    # (the referral is not a finding), the report's conclusion (which repeats what matters), and the sentence
    # before it (what "its", "the same" refer to)
    from .report_sections import IMPRESSION, impression_of, section_at

    part = section_at(report, clause_start)
    kind = part.kind if part else ""
    conclusion = impression_of(report) if kind != IMPRESSION else ""
    before = [x for x in re.split(r"(?<=[.!?;\n])\s+", report[max(0, clause_start - 400):clause_start]) if x.strip()]
    previous = " ".join(before[-1].split())[:300] if before else ""
    development = "yes" if _DEVELOPMENT.search(" ".join([exam, report[:3000]])) else ""
    return {"exam": exam, "section": section, "areas": ",".join(sorted(areas.of(report))),
            "section_kind": kind, "conclusion": conclusion, "previous": previous, "development": development}


_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


# -- what the sentence says of the structure (1 November 2026) ------------------------------------------------
# The types of statement, grouped by what the interpreter attaches to the structure: two types of one group
# attach the same thing, so the readers agree when they agree on the group.
FINDING, HISTORY, NORMAL, MEASURE, NOTHING, ASIDE = "finding", "history", "normal", "measure", "nothing", "aside"
STATEMENT_GROUPS = {"finding": FINDING, "involved": FINDING, "history": HISTORY, "normal": NORMAL,
                    "measure": MEASURE, "reference": ASIDE, "other_structure": ASIDE, "hypothesis": ASIDE,
                    "nothing": NOTHING}
# What the reader knows of the kinds of statement, to check the model's type (never to decide it): the words of
# a past fact, of something abnormal, of something normal or absent, and of a measure.
_HISTORY = re.compile(r"\b(history of|known history|status post|s/p|post[- ]?(?:operative|surgical)?\s*(?:resection|"
                      r"surgery|removal)|operated|resected|removed|previous (?:surgery|resection)|in anamnesi|"
                      r"anamnesi|pregress[oaie]|operat[oaie]|asportat[oaie]|resezion[ei] pregress)", re.I)
_ABNORMAL = re.compile(r"\b(enlarg\w*|dilat\w*|thicken\w*|increas\w*|reduc\w*|decreas\w*|hyper\w+|hypo\w+|"
                       r"mass\w*|lesion\w*|nodul\w*|nodes?|cyst\w*|stones?|calcul\w*|calcif\w*|stenos\w*|"
                       r"narrow\w*|abnormal\w*|atroph\w*|hypertroph\w*|ectasi\w*|fractur\w*|effusion|"
                       r"fluid|edema|oedema|opacit\w*|consolidat\w*|infiltrat\w*|herni\w*|tumou?r\w*|"
                       r"ingrandit\w*|aumentat\w*|ridott\w*|ispessit\w*|dilatat\w*|lesion[ei]|nodul[oi]|"
                       r"cist[ei]|calcol[oi]|calcific\w*|stenos[ie]|versamento|frattur\w*|ipo\w+|iper\w+)", re.I)
_NEGATION = re.compile(r"\b(no|not|without|free of|absence of|negative for|non|senza|assenza di|privo di|priva di)"
                       r"\s+(?:\w+\s+){0,3}$", re.I)
_MEASURE_WORDS = frozenset("""
measure measures measured measuring measurement size sized dimension dimensions diameter diameters length long
width wide thickness thick depth deep height volume caliber calibre axis axial transverse longitudinal
anteroposterior craniocaudal cranio caudal bipolar ap tr cc ll sd max maximum maximal approximately approx about
up to of in its the a an is are and x by with mm cm ml cc m s mmhg cm2 mm2 v
misura misurano misure dimensioni diametro diametri lunghezza spessore volume asse assi trasverso longitudinale
anteroposteriore craniocaudale bipolare circa fino a di del della il lo la i gli le è sono e per
""".split())
WINDOW = 320          # characters: a longer sentence (a table of measures) is read in a window around the words


def _words(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[A-Za-zÀ-ÿ]+", text)]


def _pure_measure(piece: str, mention: str) -> bool:
    """Only numbers, units, the words of a measure and the structure's own words. A measure has a number with a
    unit (``sizes_in_text``): "S1" or "L5" is the name of a vertebra, not a measure (2 November 2026)."""
    from ..imaging.findings import sizes_in_text

    if not sizes_in_text(piece):
        return False
    own = set(_words(mention))
    return all(w in _MEASURE_WORDS or w in own or w in _SKIP_SIDES for w in _words(piece))


_SKIP_SIDES = frozenset("right left bilateral upper lower lobe pole segment destro destra sinistro sinistra".split())


def _affirmed_abnormal(piece: str) -> bool:
    for m in _ABNORMAL.finditer(piece):
        if not _NEGATION.search(piece[:m.start()]):
            return True
    return False


def type_check(group: str, piece: str, before: str, mention: str) -> str:
    """Does the type agree with what the reader knows of the piece? Returns the reason it does not, or "".
    ``before``: the few words of the sentence before the piece (a model may copy "thyroid cancer surgery"
    without its "history of")."""
    history = bool(_HISTORY.search(f"{before} {piece}"))
    if group == HISTORY:
        return ""                        # the reader cannot say that a past fact is not one
    if history and group in (FINDING, MEASURE, NORMAL):
        return "the_piece_states_a_past_fact"
    if group == FINDING and _pure_measure(piece, mention):
        return "the_piece_is_only_a_measure"
    if group == MEASURE and _affirmed_abnormal(piece):
        return "the_piece_states_more_than_a_measure"
    if group == NORMAL and _affirmed_abnormal(piece):
        return "the_piece_states_something_abnormal"
    return ""


def window(clause: str, at: int, length: int, size: int = WINDOW) -> tuple[str, int]:
    """A long sentence read around the words: the separators (``;``, ``.``, a newline, a ``:`` of a table) closest
    to ``size // 2`` characters on each side. Returns the window and the words' offset in it."""
    if len(clause) <= size:
        return clause, at
    half = size // 2
    lo = max(0, at - half)
    cut = max(clause.rfind(c, lo, at) for c in ";.\n")
    lo = cut + 1 if cut >= lo else lo
    hi = min(len(clause), at + length + half)
    ends = [i for i in (clause.find(c, at + length, hi) for c in ";.\n") if i >= 0]
    hi = min(ends) if ends else hi
    while lo < at and clause[lo].isspace():
        lo += 1
    return clause[lo:hi], at - lo


KEEP = "keep"     # only other claims: normal, a measure, its past, a device, not assessable (or nothing)


_COORDINATORS_OF_PHRASES = {"en": ("and", "or", "with", "while"), "it": ("e", "ed", "o", "con", "mentre")}


def _own_conjunct(between: str, language: str) -> bool:
    """Does a coordinator in the words between a structure and the mention open a new phrase of its own (at least
    two words that are not the articles or prepositions that lead to the mention: "and hypodense fluid-density
    lesion located in the anterior cortical region in the middle zone of the")? "the stomach and the duodenum",
    "in the liver and in the pancreas" open none."""
    coords = _COORDINATORS_OF_PHRASES.get(language, _COORDINATORS_OF_PHRASES["en"])
    words = _WORD.findall(between.lower())
    idx = [i for i, w in enumerate(words) if w in coords]
    if not idx:
        return False
    rest = words[idx[-1] + 1:]
    light = {"the", "a", "an", "of", "in", "at", "on", "to", "il", "lo", "la", "i", "gli", "le", "di", "del",
             "della", "dello", "dei", "delle", "nel", "nella", "nello", "al", "alla", "a", "in"}
    return sum(1 for w in rest if w not in light and not w.isdigit()) >= 2


_COPY_AGAIN = ("\n\nAsked again: some words of your previous answer to this question are not in the sentence as "
               "it is written. Copy every \"words\" exactly from the sentence; to leave words out between two "
               "pieces, write \" ... \" between them.")


def claims_outcome(answer: dict) -> str:
    """What one model's claims attach: FINDING (a current finding of the structure, or of another structure
    that reaches it), ASIDE (what is observed is of other structures, or the words name another one), KEEP, or
    "unsure"."""
    if not answer.get("valid") or answer.get("state_unsure"):
        return "unsure"
    observed = answer.get("observed") or []
    if answer.get("same") == "no":
        # the marked words name another structure; what the model read of the linked one still counts: what it
        # says reaches it is a finding that involves it (4 November 2026)
        return FINDING if any(o.get("reaches") == "yes" for o in observed) else ASIDE
    if answer.get("same") != "yes":
        return "unsure"
    if any(o.get("of") == "it" or o.get("reaches") == "yes" for o in observed):
        return FINDING
    if observed or answer.get("places"):
        # what is observed is of other structures, or the structure only places something else (13 November 2026)
        return ASIDE
    return KEEP


def uncertainty_readers(valid: dict[str, dict]) -> list[dict]:
    """The answers whose uncertainty is read (11 November 2026): every model's (their union) by default; with
    ``MELAMPO_UNCERTAINTY_FROM`` set to model names (comma separated), only theirs. The choice is measured, never
    assumed: on ctrate5 one model gave the uncertainty to 13 stated findings, the other to 1, and the union lost
    nothing either alone lost; ctrate6 measures the three readings with the rule fixed before (VALIDATION)."""
    import os

    chosen = [m.strip() for m in os.environ.get("MELAMPO_UNCERTAINTY_FROM", "").split(",") if m.strip()]
    if not chosen:
        return list(valid.values())
    picked = [a for m, a in valid.items() if any(c in m for c in chosen)]
    return picked or list(valid.values())     # a chosen model that did not answer: all the readings, never none


def agreed_claims(answers: list[dict], every: list[dict] | None = None) -> dict:
    """The claims every model makes, with the first model's words: its findings (of the structure or reaching
    it), and each other kind that every model fills. The uncertainty is the exception (10 November 2026): what
    any model reads as uncertain stays uncertain, from all the valid answers (``every``), also when one reading
    leads. Leaving a doubt open costs a hypothesis kept open; losing it makes a hypothesis a fact."""
    from ..evaluation.llm_reading import CLAIM_LISTS

    first = answers[0]
    out: dict = {"findings": [o for o in first.get("observed", []) if o.get("of") == "it" or o.get("reaches") == "yes"]}
    others = [o for o in first.get("observed", []) if o not in out["findings"]]
    if others:
        out["of"] = next((o["of"] for o in others if o.get("of") not in ("it", "?")), "")
    for key in CLAIM_LISTS[1:]:
        if key in ("uncertainty", "undetermined", "limits"):
            continue
        if all(a.get(key) for a in answers):
            out[key] = list(first.get(key, []))
    uncertain, about = [], []
    for a in every or answers:
        for w in a.get("uncertainty", []):
            if w not in uncertain:
                uncertain.append(w)
        for u in a.get("uncertain_about", []):
            if u not in about:
                about.append(u)
    if uncertain:
        out["uncertainty"] = uncertain
        out["uncertain_about"] = about
    # what is seen but not identified, and the limits of the exam, accompany the finding: what any model reads
    for key in ("undetermined", "limits"):
        joined = []
        for a in every or answers:
            joined += [w for w in a.get(key, []) if w not in joined]
        if joined:
            out[key] = joined
    if first.get("same") == "no":
        out["of"] = first.get("named", "")
    if first.get("measure_refs"):
        out["measure_refs"] = list(first["measure_refs"])   # checked against the rows before anything is attached
    return out


# placing words that can also mean "in" (6 November 2026): "a lesion in the region of the pancreatic head" may be
# of the head; "in the projection of" on a radiograph says only that the two overlap. The reader does not hold a
# model's reading against them: they are read by the models, as "up to" is (decision of 3 November 2026)
_AMBIGUOUS_PLACE = re.compile(r"\b(region|area|topography|territory|projection|level|zone|regione|area|sede|"
                              r"proiezione|livello|zona|ambito)\b", re.I)


def _relation_kind(words: str, language: str) -> str:
    """The kind of the relation the words express (``spatial_relation``): proximity, projection, direction,
    containment, or ""."""
    from .spatial_relation import read_relation

    text = f"{words.strip()} X"
    return read_relation(text, len(text) - 1, language).kind


_SIDE = re.compile(r"\b(left|right|sinistr[oaie]|destr[oaie]|sx|dx)\b", re.I)


def _side(text: str) -> str:
    sides = {("left" if m.group(1).lower() in ("left", "sx") or m.group(1).lower().startswith("sinistr") else "right")
             for m in _SIDE.finditer(text or "")}
    return sides.pop() if len(sides) == 1 else ""


def laterality_conflict(exam: str, cid: str, mention: str) -> str:
    """The exam names one side and the structure is of the other ("CT of the left femur", "right femur"): an
    inconsistency of the report a person should see, whatever the readers agree on (2 November 2026)."""
    exam_side = _side(exam)
    side = "left" if cid.endswith("_left") else "right" if cid.endswith("_right") else _side(mention)
    if exam_side and side and exam_side != side:
        return f"the_exam_is_of_the_{exam_side}_side_the_sentence_of_the_{side}"
    return ""


class FindingOwner:
    """``reader``: a ``CIReader``; ``chats``: model name -> chat function (local); ``anatomy``: ``Anatomy``.
    ``question``: ``"owner"`` (is the finding of this structure? 1 November 2026) or ``"statement"`` (what does
    the sentence say of this structure? 1 November 2026)."""

    def __init__(self, reader, chats: dict[str, Callable[[str], str]] | None = None, anatomy: Anatomy | None = None,
                 question: str = "owner", anomalies=None, references=None):
        self.reader = reader
        self.chats = dict(chats or {})
        self.anatomy = anatomy if anatomy is not None else Anatomy()
        if question not in ("owner", "statement", "claims"):
            raise ValueError("question must be 'owner', 'statement' or 'claims'")
        self.question = question
        self.anomalies = anomalies          # what_is_said.AnomalyWords: what the reader knows of abnormalities
        self.references = references        # reference_values.ReferenceValues: what a measure is compared with

    # -- the reader ---------------------------------------------------------------------------------------
    def _reader(self, mention: str, clause: str, at: int, language: str, cid: str,
                context: dict) -> tuple[str, str, str]:
        """(verdict, evidence, what the clause describes instead)."""
        from .hpo_candidates import site_anchors
        from .spatial_relation import read_relation

        against = []
        placed = read_relation(clause, at, language).placed
        if self.reader is not None:
            got = self.reader.read(mention, clause, at, language=language)
            if got.decision == "landmark":
                against.append("landmark")
        if _in_a_differential(clause, at):
            against.append("named_only_in_a_differential")
        areas = frozenset(a for a in (context.get("areas") or "").split(",") if a)
        if areas and cid:
            from .exam_area import OUTSIDE, ExamAreas

            if ExamAreas.load().relation(self.anatomy.regions.get(cid, ""), areas) == OUTSIDE:
                against.append(f"outside_the_exam:{'+'.join(sorted(areas))}")
        if cid:
            linked = set(site_anchors(cid)) | {f"class:{cid}"}
            # the words of the mention's own segment: a list ("the liver, the pancreas") or a coordinated clause
            # ("the duct measures 2 mm, and the portal vein") names other structures beside it, not its bearer
            # ... and only in the mention's own sentence (a line of a report can hold two)
            before = re.split(r"[,;/]|[.?!](?=\s)", clause[:at])[-1]
            base = at - len(before)
            after = clause[at + len(mention): at + len(mention) + 30]
            named = self.anatomy.find(before)
            after_words = _WORD.findall(after)[:2]
            # a structure coordinated with the mention ("the stomach and the duodenum", "in the liver ... and in
            # the pancreas") shares what the sentence says of it: it is not the bearer of the mention's finding
            # (``what_is_said.conjuncts``, 2 November 2026)
            head = [] if after_words[:1] and after_words[0].lower() in _COORDINATORS else \
                self.anatomy.find(" ".join(after_words))
            first = []
            if named:
                from .what_is_said import conjuncts

                name = named[0][0]
                where = _find_name(before, name)
                if where is None or not conjuncts(clause, (base + where[0], base + where[1]), (at, at + len(mention))):
                    first = named[:1]
                if first and where is not None and _own_conjunct(before[where[1]:], language):
                    # "lesions in the left lobe of the liver and hypodense fluid-density lesion ... of the left
                    # kidney": a coordinator between them opens a new phrase with its own head before the mention;
                    # the structure before it is in the other conjunct, not the bearer (13 November 2026, ctrate8
                    # 14068-en-1)
                    first = []
                if first and where is not None and _parallel_places(clause, base + where[0], at, language):
                    # "stones in the gallbladder lumen and, of 3 mm, in the middle zone of the left kidney": each
                    # structure has its own place, and what is observed is in both, one to many, as a reader
                    # holds them in parallel (7 November 2026). The structure before is not the bearer instead
                    first = []
            for name, ids in ([head[0]] if head and after[:1] == " " else []) + first:
                if self.anatomy.other_organ(ids, linked):
                    against.append(f"describes:{name}")
                    placed = placed or name
                    break
        if against:
            return ELSEWHERE, ",".join(against), placed
        return HERE, "", ""

    # -- the models ---------------------------------------------------------------------------------------
    def _models(self, mention: str, clause: str, at: int, language: str, linked: str,
                context: dict) -> tuple[str, dict, str]:
        from ..evaluation.llm_reading import owner_prompt, read_owner

        name = {"it": "Italian", "en": "English"}.get(language, "")
        prompt = owner_prompt(clause, at, at + len(mention), linked, exam=context.get("exam", ""),
                              section=context.get("section", ""), language=name)
        said: dict[str, dict] = {}
        for model, chat in self.chats.items():
            try:
                said[model] = read_owner(chat(prompt))
            except Exception as error:  # noqa: BLE001 -- a lost answer is "cannot say", never a verdict
                said[model] = {"answer": "uncertain", "linked": "uncertain", "valid": False,
                               "error": type(error).__name__,
                               "error_text": str(error)[:160]}
        verdicts = {{"yes": HERE, "no": ELSEWHERE}.get(a.get("answer"), UNSURE) for a in said.values()}
        verdict = verdicts.pop() if len(verdicts) == 1 else UNSURE      # two models are one stream
        placed = next((str(a.get("finding") or "") for a in said.values() if a.get("answer") == "no"), "")
        return verdict, said, placed

    def _statements(self, mention: str, clause: str, at: int, language: str, cid: str,
                    context: dict) -> dict[str, dict]:
        from ..evaluation.llm_reading import read_statement, statement_prompt

        name = {"it": "Italian", "en": "English"}.get(language, "")
        names = _class_names().get(cid, []) if cid else []
        linked = names[0] if names else (cid.replace("_", " ") or mention)
        prompt = statement_prompt(clause, at, at + len(mention), linked, linked_names=names,
                                  exam=context.get("exam", ""), section=context.get("section", ""), language=name)
        said: dict[str, dict] = {}
        for model, chat in self.chats.items():
            try:
                said[model] = read_statement(chat(prompt), clause)
            except Exception as error:  # noqa: BLE001 -- a lost answer is "cannot say", never a verdict
                said[model] = {"type": "uncertain", "piece": "", "of": "", "found": False, "valid": False,
                               "error": type(error).__name__,
                               "error_text": str(error)[:160]}
        return said

    def _by_statement(self, mention: str, clause: str, at: int, language: str, cid: str,
                      context: dict) -> Ownership:
        """What the sentence says of the structure. The models, one stream, must agree on the group of the type
        (a "nothing" gives way to the other's type); the reader checks each piece against the sentence (in
        ``read_statement``) and each type against what it knows (``type_check``), and brings its evidence that
        the structure only places something else. The reader and the models must agree; when they do not, a
        person decides, and the finding stays as it is. The reader alone, or the models alone, never set a
        finding aside."""
        text, where = window(clause, at, len(mention))
        reader, evidence, placed = self._reader(mention, text, where, language, cid, context)
        said = self._statements(mention, text, where, language, cid, context)
        groups = [STATEMENT_GROUPS.get(a.get("type", ""), "") for a in said.values()]
        stated = [g for g in groups if g and g != NOTHING]
        if "" in groups or len(set(stated)) > 1:
            return Ownership(UNSURE, evidence, said, placed, "the_models_cannot_say_or_disagree")
        group = stated[0] if stated else NOTHING
        pieces = [a.get("piece", "") for a in said.values() if STATEMENT_GROUPS.get(a.get("type", "")) == group]
        piece = min(pieces, key=len) if pieces else ""
        if group == ASIDE:
            if reader != ELSEWHERE:
                return Ownership(UNSURE, evidence, said, placed or next(
                    (a.get("of", "") for a in said.values() if a.get("of")), ""), "reader_and_model_disagree")
            of = next((a.get("of", "") for a in said.values() if a.get("of")), "")
            return Ownership(ELSEWHERE, evidence, said, of or placed, "reader_and_model_agree", ASIDE, piece)
        if reader == ELSEWHERE and group != NOTHING:
            return Ownership(UNSURE, evidence, said, placed, "reader_and_model_disagree", group, piece)
        for answer in said.values():
            p = answer.get("piece", "")
            if STATEMENT_GROUPS.get(answer.get("type", "")) != group or not p:
                continue
            at_piece = text.lower().find(p.lower())
            before = " ".join(_words(text[:max(at_piece, 0)])[-4:]) if at_piece > 0 else ""
            why = type_check(group, p, before, mention)
            if why:
                return Ownership(UNSURE, f"{evidence},{why}".strip(","), said, placed, "reader_and_model_disagree",
                                 group, piece)
        return Ownership(HERE, evidence, said, "", "reader_and_model_agree", group, piece)

    # -- claim by claim (2 November 2026) ------------------------------------------------------------------
    def _claim_answers(self, mention: str, clause: str, at: int, language: str, cid: str,
                       context: dict) -> dict[str, dict]:
        from ..evaluation.llm_reading import claims_prompt, read_claims
        from .reading_memory import active_examples, active_manual

        name = {"it": "Italian", "en": "English"}.get(language, "")
        names = _class_names().get(cid, []) if cid else []
        linked = names[0] if names else (cid.replace("_", " ") or mention)
        rows = self._reference_rows(cid, context)
        passages = [f"[{p.get('id', '')}] {p.get('title', '')}, {p.get('section', '')}: {p.get('text', '')}"
                    for p in (self.references.passages_for(rows) if self.references else [])]
        prompt = claims_prompt(clause, at, at + len(mention), linked, linked_names=names,
                               exam=context.get("exam", ""), section=context.get("section", ""), language=name,
                               references=[r.line() for r in rows], passages=passages,
                               section_kind=context.get("section_kind", ""),
                               conclusion=context.get("conclusion", ""), previous=context.get("previous", ""),
                               manual=active_manual(), examples=active_examples(clause, cid))
        said: dict[str, dict] = {}
        for model, chat in self.chats.items():
            try:
                said[model] = read_claims(chat(prompt), clause, mention, context.get("previous", ""),
                                          context.get("conclusion", ""))
                if said[model].get("dropped"):
                    # words not in the sentence as written (13 November 2026, ctrate8 22424-en-2: a model joined
                    # pieces of a list into words the sentence does not hold, and its reading was empty): the
                    # question is asked once more, saying why; the answer with fewer such words is kept
                    again = read_claims(chat(prompt + _COPY_AGAIN), clause, mention, context.get("previous", ""),
                                        context.get("conclusion", ""))
                    if again.get("valid") and again.get("dropped", 0) < said[model]["dropped"]:
                        said[model] = {**again, "asked_again": True}
            except Exception as error:  # noqa: BLE001 -- a lost answer is "cannot say", never a verdict
                said[model] = {"valid": False, "same": "unsure", "error": type(error).__name__,
                               "error_text": str(error)[:160]}
        return self._with_anatomy(said, mention, cid, language)

    def _phenotype_of_linked(self, findings: list[dict], cid: str) -> bool:
        """Does HPO give what is observed to the linked structure? The words, and what they are said to be, are
        read for the name or an EXACT synonym of a phenotypic abnormality of HPO; its structure is the UBERON
        term of its logical definition (``class_kinds.phenotypes_in``). True when that structure is the linked
        one or a whole it is part of. Without HPO or UBERON, False: nothing is read."""
        if self.anatomy is None or not cid:
            return False
        from .class_kinds import phenotypes_in
        from .hpo_candidates import site_anchors

        linked = {x for x in site_anchors(cid) if x.startswith("UBERON:")}
        if not linked:
            return False
        around = set(linked)
        for x in linked:
            around |= self.anatomy.up(x)
        for o in findings:
            for _, sites in phenotypes_in(f"{o.get('words', '')} ; {o.get('said_to_be', '')}").items():
                if sites & around:
                    return True
        return False

    def _reference_rows(self, cid: str, context: dict) -> list:
        """The reference values for the structure and the exam's technique (all techniques when unknown)."""
        if not self.references or not cid:
            return []
        return self.references.rows_for(cid, context.get("modality", ""), context.get("sex", ""))

    def _with_anatomy(self, said: dict[str, dict], mention: str, cid: str, language: str = "en") -> dict[str, dict]:
        """Whether a structure the model names is the linked one is anatomy, and anatomy is the reader's: a
        part of the linked structure is the linked structure ("the wall of the arch" of the aorta, the pons of
        the brain, the neck of the femur), and so is the whole it is a part of (2 November 2026: the models
        answered "another structure" for the parts, 20 of the 31 findings sent to a person). The model's words
        are kept; ``of_by_the_reader`` and ``same_by_the_reader`` say where the reader read them."""
        if not cid:
            return said
        from .hpo_candidates import site_anchors

        linked = set(site_anchors(cid)) | {f"class:{cid}"}
        from .what_is_said import negated

        for answer in said.values():
            if not answer.get("valid"):
                continue
            kept = []
            for o in answer.get("observed", []):
                if negated(o.get("words", ""), language, mention):
                    # "not enlarged" is not an observation but an absence: the reader reads the negation
                    answer.setdefault("normal", []).append(o["words"])
                    answer.setdefault("moved_by_the_reader", []).append(o["words"])
                    continue
                kept.append(o)
                if o.get("of") in ("it", "?", ""):
                    continue
                if self.same_structure(o["of"], linked):
                    o["of_by_the_reader"], o["of"] = o["of"], "it"
                elif not self.anatomy.find(o["of"], short=True) and self.anatomy.part_of_linked(o["of"], linked):
                    # "the wall of the arch": a part, and no other structure named, is a part of this one when
                    # UBERON gives this one such a part (13 November 2026: a fixed list of part words made "the
                    # left lobe medial segment" of the liver a part of the portal vein, ctrate6 1063-en-2)
                    o["of_by_the_reader"], o["of"] = o["of"], "it"
            answer["observed"] = kept
            if answer.get("same") == "no":
                named = answer.get("named") or answer.get("marked") or mention
                if self.same_structure(named, linked) or self.same_structure(mention, linked):
                    answer["same_by_the_reader"], answer["same"] = "no", "yes"
        return said

    def same_structure(self, words: str, linked: set[str]) -> bool:
        """Do the words name the linked structure, a part of it, or a whole it is a part of (UBERON
        ``part_of``/``is_a``, or the linker's lexicon)? Unknown words: no."""
        found = self.anatomy.find(words, short=True)
        if not found:
            return False
        for _, ids in found:
            if ids & linked:
                return True
            uber = {u for u in ids if u.startswith("UBERON:")}
            for u in uber:
                if self.anatomy.up(u) & linked:
                    return True          # a part (or a kind) of the linked structure
            for l in (x for x in linked if x.startswith("UBERON:")):
                if self.anatomy.up(l) & uber:
                    return True          # the linked structure is a part of it
        return False

    def _by_claims(self, mention: str, clause: str, at: int, language: str, cid: str, context: dict) -> Ownership:
        """What the sentence states of the structure, claim by claim, each with its words (``read_claims``).
        Nothing is chosen from a list: the readers agree when they would attach the same thing (a current
        finding at the structure, nothing at it because the finding is of another structure, or only other
        claims: normal, a measure, its past, a device, not assessable). A finding is certain when two sources
        of different mechanism hold it: the two models agreeing (one stream) and the reader not contradicting
        them, or one model's reading confirmed by the conclusion or by the reader's unambiguous knowledge.
        Otherwise it is a finding in doubt (``DOUBT``, 7 November 2026), kept in the diagnostic path with every
        reading beside it, never set aside; a person sees only a discordance of the document (``DISCORDANCES``)."""
        text, where = window(clause, at, len(mention))
        reader, evidence, placed = self._reader(mention, text, where, language, cid, context)
        said = self._claim_answers(mention, text, where, language, cid, context)
        said = {m: _one_kind_per_piece(a, mention) for m, a in said.items()}
        review = laterality_conflict(context.get("exam", ""), cid, mention)
        valid = {m: a for m, a in said.items() if a.get("valid")}
        if not valid:
            # no reading at all (the answers were lost or unreadable): nothing to keep in doubt, ask again
            return Ownership(UNSURE, evidence, said, placed, "the_models_cannot_say_or_disagree", review=review)
        outcomes = [claims_outcome(a) for a in valid.values()]
        doubt: dict = {}
        if len(valid) < len(said) or "unsure" in outcomes or len(set(outcomes)) != 1:
            # the two models do not read the same: a doubt of reading. The reading that keeps the most at the
            # structure leads (a finding, then other claims, then another structure's), the others are kept
            # beside it; the conclusion, then the reader's unambiguous knowledge, may close it
            valid = _leading_first(valid)
            outcomes = [claims_outcome(a) for a in valid.values()]
            closed = self._closed_by_conclusion(valid, outcomes, language, cid, context)
            if not closed:
                valid, closed = self._arbitrate(valid, outcomes, language, cid)
            doubt = _reading_doubt(said, valid, outcomes, closed)
            if closed:
                evidence = f"{evidence},{closed}".strip(",")
                outcomes = [FINDING]
        outcome = _unflagged_outcome(next(iter(valid.values()))) if doubt and not doubt.get("closed_by") \
            else outcomes[0]
        answers = list(valid.values()) if not doubt or doubt.get("closed_by") else list(valid.values())[:1]
        claims = agreed_claims(answers, uncertainty_readers(valid))
        piece = " … ".join(o["words"] for o in claims.get("findings", []))
        open_doubt = bool(doubt) and not doubt.get("closed_by")
        if outcome == ASIDE:
            relations = [o.get("relation", "") for a in answers for o in a.get("observed", []) if o.get("of") != "it"]
            near = any(_relation_kind(r, language) in ("proximity", "projection") for r in relations if r)
            # every model says the structure only places something, and none reads a finding of it (13 November
            # 2026, ctrate8: "in close proximity to the tail of the pancreas", "at the level of the T10 and T11
            # vertebral corpora"): what the structure was named for is said, by every reading
            places = all(a.get("places") and not any(o.get("of") == "it" for o in a.get("observed", []))
                         for a in answers)
            # the reader's own reading of the relation places the same structure the models give the finding to
            # ("observed in the pleura at the level of the lingular segment": the reader's placed is the pleura,
            # the models' "of" is the pleura): two mechanisms agree (13 November 2026, ctrate8 21230-en-3)
            owners = {w for a in answers for o in a.get("observed", []) if o.get("of") not in ("it", "?", "")
                      for w in _WORD.findall(o["of"].lower()) if len(w) > 3}
            agree_place = bool(placed) and bool(owners & {w for w in _WORD.findall(placed.lower()) if len(w) > 3})
            if open_doubt or (reader != ELSEWHERE and not near and not places and not agree_place):
                # the finding is never set aside on a reading in doubt: it stays where the linker put it, with
                # the reading that would give it to another structure beside it
                doubt = doubt or {"kind": "site", "why": "reader_and_model_disagree", "readings": {}}
                return Ownership(DOUBT, evidence, said, placed or claims.get("of", ""), "reader_and_model_disagree",
                                 ASIDE, piece, claims, review, {**doubt, "kind": "site"})
            why = evidence or "the_relation_words_place_it"
            return Ownership(ELSEWHERE, why, said, claims.get("of", "") or placed, "reader_and_model_agree", ASIDE,
                             piece, claims, review, doubt)
        why = self._contradiction(outcome, claims, evidence, text, language, cid, mention, answers)
        why = why or self._reference_check(answers, cid, context)
        why = why or self._document_check(outcome, claims, answers, evidence, cid, context)
        if why == "nothing_said_of_it_and_the_reader_reads_another_owner" and not open_doubt:
            # the models read nothing of the structure and the reader reads that the sentence describes another:
            # two mechanisms say the same, it only places what is said (13 November 2026, ctrate8 13350-en-3)
            return Ownership(ELSEWHERE, f"{evidence},{why}".strip(","), said, placed, "reader_and_model_agree",
                             ASIDE, piece, claims, review, doubt)
        if why and why.startswith(DISCORDANCES):
            # a discordance of the document: a person clarifies it, the finding stays
            return Ownership(UNSURE, f"{evidence},{why}".strip(","), said, placed, "the_document_is_discordant",
                             outcome, piece, claims, review, doubt)
        if why:
            # the reader and the models read the sentence differently: a doubt of reading, kept in the path. The
            # reader's reading is one of the doubt's readings, beside the models': none of them leads (8 November
            # 2026: in the doubts of the reader its reading was often the right one)
            doubt = {**(doubt if open_doubt else {}), "kind": doubt.get("kind", "reader") if open_doubt else "reader",
                     "why": why}
            doubt["readings"] = {**doubt.get("readings", _readings(said)), "reader": {"outcome": _READER_SAYS.get(
                why, "other"), "why": why}}
            return Ownership(DOUBT, f"{evidence},{why}".strip(","), said, placed, "reader_and_model_disagree",
                             outcome, piece, claims, review, doubt)
        if open_doubt:
            return Ownership(DOUBT, evidence, said, placed, "the_models_read_differently", outcome, piece, claims,
                             review, doubt)
        clinical = _clinical_doubt(claims)
        return Ownership(HERE, evidence, said, "", "reader_and_model_agree", outcome, piece, claims, review,
                         doubt or clinical)

    def _closed_by_conclusion(self, said: dict[str, dict], outcomes: list[str], language: str, cid: str,
                              context: dict) -> str:
        """A doubt of reading closed by the report's conclusion (7 November 2026), as a radiologist reads it: the
        leading reading is a finding of the structure, every model reads that the conclusion repeats what is
        observed here, and the reader knows, without ambiguity, that the conclusion names this structure and
        states a deviation of its tissue (CT-RATE: "43 mm, larger than normal", the conclusion "aneurysmal
        dilatation of the ascending aorta"). Two sources of different mechanism: then the finding is certain."""
        from .hpo_candidates import site_anchors
        from .what_is_said import stated_anomalies

        concl = context.get("conclusion", "")
        if not concl or self.anomalies is None or self.anatomy is None or not outcomes or outcomes[0] != FINDING:
            return ""
        if not all(a.get("conclusion") == "same" for a in said.values()):
            return ""
        # the conclusion says that there is a finding, not whose it is (9 November 2026, ctrate2: "pleural
        # effusion ... at the level of the lower lobe", one model giving it to the effusion's own place): a doubt
        # of the structure is not closed by it
        if any(a.get("same") != "yes" or _unflagged_outcome(a) == ASIDE
               or any(o.get("of") not in ("it", "?", "") for o in a.get("observed", [])) for a in said.values()):
            return ""
        linked = set(site_anchors(cid)) | {f"class:{cid}"} if cid else set()
        if not self.same_structure(concl, linked):
            return ""
        if not stated_anomalies(concl, language, self.anomalies, self.anomalies.sites_of(cid)):
            return ""
        return "the_conclusion_confirms_the_finding_one_model_adds"

    def _arbitrate(self, said: dict[str, dict], outcomes: list[str], language: str, cid: str):
        """When the two models' readings are nested (7 November 2026): one reads a finding, the other the same
        claims without it, or the same piece without being sure it differs from the usual. Most of the models'
        disagreements are of this kind (CT-RATE: "43 mm, larger than normal" read as a measure only by one;
        "it may belong to a thyroid nodule" read as a hedge only). As a third reader, the reader decides only the
        piece they differ on, and only with what it knows without ambiguity: a deviation that HPO and PATO know
        for this tissue in the words of that piece. Then the doubt is closed and the finding is certain; otherwise
        it stays in doubt. The reader never takes away what a model read: it can only confirm the larger one."""
        from .what_is_said import stated_anomalies

        if self.anomalies is None or set(outcomes) - {FINDING, KEEP, "unsure"} or FINDING not in outcomes:
            return said, ""
        sites = self.anomalies.sites_of(cid)
        larger = [m for m, o in zip(said, outcomes) if o == FINDING]
        others = [m for m, o in zip(said, outcomes) if o != FINDING]
        for m in others:
            a = said[m]
            if not a.get("valid") or a.get("same") != "yes":
                return said, ""
        pieces = [o["words"] for m in larger for o in said[m].get("observed", []) if o.get("of") == "it"]
        held = {o["words"] for m in others for o in said[m].get("observed", []) if not said[m].get("state_unsure")}
        disputed = [w for w in pieces if w not in held]
        # the other model read the same words as something else (normal, not assessable, its past): that is a
        # disagreement, not a reading that leaves a piece out. A differential is not something else: it says
        # what the finding may be, and a finding named with its alternatives is still a finding
        read_else = [w for m in others for k in ("normal", "not_assessable", "past", "device", "indication",
                                                 "procedure", "other_person", "reported") for w in said[m].get(k, [])]
        if any(w.lower() in e.lower() or e.lower() in w.lower() for w in disputed for e in read_else):
            return said, ""
        if not disputed or not any(stated_anomalies(w, language, self.anomalies, sites) for w in disputed):
            return said, ""
        ordered = {m: said[m] for m in larger} | {m: said[m] for m in others}
        return ordered, "the_reader_confirms_the_finding_one_model_adds"

    def _document_check(self, outcome: str, claims: dict, answers: list[dict], evidence: str, cid: str,
                        context: dict) -> str:
        """What the report around the sentence says, as a radiologist reads it (6 November 2026). Each reason
        sends the sentence to a person; none sets a finding aside:

        * a finding read in a section of indication: the referral's question is not a finding of the exam;
        * the report's conclusion gives the finding to another structure;
        * "its", "the same" of a finding said the structure's own refer to another structure;
        * a measure the models compare with the structure's reference is part of a lesion they observe;
        * the models read nothing of the structure while the reader reads that the sentence describes another:
          "nothing" leaves the linker's finding where it is (claims3: the dilated ureter left on the bladder)."""
        from .report_sections import INDICATION

        linked = set()
        if self.anatomy is not None and cid:
            from .hpo_candidates import site_anchors

            linked = set(site_anchors(cid)) | {f"class:{cid}"}

        def other(words: str) -> bool:
            """A structure of another organ, read in the senses the report can mean (7 November 2026): the
            name is first recalled in the context of the linked structure, as memory recalls selectively; a
            part, the whole or another part of the same organ ("lung parenchyma" for a lobe, another lobe) is
            not another structure, and a name the reader does not know says nothing."""
            if not words or self.anatomy is None:
                return False
            found = self.anatomy.find(words, short=True)
            if not found or self.same_structure(words, linked):
                return False
            return any(self.anatomy.other_organ(ids, linked) for _, ids in found)

        if outcome == FINDING and context.get("section_kind") == INDICATION:
            return "the_finding_is_in_the_indication"
        for a in answers:
            if outcome == FINDING and a.get("conclusion") == "other" and other(a.get("conclusion_of", "")):
                return "the_conclusion_gives_it_to_another_structure"
            for o in a.get("observed", []):
                # only what a pronoun of the observed words points to ("its origin", "the same"): a model may
                # fill the field without one, and that says nothing
                if o.get("of") == "it" and _POINTER.search(o.get("words", "")) and other(o.get("refers_to", "")):
                    return "its_refers_to_another_structure"
            for ref in a.get("measure_refs", []):
                if ref.get("of") == "it":
                    # the model said the measure is of the structure itself (13 November 2026, ctrate8 18982-en-1:
                    # "measured as 140 mm and increased", the judgement copied with its number among the observed)
                    continue
                if any(ref["words"] in o.get("words", "") or o.get("words", "") in ref["words"]
                       for o in a.get("observed", []) if o.get("words")):
                    return "the_measure_is_of_what_is_observed"
        if outcome == KEEP and not any(claims.get(k) for k in ("normal", "measure", "past", "device", "procedure",
                                                                 "indication", "not_assessable", "other_person",
                                                                 "reported")) \
                and any(e.startswith(("describes:", "landmark", "outside_the_exam")) for e in evidence.split(",")):
            return "nothing_said_of_it_and_the_reader_reads_another_owner"
        return ""

    def _reference_check(self, answers: list[dict], cid: str, context: dict) -> str:
        """Every comparison of a measure a model states is recomputed with the row it cites and the number the
        sentence writes (``reference_values.check``): a row not given, a number in another unit or a comparison
        that does not hold send the sentence to a person (4 November 2026)."""
        from .reference_values import check

        rows = self._reference_rows(cid, context)
        for a in answers:
            for ref in a.get("measure_refs", []):
                why = check(ref["words"], ref["reference"], ref["against"], rows)
                if why:
                    return f"measure_{why}"
        return ""

    def _contradiction(self, outcome: str, claims: dict, evidence: str, text: str, language: str, cid: str,
                       mention: str, answers: list[dict] | None = None) -> str:
        """Does what the reader knows contradict the claims? The reason, or "". It never decides a claim: a
        contradiction sends the sentence to a person."""
        from .what_is_said import past, stated_anomalies_where

        sites = self.anomalies.sites_of(cid) if self.anomalies else set()
        for w in claims.get("normal", []):
            # what the models compared with the usual and found the same, or call normal, while the reader knows
            # the words as a deviation of this tissue (HPO, PATO): the two readings do not correspond (4 November
            # 2026). The reader knows fewer deviations than a radiologist, so it never confirms a finding alone.
            # The words are read where they stand in the sentence (13 November 2026): "tumoral wall thickening",
            # copied out of "no significant tumoral wall thickening was detected", is under its negation there
            if stated_anomalies_where(text, w, language, self.anomalies, sites):
                return "the_normal_words_state_an_abnormality"
        if outcome == FINDING:
            findings = claims.get("findings", [])
            own = [o for o in findings if o.get("of") == "it"]
            # what is observed is, by HPO, an abnormality of this very structure (13 November 2026, ctrate8: "an
            # accessory spleen ... anterior to the spleen", "in the vicinity of the spleen hilum ... compatible with
            # the accessory spleen"): the words that place it do not make it another structure's
            hpo_own = bool(own) and self._phenotype_of_linked(own, cid)
            if own and not hpo_own and any(e.startswith(("landmark", "describes:", "outside_the_exam"))
                                           for e in evidence.split(",")):
                return "the_reader_reads_another_owner"
            if own and "named_only_in_a_differential" in evidence and not claims.get("uncertainty"):
                return "the_reader_reads_a_differential"
            for a in answers or [claims]:
                mine = [o for o in a.get("observed", a.get("findings", [])) if o.get("of") == "it"]
                if mine and not hpo_own and all(o.get("relation") and _relation_kind(o["relation"], language) == "proximity"
                                and not _AMBIGUOUS_PLACE.search(o["relation"]) for o in mine):
                    # "a nodule beneath the skull plate", "a focus in the topography of the pancreatic head": the
                    # model says the finding is the structure's own, and its own words only place it there. Each
                    # model's answer is checked, because one that contradicts itself is not agreement; one placed
                    # observation beside a finding of the structure ("enlarged, with blurred fat planes around
                    # it") changes nothing that is attached (3 November 2026)
                    return "the_relation_words_only_place_what_is_said_its_own"
            for o in findings:
                if o.get("relation") and o.get("of") != "it" and _relation_kind(o["relation"], language) in (
                        "proximity", "projection"):
                    # near it or superimposed on it: the words say that it does not reach the structure. Words
                    # that can say either ("up to", "reaching": a lesion extending up to the cord touches it,
                    # fluid reaching the seventh rib on a radiograph gives a height) are the model's to read
                    # in the sentence; the reader knows nothing of them and does not contradict
                    return "the_relation_words_say_near_not_reaching"
            words = [o["words"] for o in findings]
            if words and all(_pure_measure(w, mention) for w in words):
                return "the_finding_is_only_a_measure"
            if words and all(past(w, language) for w in words):
                return "the_finding_is_a_past_fact"
            return ""
        # only other claims: the reader looks for an abnormality they state
        for key in ("normal", "measure"):
            for w in claims.get(key, []):
                if stated_anomalies_where(text, w, language, self.anomalies, sites):
                    return f"the_{key}_words_state_an_abnormality"
        return ""

    def __call__(self, mention: str, clause: str, at: int, language: str = "en", cid: str = "",
                 context: dict | None = None) -> Ownership:
        context = context or {}
        token = _DEVELOPING.set(bool(context.get("development")))
        try:
            return self._call(mention, clause, at, language, cid, context)
        finally:
            _DEVELOPING.reset(token)

    def _call(self, mention: str, clause: str, at: int, language: str, cid: str, context: dict) -> Ownership:
        if self.question == "statement" and self.chats:
            return self._by_statement(mention, clause, at, language, cid, context)
        if self.question == "claims" and self.chats:
            return self._by_claims(mention, clause, at, language, cid, context)
        reader, evidence, placed = self._reader(mention, clause, at, language, cid, context)
        if not self.chats:
            if reader == ELSEWHERE:
                return Ownership(UNSURE, evidence, {}, placed, "the_reader_alone_reads_another_owner")
            return Ownership(HERE, evidence)
        linked = self.linked_name(cid) or mention
        model, said, model_placed = self._models(mention, clause, at, language, linked, context)
        if reader == model:
            return Ownership(reader, evidence, said, (model_placed or placed) if reader == ELSEWHERE else "",
                             "reader_and_model_agree")
        why = "reader_and_model_disagree" if model != UNSURE else "the_models_cannot_say_or_disagree"
        return Ownership(UNSURE, evidence, said, model_placed or placed, why)

    @staticmethod
    def linked_name(cid: str) -> str:
        """A readable name of the linked class for the model ("middle lobe of right lung")."""
        names = _class_names().get(cid, []) if cid else []
        return max(names, key=len) if names else cid.replace("_", " ")


_NAMES: dict[str, list[str]] = {}


def _class_names() -> dict[str, list[str]]:
    if not _NAMES:
        try:
            _NAMES.update({cid: c.get("en", []) for cid, c in json.loads(LEXICON.read_text("utf-8"))["classes"].items()})
        except Exception:  # noqa: BLE001
            pass
    return _NAMES


__all__ = ["DISCORDANCES", "DOUBT", "KEEP", "agreed_claims", "claims_outcome", "laterality_conflict", "ASIDE", "ELSEWHERE", "FINDING", "HERE", "HISTORY", "MEASURE", "NORMAL", "NOTHING", "STATEMENT_GROUPS",
           "UNSURE", "Anatomy", "FindingOwner", "Ownership", "report_context", "type_check", "window"]
