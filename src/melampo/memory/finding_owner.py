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

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE, ELSEWHERE, UNSURE = "here", "elsewhere", "unsure"
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
_SECTION = re.compile(r"(?:^|[\n.;])\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ /\-]{1,40}):")


@dataclass(frozen=True)
class Ownership:
    verdict: str                                   # HERE, ELSEWHERE, UNSURE
    reader: str = ""                               # the reader's reading and the evidence against the link
    models: dict[str, dict] = field(default_factory=dict)   # what each model answered
    placed: str = ""                               # what the clause describes instead, for the person
    reason: str = ""
    statement: str = ""                            # what the sentence says of the structure (STATEMENT_GROUPS)
    piece: str = ""                                # the words that say it, copied from the sentence
    claims: dict = field(default_factory=dict)     # the claims the readers agree on (question="claims")
    review: str = ""                               # what a person should look at even when the readers agree

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "reader": self.reader, "models": self.models, "placed": self.placed,
                "reason": self.reason, "statement": self.statement, "piece": self.piece, "claims": self.claims,
                "review": self.review}


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
        # UBERON's labels and exact synonyms, not its relational adjectives ("cortical", "dorsal" name no
        # structure in a clause)
        for uid, names in (uberon_names or {}).items():
            for name in names:
                self._add(name, {uid})
        self.organs = {o for o in getattr(uberon, "organs", set())
                       if not _KIND_OF_ORGAN.search(uberon.labels.get(o, ""))} if uberon is not None else set()
        self._up: dict[str, set[str]] = {}

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

    def find(self, text: str, short: bool = False) -> list[tuple[str, set[str]]]:
        """Every body structure named in ``text``, longest names first, in order (``short``: also the short
        single names, for a name a model gives)."""
        words = [_singular(w) for w in _WORD.findall(text.lower())]
        out, i = [], 0
        while i < len(words):
            for n in range(min(5, len(words) - i), 0, -1):
                ids = self.names.get(tuple(words[i:i + n])) or (
                    self.short.get(tuple(words[i:i + n])) if short and n == 1 else None)
                if ids:
                    out.append((" ".join(words[i:i + n]), ids))
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
    return {"exam": exam, "section": section, "areas": ",".join(sorted(areas.of(report))),
            "section_kind": kind, "conclusion": conclusion, "previous": previous}


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
    if observed:
        return ASIDE
    return KEEP


def agreed_claims(answers: list[dict]) -> dict:
    """The claims every model makes, with the first model's words: its findings (of the structure or reaching
    it), and each other kind that every model fills."""
    from ..evaluation.llm_reading import CLAIM_LISTS

    first = answers[0]
    out: dict = {"findings": [o for o in first.get("observed", []) if o.get("of") == "it" or o.get("reaches") == "yes"]}
    others = [o for o in first.get("observed", []) if o not in out["findings"]]
    if others:
        out["of"] = next((o["of"] for o in others if o.get("of") not in ("it", "?")), "")
    for key in CLAIM_LISTS[1:]:
        if all(a.get(key) for a in answers):
            out[key] = list(first.get(key, []))
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
                               conclusion=context.get("conclusion", ""), previous=context.get("previous", ""))
        said: dict[str, dict] = {}
        for model, chat in self.chats.items():
            try:
                said[model] = read_claims(chat(prompt), clause, mention, context.get("previous", ""),
                                          context.get("conclusion", ""))
            except Exception as error:  # noqa: BLE001 -- a lost answer is "cannot say", never a verdict
                said[model] = {"valid": False, "same": "unsure", "error": type(error).__name__,
                               "error_text": str(error)[:160]}
        return self._with_anatomy(said, mention, cid, language)

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
        from .spatial_relation import _OWN_PARTS
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
                elif (set(re.findall(r"[a-zà-ÿ]+", o["of"].lower())) & _OWN_PARTS
                      and not self.anatomy.find(o["of"], short=True)):
                    # "the wall of the arch": a part, and no other structure named, is a part of this one
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
        claims: normal, a measure, its past, a device, not assessable). The models, one stream, must agree; the
        reader checks the claims with what it knows (``_contradiction``); when they do not agree a person
        decides and the finding stays. The claims both models make are attached."""
        text, where = window(clause, at, len(mention))
        reader, evidence, placed = self._reader(mention, text, where, language, cid, context)
        said = self._claim_answers(mention, text, where, language, cid, context)
        outcomes = [claims_outcome(a) for a in said.values()]
        review = laterality_conflict(context.get("exam", ""), cid, mention)
        if "unsure" in outcomes or len(set(outcomes)) != 1:
            return Ownership(UNSURE, evidence, said, placed, "the_models_cannot_say_or_disagree", review=review)
        outcome = outcomes[0]
        claims = agreed_claims(list(said.values()))
        piece = " … ".join(o["words"] for o in claims.get("findings", []))
        if outcome == ASIDE:
            relations = [o.get("relation", "") for a in said.values() for o in a.get("observed", [])
                         if o.get("of") != "it"]
            near = any(_relation_kind(r, language) in ("proximity", "projection") for r in relations if r)
            if reader != ELSEWHERE and not near:
                return Ownership(UNSURE, evidence, said, placed or claims.get("of", ""), "reader_and_model_disagree",
                                 ASIDE, piece, claims, review)
            why = evidence or "the_relation_words_place_it"
            return Ownership(ELSEWHERE, why, said, claims.get("of", "") or placed, "reader_and_model_agree", ASIDE,
                             piece, claims, review)
        why = self._contradiction(outcome, claims, evidence, text, language, cid, mention, list(said.values()))
        why = why or self._reference_check(list(said.values()), cid, context)
        why = why or self._document_check(outcome, claims, list(said.values()), evidence, cid, context)
        if why:
            return Ownership(UNSURE, f"{evidence},{why}".strip(","), said, placed, "reader_and_model_disagree",
                             outcome, piece, claims, review)
        return Ownership(HERE, evidence, said, "", "reader_and_model_agree", outcome, piece, claims, review)

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
            return bool(words) and self.anatomy is not None and bool(self.anatomy.find(words, short=True)) \
                and not self.same_structure(words, linked)

        if outcome == FINDING and context.get("section_kind") == INDICATION:
            return "the_finding_is_in_the_indication"
        for a in answers:
            if outcome == FINDING and a.get("conclusion") == "other" and other(a.get("conclusion_of", "")):
                return "the_conclusion_gives_it_to_another_structure"
            for o in a.get("observed", []):
                if o.get("of") == "it" and other(o.get("refers_to", "")):
                    return "its_refers_to_another_structure"
            for ref in a.get("measure_refs", []):
                if any(ref["words"] in o.get("words", "") or o.get("words", "") in ref["words"]
                       for o in a.get("observed", []) if o.get("words")):
                    return "the_measure_is_of_what_is_observed"
        if outcome == KEEP and not any(claims.get(k) for k in ("normal", "measure", "past", "device", "procedure",
                                                                 "indication", "not_assessable")) \
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
        from .what_is_said import past, stated_anomalies

        sites = self.anomalies.sites_of(cid) if self.anomalies else set()
        for w in claims.get("normal", []):
            # what the models compared with the usual and found the same, or call normal, while the reader knows
            # the words as a deviation of this tissue (HPO, PATO): the two readings do not correspond (4 November
            # 2026). The reader knows fewer deviations than a radiologist, so it never confirms a finding alone
            if stated_anomalies(w, language, self.anomalies, sites):
                return "the_normal_words_state_an_abnormality"
        if outcome == FINDING:
            findings = claims.get("findings", [])
            own = [o for o in findings if o.get("of") == "it"]
            if own and any(e.startswith(("landmark", "describes:", "outside_the_exam")) for e in evidence.split(",")):
                return "the_reader_reads_another_owner"
            if own and "named_only_in_a_differential" in evidence and not claims.get("alternatives"):
                return "the_reader_reads_a_differential"
            for a in answers or [claims]:
                mine = [o for o in a.get("observed", a.get("findings", [])) if o.get("of") == "it"]
                if mine and all(o.get("relation") and _relation_kind(o["relation"], language) == "proximity"
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
                if stated_anomalies(w, language, self.anomalies, sites):
                    return f"the_{key}_words_state_an_abnormality"
        return ""

    def __call__(self, mention: str, clause: str, at: int, language: str = "en", cid: str = "",
                 context: dict | None = None) -> Ownership:
        context = context or {}
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


__all__ = ["KEEP", "agreed_claims", "claims_outcome", "laterality_conflict", "ASIDE", "ELSEWHERE", "FINDING", "HERE", "HISTORY", "MEASURE", "NORMAL", "NOTHING", "STATEMENT_GROUPS",
           "UNSURE", "Anatomy", "FindingOwner", "Ownership", "report_context", "type_check", "window"]
