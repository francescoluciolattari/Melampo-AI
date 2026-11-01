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

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "reader": self.reader, "models": self.models, "placed": self.placed,
                "reason": self.reason}


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
        if not key or (len(key) == 1 and (key[0] in _GENERIC or len(key[0]) < 5)):
            return
        self.names.setdefault(key, set()).update(ids)

    def find(self, text: str) -> list[tuple[str, set[str]]]:
        """Every body structure named in ``text``, longest names first, in order."""
        words = [_singular(w) for w in _WORD.findall(text.lower())]
        out, i = [], 0
        while i < len(words):
            for n in range(min(5, len(words) - i), 0, -1):
                ids = self.names.get(tuple(words[i:i + n]))
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
    modality) and the section the sentence is in ("TZ:", "Bladder:")."""
    from .exam_area import ExamAreas

    areas = ExamAreas.load()
    exam = ""
    if areas.modality is not None:
        for sentence in re.split(r"(?<=[.\n])\s+", report[:2000]):
            if areas.modality.search(sentence):
                exam = sentence.strip()[:160]
                break
    sections = [m for m in _SECTION.finditer(report[:clause_start + 1])]
    section = sections[-1].group(1).strip() if sections and clause_start - sections[-1].end() < 400 else ""
    return {"exam": exam, "section": section, "areas": ",".join(sorted(areas.of(report)))}


class FindingOwner:
    """``reader``: a ``CIReader``; ``chats``: model name -> chat function (local); ``anatomy``: ``Anatomy``."""

    def __init__(self, reader, chats: dict[str, Callable[[str], str]] | None = None, anatomy: Anatomy | None = None):
        self.reader = reader
        self.chats = dict(chats or {})
        self.anatomy = anatomy if anatomy is not None else Anatomy()

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
            before = re.split(r"[,;/]", clause[:at])[-1]
            after = clause[at + len(mention): at + len(mention) + 30]
            named = self.anatomy.find(before)
            head = self.anatomy.find(" ".join(_WORD.findall(after)[:2]))
            for name, ids in ([head[0]] if head and after[:1] == " " else []) + named[:1]:
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
                               "error": type(error).__name__}
        verdicts = {{"yes": HERE, "no": ELSEWHERE}.get(a.get("answer"), UNSURE) for a in said.values()}
        verdict = verdicts.pop() if len(verdicts) == 1 else UNSURE      # two models are one stream
        placed = next((str(a.get("finding") or "") for a in said.values() if a.get("answer") == "no"), "")
        return verdict, said, placed

    def __call__(self, mention: str, clause: str, at: int, language: str = "en", cid: str = "",
                 context: dict | None = None) -> Ownership:
        context = context or {}
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


__all__ = ["ELSEWHERE", "HERE", "UNSURE", "Anatomy", "FindingOwner", "Ownership", "report_context"]
