"""Procedure frames: what a procedure word needs around it (Frank's point 2, 10 October 2026).

"placement" needs something placed (a device) and a place (a structure): reading "inferior vena cava
filter placement", the head at the end draws the words before it into its slots, and the structure is
*where* the procedure is done whatever the brackets ([[IVC filter] placement] or [IVC] [filter placement]).
That is valency (the slots of a head) with selectional restrictions (what may fill each slot).

A frame here is, for a head word that names a procedure: the share of evidence that it takes a *site*
(a structure) and a *device*. The evidence comes from three sources, each written in the file with its
counts, none hand-written:

- **NCIt relations** (CC BY 4.0): procedures with ``Procedure_Has_Target_Anatomy``, ``..._Imaged_Anatomy``,
  ``..._Excised_Anatomy`` (site) and ``Procedure_Uses_Manufactured_Object`` (device), counted on the last
  word of their names. Thin: 331 procedures carry one, none a device.
- **Text**: which kinds of words come before the head in the same noun phrase in what was read ("inferior
  vena cava filter placement": a structure and a device; "liver biopsy": a structure). That is how a reader
  learns valency, from exposure; it grows with the text.
- **SNOMED CT** (``Procedure site``, ``... - Direct``, ``... - Indirect``, ``Using device``, ``Direct
  device``) when a release is available to whoever builds the file. Italy is not a member of SNOMED
  International (members page, October 2026): use needs an affiliate licence through MLDS. The builder
  reads RF2 files; nothing of SNOMED is in the repository.

How the chunk lattice uses it (``ChunkLattice(frames=...)``): when a block that holds the mention ends, to
its right, with a procedure head whose frame has a site slot, and every word between the mention and the
head fills a slot (a device for the device slot, a structure for the site slot), the structure is the
site of that procedure (role ``procedure_site``), whichever smaller block the costs prefer; the reading
writes the frame (``frame:placement:device=filter``).
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SITE, DEVICE = "site", "device"
MIN_EVIDENCE = 3  # fewer observations of a head say nothing about its slots
# NCIt relation ids (ncit.obo Typedefs) and the slot they show
NCIT_RELATIONS = {
    "R163": SITE, "R165": SITE, "R166": SITE, "R167": SITE, "R168": SITE, "R169": SITE, "R170": SITE,
    "R171": SITE, "R181": DEVICE,
}
# SNOMED CT attribute concept ids and the slot they show
SNOMED_ATTRIBUTES = {
    "363704007": SITE,  # Procedure site
    "405813007": SITE,  # Procedure site - Direct
    "405814001": SITE,  # Procedure site - Indirect
    "424226004": DEVICE,  # Using device
    "363699004": DEVICE,  # Direct device
}
_WORD = re.compile(r"[a-z]+")
# the prepositions through which a procedure takes what follows it ("placement of a stent in the duct")
_PREPOSITIONS = frozenset("of in into at to within through via across on onto from".split())


def _head(name: str) -> str:
    words = _WORD.findall(name.lower())
    return _singular(words[-1]) if words else ""


def _singular(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = k / n
    centre = p + z * z / (2 * n)
    spread = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return (centre - spread) / (1 + z * z / n)


class ProcedureFrames:
    def __init__(self) -> None:
        # head -> source -> Counter({"n": observations, "site": ..., "device": ...})
        self.evidence: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))

    # -- evidence ------------------------------------------------------------------------------

    def observe(self, head: str, source: str, slots: set[str], weight: int = 1) -> None:
        self._base = None
        c = self.evidence[head][source]
        c["n"] += weight
        for s in slots:
            c[s] += weight

    def add_ncit(self, path: Path) -> ProcedureFrames:
        """Every NCIt class with a site or device relation, counted on the last word of its name."""
        name, slots = None, set()
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line in ("[Term]", "[Typedef]"):
                    if name and slots:
                        self.observe(_head(name), "ncit", slots)
                    name, slots = None, set()
                elif line.startswith("name: "):
                    name = line[6:]
                elif line.startswith("relationship: NCIT:R"):
                    rel = line.split()[1].split(":")[1]
                    if rel in NCIT_RELATIONS:
                        slots.add(NCIT_RELATIONS[rel])
        if name and slots:
            self.observe(_head(name), "ncit", slots)
        return self

    def add_texts(self, texts, kind_of, window: int = 5, right: int = 7) -> ProcedureFrames:
        """From text read: for every word that names a procedure (``kind_of(word) == "procedure"``), the
        kinds of the words that fill its slots in the same run (no punctuation crossed, at most ``window``
        words; 7 after a preposition, the lattice's right window): before it ("inferior vena cava filter
        placement": a structure and a device) and after it through a preposition ("placement of a stent in the bile duct"). A structure there shows a site
        slot, a device a device slot. Occurrences with no typed word around the head say nothing and are
        not counted. ``kind_of(word)`` returns the kind of a word (the block memory's heads) or ``""``."""
        from .collocations import runs

        for text in texts:
            for run in runs(text):
                kinds = [kind_of(w) for w in run]
                for i, k in enumerate(kinds):
                    if k != "procedure":
                        continue
                    around = [x for x in kinds[max(0, i - window) : i] if x]
                    if i + 1 < len(run) and run[i + 1] in _PREPOSITIONS:
                        around += [x for x in kinds[i + 2 : i + 2 + right] if x]
                    if not around:
                        continue
                    slots = set()
                    if "anatomy" in around:
                        slots.add(SITE)
                    if "device" in around:
                        slots.add(DEVICE)
                    self.observe(_singular(run[i]), "text", slots)
        return self

    def add_snomed(self, relationships: Path, descriptions: Path) -> ProcedureFrames:
        """RF2 snapshot files: ``sct2_Relationship_Snapshot*.txt`` and ``sct2_Description_Snapshot*.txt``
        (English). Every active concept with a site or device attribute, counted on the last word of each
        of its active English terms."""
        slots: dict[str, set[str]] = defaultdict(set)
        with open(relationships, encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            i_active, i_src, i_type = header.index("active"), header.index("sourceId"), header.index("typeId")
            for line in fh:
                row = line.rstrip("\n").split("\t")
                if row[i_active] == "1" and row[i_type] in SNOMED_ATTRIBUTES:
                    slots[row[i_src]].add(SNOMED_ATTRIBUTES[row[i_type]])
        with open(descriptions, encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            i_active, i_concept, i_term, i_lang = (header.index("active"), header.index("conceptId"),
                                                   header.index("term"), header.index("languageCode"))
            for line in fh:
                row = line.rstrip("\n").split("\t")
                if row[i_active] != "1" or row[i_lang] != "en" or row[i_concept] not in slots:
                    continue
                term = re.sub(r"\s*\([^)]*\)$", "", row[i_term])  # the semantic tag of a fully specified name
                self.observe(_head(term), "snomed", slots[row[i_concept]])
        return self

    # -- the frame -----------------------------------------------------------------------------

    def frame(self, head: str) -> dict[str, float] | None:
        """``{"site": share, "device": share}`` over all sources, or ``None`` with too little evidence."""
        sources = self.evidence.get(head)
        if not sources:
            return None
        total = Counter()
        for c in sources.values():
            total.update(c)
        if total["n"] < MIN_EVIDENCE:
            return None
        return {s: total[s] / total["n"] for s in (SITE, DEVICE)}

    def base_rate(self) -> dict[str, float]:
        """How often a slot is shown by any procedure head (all heads, all sources together)."""
        if getattr(self, "_base", None) is None:
            total = Counter()
            for by in self.evidence.values():
                for c in by.values():
                    total.update(c)
            n = total["n"] or 1
            self._base = {s: total[s] / n for s in (SITE, DEVICE)}
        return self._base

    def slots(self, head: str) -> set[str]:
        """The slots this head shows more often than procedure heads in general, with 95% confidence (the
        Wilson lower bound of its share is above the base rate): a slot is what distinguishes the head."""
        sources = self.evidence.get(head)
        if not sources:
            return set()
        total = Counter()
        for c in sources.values():
            total.update(c)
        n = total["n"]
        if n < MIN_EVIDENCE:
            return set()
        base = self.base_rate()
        out = set()
        for s in (SITE, DEVICE):
            if _wilson_lower(total[s], n) > base[s]:
                out.add(s)
        return out

    # -- files ---------------------------------------------------------------------------------

    def to_json(self) -> dict:
        return {h: {src: dict(c) for src, c in by.items()} for h, by in sorted(self.evidence.items())}

    @classmethod
    def from_json(cls, data: dict) -> ProcedureFrames:
        f = cls()
        for h, by in data.items():
            for src, c in by.items():
                f.evidence[h][src] = Counter(c)
        return f

    @classmethod
    def load(cls, path: Path) -> ProcedureFrames:
        return cls.from_json(json.loads(Path(path).read_text("utf-8"))["frames"])


DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "linking" / "procedure_frames.json"

__all__ = ["DEFAULT_PATH", "DEVICE", "SITE", "ProcedureFrames"]
