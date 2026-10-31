"""The HPO terms a finding at a site can be, and two ways of choosing among them (31 October 2026).

A report's finding is read as a site (the linker: a TotalSegmentator class, anchored to UBERON in
``data/linking/anatomy_graph_anchors.json``) and a clause ("Nel lobo superiore destro nodulo solido di 12 mm a margini
spiculati"). The diagnostic graph wants an HPO term (*Solid pulmonary nodule*). This module gives:

* ``Phenotypes.candidates(site)`` -- the HPO terms **localised at that site**, by three routes, because none is
  enough alone (HPO 2026-09-01: only 4,170 of 19,894 terms have a logical definition, and none of the 34 "nodule"
  terms has one):

  The site's classes are its UBERON anchors (lung_upper_lobe_right -> upper lobe of right lung, upper lobe of lung)
  and, for each, the nearest ancestor (``part_of``/``is_a``) that UBERON marks as an organ (subsets ``organ_slim``,
  ``major_organ``): the lung. No higher: two steps up from the liver are "abdomen" and "digestive system", and every
  abdominal cyst would become a candidate for the liver (measured, 31 October 2026).

  1. **logical definition**: the term's ``intersection_of`` names one of the site's classes: *Pleural effusion* =
     edematous, inheres in pleural cavity;
  2. **name**: the term's label or an exact synonym contains, as whole words, a name of one of the site's classes
     -- its label, its exact synonyms, its relational adjective (UBERON ``has_relational_adjective``: lung ->
     "pulmonary"): *Pulmonary nodule*;
  3. **descendants** of the terms found by 1 and 2: *Solid pulmonary nodule* under *Pulmonary nodule*.

* ``Phenotypes.shortlist(site, words, k)`` -- the ``k`` candidates nearest to the clause's words (shared stems,
  label and synonyms), plus the most general term found by route 1 or 2 (the safe answer when the clause names no
  specific abnormality); this is what a model chooses from, with "none of these".

* ``Phenotypes.deterministic(site, words)`` -- method A: the most specific candidate whose label or exact synonym
  is entirely made of the clause's words and the site's names (stems), with at least one word that is not a site
  name; two such terms neither of which is below the other is an abstention.

The ontologies are read from their OBO releases: ``hp-full.obo`` (with the logical definitions; ``hp.obo`` has
none) and ``uberon.obo`` (the full one: ``uberon-basic.obo`` has no relational adjectives). Nothing is written by
hand: no per-finding table.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

ANCHORS = Path(__file__).resolve().parents[3] / "data" / "linking" / "anatomy_graph_anchors.json"
ORGAN_SUBSETS = frozenset({"organ_slim", "major_organ"})
ORGAN_SEARCH_DEPTH = 3       # chosen on the development labels (31 October 2026): see the probe doc
WHOLE_DEPTH = 0              # 1 or 2 add "intestinal" for the colon but triple the candidates and lose ranks
PHENOTYPE_ROOT = "HP:0000118"          # Phenotypic abnormality
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("of the a an in on at to and or with without by for from as is are be this that".split())


def stem(word: str) -> str:
    """A light English stem: plurals, the commonest suffixes and a final "e", enough to match "nodules" with
    "nodule", "infiltrate" with "infiltrates", "fractured" with "fracture" (31 October 2026: without the final "e"
    and "-ed", *Pulmonary infiltrates* never met "infiltrate")."""
    w = word.lower()
    for suffix in ("ations", "ation", "ments", "ment", "ities", "ity", "ings", "ing", "otic", "osis", "ies", "ous",
                   "ed", "es", "s", "al", "ic"):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            w = w[: -len(suffix)] + ("y" if suffix == "ies" else "")
            break
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w


def stems(text: str) -> tuple[str, ...]:
    return tuple(stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP)


@dataclass
class Term:
    id: str
    name: str
    synonyms: list[str] = field(default_factory=list)       # exact synonyms only (no layperson, no related)
    parents: list[str] = field(default_factory=list)
    anatomy: set[str] = field(default_factory=set)           # UBERON ids named by the logical definition
    other_names: list[str] = field(default_factory=list)     # layperson and related synonyms: for ranking only


def read_obo(lines: Iterable[str], prefix: str) -> dict[str, dict]:
    """The [Term] stanzas of an OBO file whose id starts with ``prefix``, as plain dicts of lists."""
    out: dict[str, dict] = {}
    cur: dict | None = None
    for raw in lines:
        line = raw.rstrip("\n")
        if line == "[Term]":
            cur = defaultdict(list)
            continue
        if line.startswith("["):
            cur = None
            continue
        if cur is None or ": " not in line:
            continue
        key, value = line.split(": ", 1)
        if key == "id":
            if value.startswith(prefix):
                out[value] = cur
            cur["id"] = [value]
        else:
            cur[key].append(value)
    return {k: v for k, v in out.items() if not v.get("is_obsolete")}


def _quoted(value: str) -> str:
    return value.split('"')[1] if value.count('"') >= 2 else ""


@dataclass
class Anatomy:
    """UBERON: names, relational adjectives and the anatomical ancestors of a class."""

    names: dict[str, list[str]]
    parents: dict[str, list[str]]
    labels: dict[str, str]
    organs: set[str] = field(default_factory=set)
    part_of: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_obo(cls, lines: Iterable[str]) -> Anatomy:
        names, parents, labels, organs, part_of = {}, {}, {}, set(), {}
        for uid, t in read_obo(lines, "UBERON:").items():
            label = (t.get("name") or [""])[0]
            labels[uid] = label
            if ORGAN_SUBSETS & set(t.get("subset", [])):
                organs.add(uid)
            ns = [label] + [_quoted(s) for s in t.get("synonym", []) if " EXACT" in s]
            ns += [_quoted(p) for p in t.get("property_value", []) if p.startswith("has_relational_adjective")]
            names[uid] = [n for n in dict.fromkeys(ns) if n and len(n) > 2]
            ps = [p.split(" ")[0] for p in t.get("is_a", []) if p.startswith("UBERON:")]
            po = [r.split(" ")[1] for r in t.get("relationship", []) if r.startswith("part_of UBERON:")]
            parents[uid] = ps + po
            part_of[uid] = po
        return cls(names, parents, labels, organs, part_of)

    def site_classes(self, anchors: list[str]) -> list[str]:
        """The anchors; for an anchor that is not an organ, every class met on the way up (``part_of`` or ``is_a``,
        breadth first) until the level where an organ is reached, that level included: lumbar vertebra 4 ->
        lumbar vertebra, vertebra, bone of dorsum ... bone element. For an organ, its ``part_of`` parents up to
        ``WHOLE_DEPTH`` steps (colon -> large intestine -> intestine: "intestinal"). An anchor whose ancestors never
        reach an organ in a few steps adds nothing."""
        out = list(dict.fromkeys(anchors))
        for a in anchors:
            if a in self.organs:
                frontier = [a]
                for _ in range(WHOLE_DEPTH):
                    frontier = [p for u in frontier for p in self.part_of.get(u, []) if p not in out]
                    out.extend(frontier)
                continue
            frontier, seen, met = [a], {a}, []
            for _ in range(ORGAN_SEARCH_DEPTH):
                nxt = [p for u in frontier for p in self.parents.get(u, []) if p not in seen]
                seen.update(nxt)
                met.extend(nxt)
                if any(p in self.organs for p in nxt):
                    out.extend(m for m in met if m not in out)
                    break
                frontier = nxt
        return out


def site_anchors(cid: str, path: Path = ANCHORS) -> list[str]:
    """The UBERON classes of a TotalSegmentator class (its own, then its family): lung_upper_lobe_right -> the upper
    lobe of the right lung, the right lung, the lung."""
    data = json.loads(path.read_text("utf-8")).get("anchors", {})
    entry = data.get(cid) or {}
    parts = cid.split("_")
    while not entry and len(parts) > 1:      # kidney_cyst_right -> kidney_cyst -> kidney (a class with no anchor)
        parts = parts[:-1]
        entry = data.get("_".join(parts)) or data.get("_".join(parts) + "_right") or {}
    return list(dict.fromkeys(list((entry.get("uberon") or {}).keys()) + list((entry.get("family") or {}).keys())))


@dataclass
class Phenotypes:
    terms: dict[str, Term]
    anatomy: Anatomy
    children: dict[str, list[str]] = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_obo(cls, hpo_lines: Iterable[str], anatomy: Anatomy) -> Phenotypes:
        terms: dict[str, Term] = {}
        for hid, t in read_obo(hpo_lines, "HP:").items():
            inter = t.get("intersection_of", [])
            terms[hid] = Term(
                hid, (t.get("name") or [""])[0],
                [_quoted(s) for s in t.get("synonym", []) if " EXACT" in s and "layperson" not in s],
                [p.split(" ")[0] for p in t.get("is_a", []) if p.startswith("HP:")],
                {w for x in inter for w in x.split() if w.startswith("UBERON:")},
                [_quoted(s) for s in t.get("synonym", []) if " EXACT" not in s or "layperson" in s],
            )
        children: dict[str, list[str]] = defaultdict(list)
        for t in terms.values():
            for p in t.parents:
                children[p].append(t.id)
        return cls(terms, anatomy, dict(children))

    # -- the candidates ----------------------------------------------------------------------------------------
    def phenotypic(self, hid: str) -> bool:
        return hid == PHENOTYPE_ROOT or PHENOTYPE_ROOT in self.ancestors(hid)

    def ancestors(self, hid: str) -> set[str]:
        key = ("anc", hid)
        if key not in self._cache:
            self._cache[key] = self._ancestors(hid)
        return self._cache[key]

    def _ancestors(self, hid: str) -> set[str]:
        out, stack = set(), [hid]
        while stack:
            for p in self.terms.get(stack.pop(), Term("", "")).parents:
                if p not in out:
                    out.add(p)
                    stack.append(p)
        return out

    def descendants(self, hid: str) -> set[str]:
        key = ("desc", hid)
        if key not in self._cache:
            self._cache[key] = self._descendants(hid)
        return self._cache[key]

    def _descendants(self, hid: str) -> set[str]:
        out, stack = set(), [hid]
        while stack:
            for c in self.children.get(stack.pop(), []):
                if c not in out:
                    out.add(c)
                    stack.append(c)
        return out

    def site_words(self, uberon: list[str]) -> list[tuple[str, ...]]:
        """The names of the site's anchors, as stem tuples ("pulmonary" -> ("pulmonary",))."""
        out = []
        for u in self.anatomy.site_classes(uberon):
            for n in self.anatomy.names.get(u, []):
                s = stems(n)
                if s and s not in out:
                    out.append(s)
        return out

    def candidates(self, uberon: list[str]) -> dict[str, set[str]]:
        """HPO id -> the routes that found it ("definition", "name", "descendant")."""
        key = ("cand", tuple(uberon))
        if key not in self._cache:
            self._cache[key] = self._candidates(uberon)
        return self._cache[key]

    def _candidates(self, uberon: list[str]) -> dict[str, set[str]]:
        lineage = set(self.anatomy.site_classes(uberon))
        words = self.site_words(uberon)
        found: dict[str, set[str]] = defaultdict(set)
        for t in self.terms.values():
            if t.anatomy & lineage:
                found[t.id].add("definition")
            for text in [t.name, *t.synonyms, *t.other_names]:
                s = stems(text)
                if any(_contains(s, w) for w in words):
                    found[t.id].add("name")
                    break
        for hid in list(found):
            for d in self.descendants(hid):
                found[d].add("descendant")
        return {h: r for h, r in found.items() if self.phenotypic(h)}

    def general(self, cands: dict[str, set[str]]) -> str | None:
        """The candidate found by name or definition that most other candidates descend from (the safe answer)."""
        direct = [h for h, r in cands.items() if r & {"definition", "name"}]
        if not direct:
            return None
        return max(direct, key=lambda h: (len(self.descendants(h) & set(cands)), -len(self.ancestors(h)), h))

    def shortlist(self, uberon: list[str], text: str, k: int = 15) -> list[str]:
        """The ``k`` candidates nearest to the words of ``text``, the general one always among them."""
        cands = self.candidates(uberon)
        words = set(stems(text))
        site = {w for s in self.site_words(uberon) for w in s}

        def score(h: str) -> tuple:
            t = self.terms[h]
            best = (0, 0.0)
            for name in [t.name, *t.synonyms, *t.other_names]:
                s = {w for w in stems(name) if w not in site}
                if s:
                    shared = len(s & words)
                    best = max(best, (shared, shared / len(s)))
            # more words of the sentence first ("ground-glass nodule": the ground-glass nodule before the nodule),
            # then the share of the term's own words that the sentence holds
            return (-best[0], -best[1], len(self.ancestors(h)), t.name)

        ranked = [h for h in sorted(cands, key=score) if score(h)[0] < 0][:k]
        for g in self.organ_terms(uberon, cands):
            if g not in ranked:
                ranked.append(g)
        g = self.general(cands)
        if g and g not in ranked:
            ranked.append(g)
        return ranked

    def organ_terms(self, uberon: list[str], cands: dict[str, set[str]], limit: int = 3) -> list[str]:
        """The site's own general terms ("Abnormal adrenal morphology", "Abnormality of the liver"): found by name or
        definition, with no word besides the site's names and the words of a general term."""
        site = {w for s in self.site_words(uberon) for w in s}
        generic = {stem(w) for w in ("abnormal", "abnormality", "morphology", "anomaly", "abnormalities")}
        out = []
        for h, routes in cands.items():
            if not routes & {"definition", "name"}:
                continue
            for name in [self.terms[h].name, *self.terms[h].synonyms]:
                s = set(stems(name))
                if s and s <= site | generic and s & generic:
                    out.append(h)
                    break
        return sorted(out, key=lambda h: (-len(self.descendants(h) & set(cands)), h))[:limit]

    def deterministic(self, uberon: list[str], text: str) -> tuple[str | None, str]:
        """Method A: (HPO id or None, reason)."""
        cands = self.candidates(uberon)
        words = set(stems(text))
        site = [s for s in self.site_words(uberon)]
        site_stems = {w for s in site for w in s}
        hits = []
        for h in cands:
            t = self.terms[h]
            for name in [t.name, *t.synonyms]:
                s = stems(name)
                if not s or not (set(s) - site_stems):
                    continue
                if all(w in words or w in site_stems for w in s):
                    hits.append(h)
                    break
        if not hits:
            return None, "no_candidate_named_by_the_clause"
        top = [h for h in hits if not (self.descendants(h) & set(hits))]
        if len(top) > 1:
            return None, "several_terms_fit:" + ",".join(sorted(top)[:4])
        return top[0], "named"


def _contains(seq: tuple[str, ...], sub: tuple[str, ...]) -> bool:
    n = len(sub)
    return any(seq[i:i + n] == sub for i in range(len(seq) - n + 1))


def load(hpo_path: Path, uberon_path: Path) -> Phenotypes:
    with open(uberon_path, encoding="utf-8") as fh:
        anatomy = Anatomy.from_obo(fh)
    with open(hpo_path, encoding="utf-8") as fh:
        return Phenotypes.from_obo(fh, anatomy)


__all__ = ["Anatomy", "Phenotypes", "Term", "load", "site_anchors", "stem", "stems"]
