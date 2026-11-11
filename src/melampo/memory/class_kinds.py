"""What a class of the atlas is: a structure of the body, or a finding in one (11 November 2026).

The atlas the linker links to (TotalSegmentator's classes, ``data/linking/anatomy_lexicon.json``) holds the
structures of the body and, among them, things a structure can have: TotalSegmentator segments the cyst of the
kidney as it segments the kidney. A cyst is matter of the body, but it is not a part of the body's plan: it is a
finding of the kidney (HPO, Renal cyst, HP:0000107). When the linker links "cyst in the left kidney" to it, the
claims reading asks what the sentence states of "the cyst of the left kidney": the finding of a finding; the
structure the finding belongs to (the kidney, its parts, its reference values) and the phenotype are lost.

There is no list of such classes. Each class is read from the sources, the same way for every class:

1. a class anchored to UBERON (``anatomy_graph_anchors.json``: its terms, or the term of its family) is a
   structure of the body: UBERON is the ontology of the body's plan;
2. a class with no UBERON anchor whose name, without its side, is the name or an EXACT synonym of a phenotypic
   abnormality of HPO is a finding: its structure is the one HPO says the abnormality is of (the UBERON term of
   its logical definition, or of its nearest ancestor's), read back to the class of the atlas anchored to that
   term, on the side the name says; the class stays the finding's, for the image (what to isolate);
3. any other class is left as it is, a structure (the segments of the liver have no UBERON term and are no
   abnormality: they are parts of the liver).

Without HPO nothing is reinterpreted, and the reason is said: a class is never turned into a finding by guess.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEXICON = ROOT / "data/linking/anatomy_lexicon.json"
ANCHORS = ROOT / "data/linking/anatomy_graph_anchors.json"
PHENOTYPE_ROOT = "HP:0000118"
_SIDES = {"left": "left", "right": "right", "sinistro": "left", "sinistra": "left", "destro": "right",
          "destra": "right", "sx": "left", "dx": "right"}

_CACHE: dict[str, dict] = {}


def _norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def _side_and_rest(name: str) -> tuple[str, str]:
    words = _norm(name).split()
    sides = {_SIDES[w] for w in words if w in _SIDES}
    rest = " ".join(w for w in words if w not in _SIDES)
    return (sides.pop() if len(sides) == 1 else ""), rest


def hpo_path() -> Path | None:
    """HPO for the reading of the classes: ``MELAMPO_HPO_OBO``, else hp-full.obo or data/hp.obo if present."""
    for p in (os.environ.get("MELAMPO_HPO_OBO", ""), "hp-full.obo", str(ROOT / "data/hp.obo"),
              str(ROOT / "hp-full.obo")):
        if p and Path(p).exists():
            return Path(p)
    return None


def _phenotypes(path: Path) -> tuple[dict[str, str], dict[str, list[str]], dict[str, set[str]], dict[str, str]]:
    """(name or EXACT synonym -> HP id, HP id -> parents, HP id -> UBERON of its logical definition, names)."""
    from .hpo_candidates import read_obo

    with open(path, encoding="utf-8") as fh:
        terms = read_obo(fh, "HP:")
    parents = {h: [p.split(" ")[0] for p in t.get("is_a", [])] for h, t in terms.items()}
    seen: dict[str, bool] = {}

    def phenotypic(h: str, depth: int = 0) -> bool:
        if h == PHENOTYPE_ROOT:
            return True
        if h not in seen:
            seen[h] = depth < 40 and any(phenotypic(p, depth + 1) for p in parents.get(h, []))
        return seen[h]

    by_name: dict[str, str] = {}
    sites: dict[str, set[str]] = {}
    names: dict[str, str] = {}
    for h, t in terms.items():
        if not phenotypic(h):
            continue
        names[h] = (t.get("name") or [""])[0]
        for n in [names[h]] + [s.split('"')[1] for s in t.get("synonym", []) if " EXACT" in s and '"' in s]:
            by_name.setdefault(_norm(n), h)
        sites[h] = {w for x in t.get("intersection_of", []) for w in x.split() if w.startswith("UBERON:")}
    return by_name, parents, sites, names


def _site_of(h: str, parents: dict[str, list[str]], sites: dict[str, set[str]]) -> set[str]:
    """The UBERON terms of the nearest logical definition, from the term up."""
    level, seen = [h], set()
    for _ in range(40):
        found = {u for x in level for u in sites.get(x, set())}
        if found:
            return found
        seen.update(level)
        level = [p for x in level for p in parents.get(x, []) if p not in seen]
        if not level:
            break
    return set()


def class_kinds(hpo: Path | None = None) -> dict[str, dict]:
    """class id -> {"kind": "structure"|"finding", ...}; a finding with its "site" (a class of the atlas), its
    "phenotype" (HP id and name) and "why"."""
    path = hpo or hpo_path()
    key = str(path)
    if key in _CACHE:
        return _CACHE[key]
    classes = json.loads(LEXICON.read_text("utf-8"))["classes"]
    anchors = json.loads(ANCHORS.read_text("utf-8"))["anchors"]
    anchored = {cid for cid, a in anchors.items() if a.get("uberon") or a.get("family")}
    out: dict[str, dict] = {cid: {"kind": "structure", "why": "anchored to UBERON"} for cid in anchored}
    loose = [cid for cid in classes if cid not in anchored]
    if path is None:
        for cid in loose:
            out[cid] = {"kind": "structure", "why": "no UBERON anchor and no HPO at hand: not read"}
        _CACHE[key] = out
        return out
    by_name, parents, sites, names = _phenotypes(path)
    for cid in loose:
        out[cid] = {"kind": "structure", "why": "no UBERON anchor, and not a phenotype of HPO"}
        for name in classes[cid].get("en", []):
            side, rest = _side_and_rest(name)
            h = by_name.get(rest)
            if not h:
                continue
            uberon = _site_of(h, parents, sites)
            candidates = [c for c in anchored
                          if uberon & (set(anchors[c].get("uberon", {})) | set(anchors[c].get("family", {})))]
            # the side the name says chooses among the classes of the term's family
            sided = [c for c in candidates if not side or _side_and_rest(" ".join(classes.get(c, {}).get(
                "en", [c])))[0] in (side, "")]
            exact = [c for c in sided if set(anchors[c].get("uberon", {})) & uberon] or sided
            if len(exact) == 1:
                out[cid] = {"kind": "finding", "site": exact[0], "phenotype": h, "phenotype_name": names[h],
                            "why": f"its name is {names[h]} ({h}), of {sorted(uberon)} in HPO"}
            else:
                out[cid] = {"kind": "structure", "why": f"{names[h]} ({h}), but its structure is not one class "
                                                       f"of the atlas ({sorted(exact)})"}
            break
    _CACHE[key] = out
    return out


def as_site(cid: str, hpo: Path | None = None) -> dict:
    """What the linker's class means for the site: {"cid": the structure, "finding_class": the class if it is a
    finding, "phenotype": ...}."""
    kind = class_kinds(hpo).get(cid, {"kind": "structure"})
    if kind.get("kind") != "finding":
        return {"cid": cid}
    return {"cid": kind["site"], "finding_class": cid, "phenotype": kind["phenotype"],
            "phenotype_name": kind["phenotype_name"]}


__all__ = ["as_site", "class_kinds", "hpo_path"]
