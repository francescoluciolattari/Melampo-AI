"""The anatomical graph the linker's streams converge on: UBERON is-a and part-of, anchored to the
TotalSegmentator classes.

Two jobs, both from the nine steps (`claude/nove_stadi_dettaglio_2026-10-06.md`) and the parallel
design (`claude/architettura_parallela_predittiva_linker_2026-10-06.md`):

* **Fallback to the parent (step 9).** When the models link a mention to an UBERON term finer than
  any segmentation class ("sigmoid colon", "head of pancreas"), the graph climbs is-a and part-of to
  the nearest class and says how (``part_of``). MedPath (2025) found that about 40% of biomedical
  linking errors are of granularity. **By default this is a proposal, not a link**
  (``AnatomyLinker.accept_parent_fallback=False``): the result keeps the UBERON term and carries the
  proposed class in ``fallback``, for the review queue and for measurement on the bench.
* **Neighbours (step 5).** Two structures close in the graph -- parent and child, sisters, the two
  sides of a pair, structures UBERON declares spatially disjoint -- are what a reader confuses
  without noticing (the Moses illusion). When the models' choice has such a neighbour among the
  options that passed the same deterministic checks, nothing but the models' agreement separates
  them, and the agreement of two LLMs is not independent evidence. The linker abstains.

**Why the fallback only proposes.** UBERON part-of is anatomy, not containment in a CT mask. Three
blind reviews of 100 random lifts each (7 October 2026) found 3 wrong and 9 doubtful, then 1 and 13,
then 5 and 12: the pituitary "part of" the brain, the lesser omentum "part of" the stomach, the
urachus "part of" the bladder, a duodenal crypt "part of" the small bowel (a separate class in the
segmentation). Each review added rules (``REFUSED_KINDS``, the curated ``refused`` list, the
two-classes rule), and each new sample still found errors. A rate of a few percent is far from the
1% target, so the fallback decides nothing until the gold set certifies it.

**Safety rules.**

* Anchors (class <-> UBERON term) are a curated file, ``data/linking/anatomy_graph_anchors.json``,
  not a run-time name match: automatic matching on synonyms proposed "insect arista" for liver
  segment 6 and a zebrafish "inferior lobe" for the lower lobes.
* A *family* anchor ("kidney" for kidney_left and kidney_right) is used only when the mention states
  the side. Without it the fallback refuses (the part-table rule: a part of a paired organ needs a
  side). An anchor on the other side of the stated one is refused.
* No climb through a space, a surface, a point or groove, a bone foramen, a vessel, a lymph node, a
  nerve, a ligament, a mesentery, a ventricle, a bronchus, a fontanelle or an embryonic structure;
  no "kind of" (xiphoid cartilage is-a costal cartilage in UBERON, not a costal cartilage of the
  segmentation); no term that reaches two classes at any depth; never beyond ``max_depth``.
* Laterality inside UBERON lives in is-a ("left kidney" is-a "kidney"), so the sides of a pair are
  known only through the anchors.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_ANCHORS = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "linking"
    / "anatomy_graph_anchors.json"
)
_EDGE_KINDS = ("is_a", "part_of")
_DEVELOPING_ROOTS = frozenset(("UBERON:0005423", "UBERON:0002050"))
_SYNONYM = re.compile(r'synonym: "(.*?)" (?:EXACT|RELATED)\b')
# A parent with more children than this is a category ("organ", "bone element"), not a place:
# two structures that only share it are not neighbours.
_MAX_SIBLING_FANOUT = 40
_RIGHT, _LEFT = "<dx>", "<sn>"
# Kinds of structure that are not inside a segmentation mask even when UBERON makes them part of
# the organ: a sinus or a sulcus is a space, the falciform ligament and the mesentery lie outside
# the organ, embryonic structures do not appear in an adult report. Found by auditing every lift.
REFUSED_KINDS = {
    "UBERON:0000464": "anatomical_space",
    "UBERON:0000211": "ligament",
    "UBERON:0002095": "mesentery",
    "UBERON:0002050": "embryonic_structure",
    # A vessel is not part of the organ it supplies or drains in a segmentation mask, and the
    # tributaries of a vein are not that vein ("jejunal vein" is not the portal vein): no lift
    # goes through a vessel. This also refuses some correct ones (a named pulmonary vein).
    "UBERON:0001981": "blood_vessel",
    "UBERON:0002049": "vasculature",
    # Added after an independent review of 100 random lifts (3 wrong, 9 doubtful): a point, a
    # groove or a surface is the boundary of the mask, not inside it; a foramen is a hole that
    # carries vessels and nerves; lymph nodes and nerves lie next to the organ named after them.
    "UBERON:0000466": "immaterial_entity",
    "UBERON:0036215": "surface_region",
    "UBERON:0006984": "surface",
    "UBERON:0005744": "bone_foramen",
    "UBERON:0000029": "lymph_node",
    "UBERON:0001021": "nerve",
    # Added after a second, blind review (1 wrong, 13 doubtful of 100): ventricles are fluid
    # cavities and lobar bronchi start at the hilum, outside the masks that usually leave them out;
    # a fontanelle is a membrane between infant bones.
    "UBERON:0005358": "ventricle_cavity",
    "UBERON:0002185": "bronchus",
    "UBERON:0002221": "fontanelle",
}


def _stem(cid: str) -> str:
    """The class without its side: kidney_left -> kidney, rib_right_3 -> rib_3."""
    return cid.replace("_left", "").replace("_right", "")


@dataclass(frozen=True)
class Lift:
    cid: str
    relation: str  # equal, part_of, kind_of
    depth: int
    path: tuple[str, ...]


def _side_of(cid: str) -> str | None:
    if cid.endswith("_right") or cid.startswith("rib_right_"):
        return _RIGHT
    if cid.endswith("_left") or cid.startswith("rib_left_"):
        return _LEFT
    return None


@dataclass
class AnatomyGraph:
    names: dict[str, str]
    parents: dict[str, list[tuple[str, str]]]  # child -> [(parent, "is_a"|"part_of")]
    disjoint: dict[str, set[str]]
    anchors: dict[str, str]  # UBERON id -> class
    # UBERON id -> the sided classes it is the unsided form of
    families: dict[str, frozenset[str]]
    # UBERON id -> why this term never climbs to a class (curated, in the anchors file)
    refused: dict[str, str] = field(default_factory=dict)
    children: dict[str, set[str]] = field(default_factory=dict)
    # UBERON id -> its EXACT and RELATED synonyms of two words or more ("aortic arch artery" is a
    # synonym of "pharyngeal arch artery"): other written names of the same structure, so a
    # mention inside one of them is a word of that name.
    synonyms: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.children:
            kids: dict[str, set[str]] = defaultdict(set)
            for child, edges in self.parents.items():
                for parent, _ in edges:
                    kids[parent].add(child)
            self.children = dict(kids)
        self._class_nodes: dict[str, set[str]] = defaultdict(set)
        for node, cid in self.anchors.items():
            self._class_nodes[cid].add(node)

    @classmethod
    def from_obo(
        cls, lines: Iterable[str], anchors: dict[str, Any] | None = None
    ) -> "AnatomyGraph":
        """Read is-a, part-of and spatial disjointness from an OBO file (uberon-basic)."""
        if anchors is None:
            anchors = json.loads(DEFAULT_ANCHORS.read_text("utf-8"))
        names: dict[str, str] = {}
        synonyms: dict[str, list[str]] = defaultdict(list)
        parents: dict[str, list[tuple[str, str]]] = defaultdict(list)
        disjoint: dict[str, set[str]] = defaultdict(set)
        current: str | None = None
        obsolete: set[str] = set()
        for raw in lines:
            line = raw.rstrip("\n")
            if line.startswith("["):
                current = None if line != "[Term]" else ""
                continue
            if current is None:
                continue
            if line.startswith("id: "):
                current = line[4:].strip()
            elif not current:
                continue
            elif line.startswith("name: "):
                names[current] = line[6:].strip()
            elif line.startswith("synonym: "):
                said = _SYNONYM.match(line)
                if said:
                    synonyms[current].append(said.group(1))
            elif line.startswith("is_a: "):
                parents[current].append((line[6:].split()[0], "is_a"))
            elif line.startswith("relationship: "):
                kind, target = line[14:].split()[:2]
                if kind == "part_of":
                    parents[current].append((target, "part_of"))
                elif kind == "mutually_spatially_disjoint_with":
                    disjoint[current].add(target)
                    disjoint[target].add(current)
            elif line.startswith("is_obsolete: true"):
                obsolete.add(current)
        for node in obsolete:
            parents.pop(node, None)
            names.pop(node, None)
            synonyms.pop(node, None)
        direct: dict[str, str] = {}
        family: dict[str, set[str]] = defaultdict(set)
        for cid, entry in anchors["anchors"].items():
            for node in entry.get("uberon", {}):
                if node in direct and direct[node] != cid:
                    raise ValueError(f"{node} anchored to {direct[node]} and {cid}")
                direct[node] = cid
            for node in entry.get("family", {}):
                if _side_of(cid) is None:
                    raise ValueError(f"family anchor {node} on unsided class {cid}")
                family[node].add(cid)
        return cls(
            names=names,
            parents=dict(parents),
            disjoint=dict(disjoint),
            anchors=direct,
            families={k: frozenset(v) for k, v in family.items()},
            refused=dict(anchors.get("refused", {})),
            synonyms=dict(synonyms),
        )

    def _classes_at(self, node: str, sides: frozenset[str]) -> tuple[set[str], str]:
        """Classes this node stands for, given the sides the mention states; and a refusal reason."""
        if node in self.anchors:
            cid = self.anchors[node]
            side = _side_of(cid)
            if side and sides and side not in sides:
                return set(), "the_class_is_on_the_other_side"
            return {cid}, ""
        members = self.families.get(node)
        if not members:
            return set(), ""
        if not sides:
            return set(), "part_of_a_paired_structure_without_a_side"
        if len(sides) > 1:
            return set(), "mention_names_both_sides"
        chosen = {m for m in members if _side_of(m) in sides}
        return chosen, "" if chosen else "no_class_on_the_stated_side"

    def kinds(self, node: str) -> set[str]:
        """Every is-a ancestor of ``node``."""
        seen: set[str] = set()
        stack = [node]
        while stack:
            for parent, kind in self.parents.get(stack.pop(), ()):
                if kind == "is_a" and parent not in seen:
                    seen.add(parent)
                    stack.append(parent)
        return seen

    def lift(
        self, node: str, sides: Iterable[str] = (), max_depth: int = 4
    ) -> Lift | str | None:
        """The nearest segmentation class above ``node``; a refusal reason; or None (no class near)."""
        result = self._climb(node, sides, max_depth)
        if isinstance(result, Lift) and result.depth:
            if result.relation == "kind_of":
                # "xiphoid cartilage" is-a "costal cartilage" in UBERON, but it is not one of the
                # costal cartilages of the segmentation: a kind of X is not X.
                return "a_kind_of_the_class_is_not_the_class"
            # Every class the term reaches, not only the nearest: UBERON puts the duodenum inside the
            # small intestine, the segmentation does not, so a duodenal term reaches both. Two
            # classes is a disagreement, whatever their depth (third blind review).
            reachable = self._reachable(node, frozenset(sides), max_depth + 2)
            if reachable - {result.cid}:
                return "graph_reaches_two_classes"
            for step in result.path[:-1]:
                if step in self.refused:
                    return "listed_as_outside_the_class"
                refused = self.kinds(step) & set(REFUSED_KINDS)
                if step in REFUSED_KINDS or refused:
                    kind = REFUSED_KINDS[
                        step if step in REFUSED_KINDS else min(refused)
                    ]
                    return f"{kind}_is_not_inside_the_class"
        return result

    def _reachable(self, node: str, sides: frozenset[str], depth: int) -> set[str]:
        """Classes anchored anywhere among the ancestors of ``node`` up to ``depth`` steps."""
        sides = frozenset(s for s in sides if s in (_RIGHT, _LEFT))
        seen = {node}
        frontier = [node]
        classes: set[str] = set()
        for _ in range(depth):
            step = []
            for current in frontier:
                for parent, _kind in self.parents.get(current, ()):
                    if parent not in seen:
                        seen.add(parent)
                        step.append(parent)
            frontier = step
        for current in seen:
            if current in self.anchors:
                classes.add(self.anchors[current])
            for member in self.families.get(current, ()):
                if not sides or _side_of(member) in sides:
                    classes.add(member)
        return classes

    def _climb(
        self, node: str, sides: Iterable[str], max_depth: int
    ) -> Lift | str | None:
        sides = frozenset(s for s in sides if s in (_RIGHT, _LEFT))
        found, why = self._classes_at(node, sides)
        if why:
            return why
        if found:
            # An anchor, or the unsided term plus the side the mention states ("kidney" + right).
            return Lift(found.pop(), "equal", 0, (node,))
        # Breadth-first climb; each node remembers how it was first reached.
        reached: dict[str, tuple[str | None, bool]] = {node: (None, False)}
        frontier = [node]
        for depth in range(1, max_depth + 1):
            step: list[str] = []
            for current in frontier:
                via_part = reached[current][1]
                for parent, kind in self.parents.get(current, ()):
                    if parent not in reached:
                        reached[parent] = (current, via_part or kind == "part_of")
                        step.append(parent)
            hits: dict[str, str] = {}
            reasons = set()
            for candidate in step:
                classes, why = self._classes_at(candidate, sides)
                if why:
                    reasons.add(why)
                for cid in classes:
                    hits.setdefault(cid, candidate)
            if reasons and not hits:
                return sorted(reasons)[0]
            if len(hits) > 1:
                return "graph_parents_disagree"
            if hits:
                cid, at = next(iter(hits.items()))
                path = [at]
                while reached[path[-1]][0] is not None:
                    path.append(reached[path[-1]][0])  # type: ignore[arg-type]
                relation = "part_of" if reached[at][1] else "kind_of"
                return Lift(cid, relation, depth, tuple(reversed(path)))
            frontier = step
            if not frontier:
                break
        return None

    def developing(self) -> frozenset[str]:
        """The nodes below "developing anatomical structure" or "embryonic structure"."""
        if getattr(self, "_developing", None) is None:
            memo: dict[str, bool] = {}

            def below(node: str, trail: frozenset[str] = frozenset()) -> bool:
                if node in memo:
                    return memo[node]
                if node in _DEVELOPING_ROOTS:
                    return True
                if node in trail:
                    return False
                found = any(
                    below(parent, trail | {node})
                    for parent, _ in self.parents.get(node, ())
                )
                memo[node] = found
                return found

            self._developing = frozenset(n for n in self.names if below(n))
        return self._developing

    def nodes_of(self, cid_or_node: str) -> set[str]:
        """The UBERON nodes a class stands for (its anchors), or the node itself."""
        nodes = set(self._class_nodes.get(cid_or_node, ()))
        nodes |= {n for n, members in self.families.items() if cid_or_node in members}
        return nodes or {cid_or_node}

    def related(self, a: str, b: str) -> str | None:
        """How two candidates are close in the graph, or None. Classes or UBERON ids."""
        if a == b:
            return None
        side_a, side_b = _side_of(a), _side_of(b)
        if side_a and side_b and side_a != side_b and _stem(a) == _stem(b):
            return "the_other_side"
        nodes_a, nodes_b = self.nodes_of(a), self.nodes_of(b)
        for x in nodes_a:
            parents_x = {p for p, _ in self.parents.get(x, ())}
            for y in nodes_b:
                parents_y = {p for p, _ in self.parents.get(y, ())}
                if y in parents_x or x in parents_y:
                    return "parent_and_child"
                if y in self.disjoint.get(x, ()):
                    return "declared_disjoint"
                shared = {
                    p
                    for p in parents_x & parents_y
                    if len(self.children.get(p, ())) <= _MAX_SIBLING_FANOUT
                }
                if shared:
                    return "sisters"
        return None
