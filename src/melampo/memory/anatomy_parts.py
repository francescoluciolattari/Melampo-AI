"""Curated part-of knowledge: "sigma", "lingula", "testa femorale sinistra" -> the whole they belong to.

A reader knows that the sigmoid is part of the colon and the femoral head part of the
femur. That is knowledge, not similarity, so it is a table written from anatomy
(``data/linking/anatomy_parts.json``, built by ``scripts/build_anatomy_parts.py``), not a
model's guess. The link says what it is: ``relation`` is ``equal`` (synonym), ``part_of``,
``contour_of`` (the organ's outline on a radiograph) or ``approx``. A part is never
reported as the whole.

Two ways to match, both exact after the linker's normalisation:

* a whole expression in ``direct`` ("colon discendente");
* a part word plus the name of a whole it is allowed to belong to ("polo" + "milza").
  "testa del fegato" is not linked: the liver has no head in the table.

Side comes from the mention. For a class that exists on both sides (femur, hip) the
mention must state it, and it picks the class; for a single structure (the pancreas) a stated
side describes the part and is kept out of the choice.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from .anatomy_linker import (
    POS_DOWN,
    POS_MID,
    POS_UP,
    SIDE_BOTH,
    SIDE_LEFT,
    SIDE_RIGHT,
    Lexicon,
    _bare,
    _fold,
    _NEURO_CUES,
    _NOISE_CANON,
    _raw_tokens,
    normalise,
)

_SIDES = frozenset((SIDE_RIGHT, SIDE_LEFT, SIDE_BOTH))
_POSITIONS = frozenset((POS_UP, POS_MID, POS_DOWN))


@dataclass(frozen=True)
class PartLink:
    cid: str
    relation: str
    part: str
    reason: str


_HARMLESS_NOISE = frozenset(("level",))
_TISSUE_OF_THE_WHOLE = frozenset(("wall", "parenchyma", "body"))
_NON_NEURAL_WHOLES = frozenset(("heart",))


@dataclass
class PartTable:
    lexicon: Lexicon
    strict: dict[frozenset[str], list[tuple[str, str, str, frozenset[str]]]] = field(
        default_factory=dict
    )
    parts: dict[str, tuple[frozenset[str], str]] = field(default_factory=dict)
    never: dict[frozenset[str], str] = field(default_factory=dict)
    # One-word names written only in Italian ("sigma", "anca"): not names in an English sentence.
    italian_only: frozenset[frozenset[str]] = frozenset()
    _whole_names: dict[tuple[str, ...], set[str]] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict[str, Any], lexicon: Lexicon) -> "PartTable":
        table = cls(lexicon=lexicon)
        for entry in data["direct"]:
            for name in entry["names"]:
                tokens = normalise(name, keep_noise=True)
                key = frozenset(t for t in tokens if t not in _SIDES)
                named = frozenset(t for t in tokens if t in _SIDES)
                table.strict.setdefault(key, []).append(
                    (entry["whole"], entry["relation"], name, named)
                )
        for entry in data.get("never", ()):
            for name in entry["names"]:
                table.never[frozenset(normalise(name, keep_noise=True))] = entry[
                    "reason"
                ]
        table.italian_only = frozenset(
            frozenset(normalise(name, keep_noise=True))
            for name in data.get("italian_only", {}).get("names", ())
        )
        for entry in data["parts"]:
            for word in entry["words"]:
                for token in normalise(word):
                    table.parts[token] = (frozenset(entry["wholes"]), entry["relation"])
        for key, hits in lexicon.index.items():
            if any(region is None for _, region in hits):
                bare = tuple(sorted(_bare(key)))
                table._whole_names.setdefault(bare, set()).update(
                    cid for cid, region in hits if region is None
                )
        return table

    def _family(self, whole: str) -> list[str]:
        classes = self.lexicon.classes
        if whole in classes:
            return [whole]
        return [c for c in (f"{whole}_left", f"{whole}_right") if c in classes]

    def _by_side(self, whole: str, sides: frozenset[str]) -> tuple[str | None, str]:
        family = self._family(whole)
        if not family:
            return None, "whole_not_in_the_lexicon"
        wanted = {SIDE_RIGHT: "_right", SIDE_LEFT: "_left"}
        if len(family) == 1:
            only = family[0]
            for token, suffix in wanted.items():
                other = [t for t in wanted if t != token][0]
                if only.endswith(suffix) and other in sides:
                    return None, "side_contradicts_the_structure"
            return only, ""
        if len(sides) != 1 or next(iter(sides)) not in wanted:
            return None, "part_of_a_paired_structure_without_a_side"
        suffix = wanted[next(iter(sides))]
        return f"{whole}{suffix}", ""

    def _guard(self, whole: str, sentence: str, mention: str = "") -> str | None:
        """A heart part named in a sentence about the brain is a ventricle of the brain."""
        if whole in _NON_NEURAL_WHOLES and sentence:
            words = {_fold(t) for t in _raw_tokens(sentence)} - {
                _fold(t) for t in _raw_tokens(mention)
            }
            if words & _NEURO_CUES:
                return "sentence_is_about_the_brain_not_this_structure"
        return None

    def resolve(self, mention: str, sentence: str = "") -> PartLink | str | None:
        """A link, an abstention reason (a known part that cannot be placed), or None (not in the table)."""
        kept = normalise(mention, keep_noise=True)
        trap = self.never.get(frozenset(kept))
        if trap:
            return trap
        sides = frozenset(t for t in kept if t in _SIDES)
        key = frozenset(t for t in kept if t not in _SIDES)
        hits = self.strict.get(key)
        if hits:
            wholes = {whole for whole, _, _, _ in hits}
            if len(wholes) > 1:
                return "part_name_belongs_to_several_structures"
            whole, relation, name, named = hits[0]
            if named and sides != named:
                return "the_name_carries_a_side_the_mention_does_not_state"
            cid, why = self._by_side(whole, sides)
            if cid is None:
                return why
            blocked = self._guard(whole, sentence, mention)
            if blocked:
                return blocked
            return PartLink(cid, relation, name, "known_part")
        dropped = normalise(mention)
        noise = {_NOISE_CANON.get(t, t) for t in set(kept) - set(dropped)}
        if noise - _HARMLESS_NOISE:
            if "lumen" in noise:
                return "a_lumen_is_a_space_not_the_organ"
            if noise - _HARMLESS_NOISE <= _TISSUE_OF_THE_WHOLE:
                remainder = tuple(sorted(t for t in dropped if t not in _SIDES))
                families = {
                    self._family_of(c) for c in self._whole_names.get(remainder, set())
                }
                if len(families) != 1:
                    return None
                whole = next(iter(families))
                cid, why = self._by_side(whole, sides)
                if cid is None:
                    return why
                blocked = self._guard(whole, sentence, mention)
                if blocked:
                    return blocked
                return PartLink(
                    cid,
                    "part_of",
                    sorted(noise - _HARMLESS_NOISE)[0],
                    "tissue_of_the_whole",
                )
            return "tissue_word_changes_the_structure"
        core = frozenset(dropped) - _SIDES - _POSITIONS
        found = [t for t in core if t in self.parts]
        if len(found) != 1:
            return None
        word = found[0]
        allowed, relation = self.parts[word]
        remainder = tuple(sorted(core - {word}))
        if not remainder:
            return None
        names = self._whole_names.get(remainder, set())
        families = {self._family_of(c) for c in names}
        usable = {f for f in families if f in allowed}
        if len(usable) != 1:
            return None
        whole = next(iter(usable))
        cid, why = self._by_side(whole, sides)
        if cid is None:
            return why
        blocked = self._guard(whole, sentence, mention)
        if blocked:
            return blocked
        return PartLink(cid, relation, word, "part_word_with_whole")

    @staticmethod
    def _family_of(cid: str) -> str:
        for suffix in ("_left", "_right"):
            if cid.endswith(suffix):
                return cid[: -len(suffix)]
        return cid


def strip_wrapper(mention: str) -> str:
    """ "a carico della colecisti" names the gallbladder: the phrase only says what is affected."""
    stripped = re.sub(
        r"^\s*a\s+carico\s+(?:(?:dello|della|delle|degli|del|dei|di)(?=\s|$)|dell['’])\s*",
        "",
        mention,
        flags=re.IGNORECASE,
    )
    return stripped or mention
