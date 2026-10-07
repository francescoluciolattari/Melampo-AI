"""Which written forms can mean more than one thing: found from the data, not one error at a time.

The sense inventory (``word_senses``) decides *in context* which meaning a listed form has. What
it cannot do is know which forms to list. A reader knows a word is ambiguous before the sentence
settles it; this module is that knowledge, found by scanning the names the linker accepts
(lexicon, part table) against structural signals, so a new ambiguous form is surfaced for a sense
profile instead of waiting for an error in a report:

* ``shared``     the same written key names more than one class;
* ``head_dropped`` the key lost a head noun ("left paraspinal" from "left paraspinal muscles"):
                 what is left is an adjective or a region, and the sentence must say which;
* ``short``      an abbreviation written in capitals (SVC, LM, GB, LAD, IVC, CBD);
* ``adjective``  a single adjectival word, which describes a structure rather than names it;
* ``region``     an adjective with a spatial prefix (para-, peri-, retro-, sub-, supra-...): the
                 region around a structure, not the structure;

A flag is a reason to look, not a verdict: a flagged form with a sense profile is read in context;
a flagged form without one is *exposed* (the linker accepts it from the name alone), and the
exposure on real text is what ranks the work. It models nothing about meaning, and an unflagged
form can still be ambiguous (the signals are structural); the gold set is what measures that.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from melampo.memory.anatomy_linker import (
    _ADJECTIVES,
    Lexicon,
    normalise,
)

# Adjectival endings in English and Italian. A heuristic: "adrenal" and "paraspinal" match,
# "gland" and "fegato" do not.
_ADJECTIVAL = re.compile(
    r"(?:al|ial|ic|ar|ary|ive|oid|ous|ale|ali|ico|ici|ica|iche|are|ari|ario|ale|ivo|ive|oso|osa)$"
)
# A spatial prefix on an adjective names a region around a structure.
_SPATIAL_PREFIX = re.compile(
    r"^(?:para|peri|retro|pre|sub|supra|infra|inter|intra|extra|juxta|circum|sovra|sotto)"
)


@dataclass
class FormFlag:
    key: tuple[str, ...]
    classes: frozenset[str]
    names: tuple[str, ...]
    signals: set[str] = field(default_factory=set)


def _is_adjectival(token: str) -> bool:
    return len(token) > 4 and bool(_ADJECTIVAL.search(token))


def audit(
    lexicon: Lexicon,
    part_names: Iterable[tuple[str, str]] = (),
) -> dict[tuple[str, ...], FormFlag]:
    """Flags per normalised key. ``part_names`` are (name, class) pairs from the part table."""
    names_of: dict[tuple[str, ...], set[str]] = defaultdict(set)
    classes_of: dict[tuple[str, ...], set[str]] = defaultdict(set)
    dropped: set[tuple[str, ...]] = set()
    for cid, entry in lexicon.classes.items():
        for name in entry["it"] + entry["en"]:
            key = normalise(name)
            names_of[key].add(name)
            classes_of[key].add(cid)
            if normalise(name, keep_noise=True) != key:
                dropped.add(key)
    for name, cid in part_names:
        key = normalise(name)
        names_of[key].add(name)
        classes_of[key].add(cid)
        if normalise(name, keep_noise=True) != key:
            dropped.add(key)

    flags: dict[tuple[str, ...], FormFlag] = {}
    for key, names in names_of.items():
        signals: set[str] = set()
        if len(classes_of[key]) > 1:
            signals.add("shared")
        if key in dropped:
            signals.add("head_dropped")
        for name in names:
            # the words as written, side and number words aside: "adrenal", not "surrene"
            words = [t for t in normalise(name, map_words=False) if t[:1] not in "<#@"]
            if not words:
                continue
            if any(
                w.isupper() and 2 <= len(w) <= 4 for w in re.findall(r"[A-Za-z]+", name)
            ):
                signals.add("short")
            if len(words) == 1:
                word = words[0]
                if word in _ADJECTIVES or _is_adjectival(word):
                    signals.add("adjective")
            if _SPATIAL_PREFIX.match(words[-1]) and (
                _is_adjectival(words[-1]) or words[-1] in _ADJECTIVES
            ):
                signals.add("region")
        if signals:
            flags[key] = FormFlag(
                key, frozenset(classes_of[key]), tuple(sorted(names)), signals
            )
    return flags


def exposed(
    flags: dict[tuple[str, ...], FormFlag], listed_forms: Iterable[str]
) -> dict[tuple[str, ...], FormFlag]:
    """Flagged keys none of whose words is a listed (folded) form of the sense inventory."""
    listed = set(listed_forms)
    return {key: flag for key, flag in flags.items() if not (listed & {t for t in key})}


def keys(flags: dict[tuple[str, ...], FormFlag]) -> frozenset[tuple[str, ...]]:
    """The flagged keys, as the linker's ``ambiguous`` field takes them."""
    return frozenset(flags)
