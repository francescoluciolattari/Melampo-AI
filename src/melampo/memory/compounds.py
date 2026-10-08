"""A structure's name inside a longer name that is not the structure's: three signs in the writing.

The mention is what the linker proposes (a known name). A reader looks at what surrounds it before
believing it names the structure: the word may be a piece of a compound ("gut-brain axis"), a piece
of a name the text itself defines ("TWIK-related spinal cord K(+) (TRESK) channel"), or the material
of a graft ("autologous costal cartilage"). Each sign is read from the writing alone, with no
word list of structures and no model, and each is the same for every structure.

1. **Hyphen compound** (``hyphen_compound``). "gut-brain", "pro-brain", "donor-specific spleen":
   the mention is joined by a hyphen to another word, so it is a piece of a compound word. The
   words that only say where (a prefix such as "intra" or "peri", a side, a number) do not count.
2. **Defined abbreviation** (``defined_abbreviation``). The text writes a long form and, in
   brackets, its abbreviation ("liver fatty acid binding protein (L-FABP)"). A mention that is a
   part of the long form, but not the whole of it, is a word of a name the text defines, not a
   structure. The long form is found with the algorithm of Schwartz and Hearst (2003, "A simple
   algorithm for identifying abbreviation definitions in biomedical text", PSB 8:451-462): the
   letters of the abbreviation must appear, in order, in the words before the bracket, the first
   at the start of a word. Nothing is decided when no long form is found.
3. **Material qualifier** (``material_qualifier``, in ``word_senses``): data, not code.

What they do not do: decide that the structure is *not* meant. They say it is *not the name it
looks like*; the linker then abstains with the reason, and a reader or a model may still link the
longer name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .word_senses import _fold, locate, noun_phrase_after, phrase_head

_LETTERS = "A-Za-zÀ-ÿ"
# hyphen, non-breaking hyphen, en dash: joined only when there is no space on either side
_HYPHENS = "\\-\u2010\u2011\u2013"

# Words joined by a hyphen that only say where, which side or how many: the structure is still
# the structure ("intra-hepatic", "peri-renal", "right-sided"). Closed class, same for every
# structure. A prefix that makes a different thing ("anti", "pro", "pre-pro", "non") is not here.
_WHERE = frozenset(
    ["pre", "post", "peri", "sub", "supra", "infra", "intra", "extra", "inter", "para", "retro", "trans", "endo", "epi", "ecto", "meso", "peri", "antero", "postero", "latero", "medio", "dorso", "ventro", "cranio", "caudo", "cephalo", "capo", "left", "right", "sx", "dx", "bilateral", "sided", "side", "lateral", "medial", "anterior", "posterior", "superior", "inferior", "upper", "lower", "mid", "proximal", "distal", "central", "ventral", "dorsal"]
)


def hyphen_compound(mention: str, sentence: str, start: int | None = None) -> str:
    """The word the mention is joined to by a hyphen ("gut-brain" -> "gut"), else "".

    Joined means a hyphen with a letter on both sides. Numbers ("C5-C6", "L4-5", "T12-L1") and the
    words of ``_WHERE`` are not compounds."""
    found = locate(mention, sentence, start)
    if not found:
        return ""
    before, after = sentence[: found.start()], sentence[found.end() :]
    for pattern, text in (
        (re.compile(rf"([{_LETTERS}]{{2,}})[{_HYPHENS}]$"), before),
        (re.compile(rf"^[{_HYPHENS}]([{_LETTERS}]{{2,}})"), after),
    ):
        near = pattern.search(text)
        if not near:
            continue
        word = near.group(1).lower()
        if word in _WHERE:
            continue
        return word
    return ""


# -- defined abbreviations (Schwartz and Hearst 2003) ----------------------------------------------

_BRACKETED = re.compile(r"[(\[]([^()\[\]]{2,30})[)\]]")


@dataclass(frozen=True)
class Definition:
    short: str
    long: str
    start: int  # of the long form in the sentence
    end: int
    after: int = 0  # where the bracket closes


def _best_long_form(short: str, candidate: str) -> tuple[str, int] | None:
    """The long form in ``candidate`` (the words before the bracket), and where it starts.

    The algorithm of Schwartz and Hearst: walk both strings from the right; each alphanumeric
    letter of the abbreviation must be found in the candidate, and the first letter of the
    abbreviation must be the first letter of a word."""
    s_index, l_index = len(short) - 1, len(candidate) - 1
    while s_index >= 0:
        char = short[s_index].lower()
        if not char.isalnum():
            s_index -= 1
            continue
        while l_index >= 0 and (
            candidate[l_index].lower() != char
            or (
                s_index == 0
                and l_index > 0
                and candidate[l_index - 1].isalnum()
            )
        ):
            l_index -= 1
        if l_index < 0:
            return None
        l_index -= 1
        s_index -= 1
    start = candidate.rfind(" ", 0, l_index + 1) + 1
    long = candidate[start:]
    stripped = long.lstrip("\"'“”‘’")
    return stripped.rstrip("\"'“”‘’"), start + len(long) - len(stripped)


def _short_form_ok(short: str) -> bool:
    words = short.split()
    if not 1 <= len(words) <= 2 or len(short) < 2:
        return False
    if not short[0].isalnum() or not any(c.isalpha() for c in short):
        return False
    # an abbreviation is mostly capitals or digits ("L-FABP", "NT-proBNP", "DST"), not a word
    caps = sum(c.isupper() or c.isdigit() for c in short)
    return caps >= 2 or (caps >= 1 and len(short) <= 3)


def definitions(sentence: str) -> list[Definition]:
    """Every "long form (SHORT)" and "SHORT (long form)" in the sentence."""
    found: list[Definition] = []
    for bracket in _BRACKETED.finditer(sentence):
        inside = bracket.group(1).strip()
        before = sentence[: bracket.start()].rstrip()
        if _short_form_ok(inside):
            # "long form (SHORT)": the long form is among the words before the bracket
            words = before.split()
            limit = min(len(inside) + 5, len(inside) * 2)
            window = words[-limit:]
            candidate = " ".join(window)
            first = len(before) - len(candidate)
            hit = _best_long_form(inside, candidate)
            if hit and len(hit[0]) > len(inside) and hit[0].split()[0].lower() not in (
                "the",
                "a",
                "an",
            ):
                long, offset = hit
                found.append(
                    Definition(
                        inside,
                        long,
                        first + offset,
                        first + offset + len(long),
                        bracket.end(),
                    )
                )
        elif before:
            # "SHORT (long form)": the abbreviation is the word before the bracket
            word = before.split()[-1]
            if _short_form_ok(word) and len(inside) > len(word):
                hit = _best_long_form(word, inside)
                if hit and len(hit[0]) > len(word):
                    long, offset = hit
                    base = bracket.start() + 1 + (len(bracket.group(1)) - len(bracket.group(1).lstrip()))
                    found.append(
                        Definition(
                            word,
                            long,
                            base + offset,
                            base + offset + len(long),
                            bracket.end(),
                        )
                    )
    return found


def _bare(text: str) -> str:
    return " ".join(re.findall(rf"[{_LETTERS}0-9]+", text.lower()))


def names_another_thing(
    definition: Definition,
    sentence: str,
    heads: frozenset[str],
    abbreviation_is_mentioned: bool = False,
) -> str:
    """The head that makes a defined name something other than a structure, else "".

    The last word of the long form ("... fatty acid binding **protein** (L-FABP)") decides. When the
    long form ends in a symbol or a one- or two-letter word ("TWIK-related spinal cord K(+)
    (TRESK) **channel**"), its head stands after the bracket. When the mention is the abbreviation
    itself ("(DENS) **scale**") the phrase after the bracket counts too. A long form that ends in an
    ordinary word ("spinal cord injury (SCI) model") is a condition of the structure, and the
    structure stays named."""
    words = re.findall(rf"[{_LETTERS}0-9]+", definition.long)
    found = phrase_head([_fold(w) for w in words], heads)
    if found:
        return found
    ends_in_a_symbol = not definition.long[-1:].isalpha() or (
        bool(words) and len(words[-1]) <= 2
    )
    if abbreviation_is_mentioned or ends_in_a_symbol:
        return phrase_head(noun_phrase_after(sentence, definition.after), heads)
    return ""


def defined_abbreviation(
    mention: str,
    sentence: str,
    start: int | None = None,
    heads: frozenset[str] = frozenset(),
) -> str:
    """The abbreviation whose long form the mention is a part of, but not the whole of, when the
    long form names another thing (``heads``: protein, peptide, channel, scale, response...), else "".

    "spinal cord" in "TWIK-related spinal cord K(+) (TRESK) channel" -> "TRESK". "inferior vena
    cava" in "inferior vena cava (IVC)" -> "" (the long form is the mention itself). "heart" in
    "congenital heart disease (CHD)" -> "" (a disease of the structure names the structure), and so
    for "Mouse Brain Library (MBL)": the head says whether the name is a structure's."""
    found = locate(mention, sentence, start)
    if not found:
        return ""
    own = _bare(mention)
    for definition in definitions(sentence):
        if not (definition.start <= found.start() and found.end() <= definition.end):
            continue
        if _bare(definition.long) == own:
            continue
        if not names_another_thing(definition, sentence, heads):
            continue
        return definition.short
    return ""


def defined_short_form(
    mention: str,
    sentence: str,
    start: int | None = None,
    heads: frozenset[str] = frozenset(),
) -> str:
    """The long form the text gives to the mention when the mention *is* the abbreviation
    ("DENS" in 'drug effects on the nervous system (DENS) scale') and that long form names another
    thing (``heads``), else ""."""
    found = locate(mention, sentence, start)
    if not found:
        return ""
    for definition in definitions(sentence):
        if _bare(definition.short) == _bare(mention) and names_another_thing(
            definition, sentence, heads, abbreviation_is_mentioned=True
        ):
            return definition.long
    return ""
