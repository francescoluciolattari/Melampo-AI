"""How a chat model is asked to read the phrase a mention stands in, and how its answer is checked.

Used by the probes (``scripts/phrase_probe.py``, ``scripts/role_probe.py``) and by the LoRA data builder
(``scripts/lora_roles.py``); the linker does not use it. Written on 12 October 2026 after the informed
phrase-probe run, where one model answered with the bare word in 81% of the cases ("brain" for
"brain weights"). Three causes, all ours:

1. **One mark, two meanings.** In the annotated examples ``[[ ]]`` enclosed the whole concept (the
   answer); in the sentence to read it enclosed only the word. The model learned from the examples
   that what is inside the mark is the concept.
2. **Examples leaning to the bare word.** They were the most similar annotated sentences with the same
   word, and in those the word is often annotated alone.
3. **No check.** "brain / anatomical structure / body site: no" contradicts itself and was accepted;
   the question ("the concept whose phrase contains the word") is satisfied by the word itself.

Here: one mark (``<w> </w>``) that always means "the word asked about", in the examples too, where the
answer stands only after the arrow; examples balanced so that at least one has a phrase longer than the
word when there is one; a structured answer (phrase copied from the sentence, head, type, role of the
word in our schema or ``uncertain``); a deterministic check (the phrase is a contiguous piece of the
sentence that covers the word, the head is a word of the phrase, the role is in the schema) and **one**
explicit follow-up question when the phrase stops at the word while the sentence goes on with more words
of a phrase ("Does «weights» belong to the same phrase?"). Rejections, follow-ups and changed answers
are counted, never hidden.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

TYPES = (
    "anatomical structure", "disease or finding", "procedure", "device or object", "food",
    "protein or gene", "chemical or drug", "organism", "process or function",
    "measurement or property", "other",
)

# The role of the word inside the phrase (the schema of the gold set, plus "uncertain").
ROLES = {
    "structure": "the word names the body structure the text speaks of (a lesion IN it, its shape, the organ itself)",
    "procedure_site": "the structure is the site of a procedure or exam named in the phrase or by the word that governs it, also outside the phrase (liver biopsy, CT of the chest, liberation of the caecum and of the appendix)",
    "device_site": "the structure is where a device is or goes (vena cava filter, biliary stent)",
    "inherent_location": "the phrase names a measure, property, function or process of the structure (brain weight, liver function)",
    "source_of": "the structure is where a substance, cell or sample comes from (liver enzymes, bone marrow cells)",
    "inside_a_name": "the word is part of the name of a gene, protein, chemical, organism or other thing (Heart and Soul gene)",
    "not_a_body_site": "the word does not name a place in a body here (heart of the matter, a cheese)",
    "other": "none of the above",
    "uncertain": "the sentence does not let you decide",
}
SITE_ROLES = frozenset({"structure", "procedure_site", "device_site", "inherent_location", "source_of"})
OPEN, CLOSE = "<w>", "</w>"

_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9'’\-/]*")
# Words that never continue a noun phrase to the right (verbs of reporting are not listed: a follow-up
# question is cheap, a missed head is not).
_STOP_RIGHT = frozenset(
    """a an the of in on at to from by with without for and or but nor as than that which who whom whose
    is are was were be been being has have had do does did will would can could may might shall should
    must not no this these those it its il lo la i gli le un una uno di da del della dei degli delle in
    con su per tra fra e o ma che non è sono era erano ha hanno al allo alla ai agli alle nel nella nei
    negli nelle dal dalla dai dagli dalle sul sulla sui sugli sulle""".split()
)


def mark(sentence: str, start: int, end: int) -> str:
    """The sentence with the word asked about between ``<w>`` and ``</w>``."""
    return f"{sentence[:start]}{OPEN}{sentence[start:end]}{CLOSE}{sentence[end:]}"


def _norm(text: str) -> str:
    return _SPACE.sub(" ", re.sub(r"\s*-\s*", "-", text.replace("’", "'"))).strip().lower()


# Articles and prepositions at the edge of the marked words: a phrase that leaves them out still covers the
# words that name the thing ("nelle membrane basali" -> "membrane basali glomerulari", 14 Oct 2026).
_EDGE_WORDS = frozenset(
    """the a an of in on at to from by with for and or
    il lo la i gli le l un uno una un' di del dello della dei degli delle dell a al allo alla ai agli alle all
    da dal dallo dalla dai dagli dalle dall in nel nello nella nei negli nelle nell su sul sullo sulla sui
    sugli sulle sull con per tra fra e""".split()
)


def next_words(sentence: str, end: int, limit: int = 3) -> list[str]:
    """Up to ``limit`` words right after the mention, before punctuation or a function word: the words
    that *could* continue its phrase (a candidate, not a parse; the model decides)."""
    out = []
    for m in _WORD.finditer(sentence, end):
        between = sentence[end if not out else out[-1][1]: m.start()]
        if between.strip():  # punctuation between: the phrase cannot go on
            break
        if m.group(0).lower() in _STOP_RIGHT:
            break
        out.append((m.group(0), m.end()))
        if len(out) >= limit:
            break
    return [w for w, _ in out]


def covering(sentence: str, phrase: str, start: int, end: int) -> tuple[int, int] | None:
    """Where ``phrase`` sits in the sentence covering ``[start, end)`` (spaces and case folded)."""
    want = _norm(phrase)
    if not want:
        return None
    # a map from the folded sentence back to offsets
    chars, offsets = [], []
    prev_space = True
    text = sentence.replace("’", "'")
    for i, ch in enumerate(text):
        if ch.isspace():
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if prev_space or nxt == "-" or (chars and chars[-1] == "-"):
                continue
            chars.append(" ")
            prev_space = True
        else:
            chars.append(ch.lower())
            prev_space = False
        offsets.append(i)
    folded = "".join(chars)
    at = folded.find(want)
    while at >= 0:
        s, e = offsets[at], offsets[at + len(want) - 1] + 1
        if s <= start and e >= end:
            return s, e
        at = folded.find(want, at + 1)
    return None


@dataclass
class Checked:
    answer: dict
    valid: bool
    problems: list[str] = field(default_factory=list)
    span: tuple[int, int] | None = None
    follow_up: str = ""  # the question to ask once, empty when none is needed

    @property
    def role(self) -> str:
        return self.answer.get("role", "") if self.valid else ""


def check(answer: dict, sentence: str, start: int, end: int) -> Checked:
    """Is the answer well formed and about the right words? (Not: is it right.)"""
    problems = []
    if not isinstance(answer, dict):
        return Checked({}, False, ["not a JSON object"])
    answer = {k: (v if isinstance(v, list) else str(v).strip() if v is not None else "") for k, v in answer.items()}
    # A model that copies the mark into its answer ("the <w>lingula</w>", 6 Nemotron answers on 13 Oct 2026)
    # has still answered about the right words: the mark is taken out before the phrase is checked.
    for key in ("phrase", "head"):
        if key in answer:
            answer[key] = re.sub(r"\s+", " ", answer[key].replace(OPEN, "").replace(CLOSE, "")).strip()
    raw_roles = answer.get("roles")
    if isinstance(raw_roles, str):
        try:
            raw_roles = json.loads(raw_roles.replace("'", '"'))
        except ValueError:
            raw_roles = [r for r in re.split(r"[,;]", raw_roles.strip("[] ")) if r.strip()]
    roles = [str(r).strip().strip("'\"").lower().replace(" ", "_") for r in (raw_roles or [])] or \
        ([answer.get("role", "").lower().replace(" ", "_")] if answer.get("role") else [])
    roles = list(dict.fromkeys(r for r in roles if r))
    if len(roles) > 1 and "uncertain" in roles:
        roles = [r for r in roles if r != "uncertain"]
    answer["roles"] = roles
    role = roles[0] if roles else ""
    answer["role"] = role
    bad = [r for r in roles if r not in ROLES]
    if not roles or bad:
        problems.append(f"role {', '.join(bad) or repr('')} is not one of {', '.join(ROLES)}")
    kind = answer.get("type", "").lower()
    if kind and kind not in TYPES:
        problems.append(f"type {kind!r} is not one of the types")
    span = covering(sentence, answer.get("phrase", ""), start, end)
    if span is None:
        # de-identified reports replace words with "XXXX" (IU X-ray): a phrase that leaves out such a
        # placeholder at the edge of the marked words still covers the words that are there
        inner = re.match(r"^((?:X{3,}\s*)*)(.*?)((?:\s*X{3,})*)$", sentence[start:end], re.S)
        if inner and inner.group(2) and (inner.group(1) or inner.group(3)):
            s0 = start + len(inner.group(1))
            span = covering(sentence, answer.get("phrase", ""), s0, s0 + len(inner.group(2)))
    if span is None:
        # the same for articles and prepositions at the edges of the marked words
        s0, e0 = start, end
        while True:
            m = re.match(r"\s*([^\W\d_]+['’]?)\s*", sentence[s0:e0])
            if not m or _norm(m.group(1)).rstrip("'") not in _EDGE_WORDS or m.end() >= e0 - s0:
                break
            s0 += m.end()
        if s0 != start:
            span = covering(sentence, answer.get("phrase", ""), s0, e0)
    if span is None:
        problems.append("the phrase is not a piece of the sentence that contains the marked word")
    head = _norm(answer.get("head", ""))
    if span is not None and head and head not in _norm(sentence[span[0]: span[1]]):
        problems.append("the head is not a word of the phrase")
    follow = ""
    if span is not None and span[1] <= end:
        after = next_words(sentence, end)
        if after:
            follow = (
                f"In the sentence, «{sentence[start:end]}» is followed by «{' '.join(after)}». If these words "
                "belong to the same phrase as the marked word (the phrase goes on to its head noun), give the "
                "whole phrase. If they do not (a verb, an adverb, the start of another phrase), give the same "
                "phrase again. Answer with the same JSON."
            )
    return Checked(answer, not problems, problems, span, follow)


def parse(text: str) -> dict:
    found = re.search(r"\{.*\}", text or "", re.S)
    if not found:
        raise ValueError("no JSON object in the answer")
    return json.loads(found.group(0))


@dataclass
class Asked:
    first: Checked
    final: Checked
    asked_again: bool = False
    changed: bool = False
    error: str = ""

    def to_json(self) -> dict:
        return {
            "answer": self.final.answer, "valid": self.final.valid, "problems": self.final.problems,
            "first": self.first.answer if self.asked_again else None, "asked_again": self.asked_again,
            "changed": self.changed, "error": self.error,
        }


def ask(chat: Callable[[str], str], prompt: str, sentence: str, start: int, end: int) -> Asked:
    """One answer, checked; one follow-up when the answer is malformed or stops at the word while the
    sentence may go on. The follow-up answer is final whatever it says (checked again, counted)."""
    try:
        first = check(parse(chat(prompt)), sentence, start, end)
    except Exception as error:  # noqa: BLE001 - a lost answer is counted, not fatal
        first = Checked({}, False, [f"unreadable: {str(error)[:100]}"])
    question = first.follow_up if first.valid else (
        "Your answer could not be used: " + "; ".join(first.problems) + ". Answer again with the same JSON."
    )
    if first.valid and not question:
        return Asked(first, first)
    try:
        previous = json.dumps(first.answer, ensure_ascii=False) if first.answer else "(no JSON)"
        again = check(parse(chat(f"{prompt}\n\nYour answer was: {previous}\n{question}")), sentence, start, end)
    except Exception as error:  # noqa: BLE001
        return Asked(first, first if first.valid else Checked({}, False, first.problems), True, False,
                     str(error)[:100])
    final = again if again.valid else first
    changed = final.valid and first.valid and _norm(final.answer.get("phrase", "")) != _norm(first.answer.get("phrase", ""))
    return Asked(first, final, True, changed or (final is again and not first.valid))


SCHEMA = (
    '{"phrase": "<the whole phrase that contains the marked word, copied exactly from the sentence>", '
    '"head": "<the head noun of that phrase>", "type": "<one of: %s>", '
    '"roles": ["<every role that applies here, one or more of: %s>"]}'
)


def schema() -> str:
    return SCHEMA % (", ".join(TYPES), ", ".join(ROLES))


def roles_text() -> str:
    return "\n".join(f"- {r}: {d}" for r, d in ROLES.items())


RULES = f"""Rules:
1. The tags {OPEN} {CLOSE} only show WHICH word you are asked about. They are never the answer and never
   the limits of the phrase.
2. The phrase is the whole noun phrase the marked word belongs to, from its first modifier to its head
   noun, with the head's complements when they name one concept ("brain weights", "inferior vena cava
   filter placement", "liver gene expression"). The marked word alone is the phrase only when nothing
   in the sentence continues it.
3. A body structure written inside a longer name of another kind of thing (a procedure, a device, a
   measurement, a function or process, a protein or gene, a chemical, an organism, a food) is part of
   that name; the role says what the structure is for that name.
4. Say "uncertain" when the sentence does not let you decide. A wrong certain answer costs more than
   an uncertain one.
5. More than one role can be true at once, and the sentence decides which: a measure of an organ that is
   also a finding in it, the place of a collection that is also where it comes from, a site examined that
   is also where a finding is. Give every role the sentence supports, the most important first."""


def example_lines(examples: Iterable[dict]) -> list[str]:
    """Examples in the same notation as the question: the word marked, the answer after the arrow."""
    out = []
    for i, e in enumerate(examples, 1):
        role = f", role {e['role']}" if e.get("role") else ""
        out.append(f"{i}. {e['marked']} -> phrase «{e['phrase']}», type {e['type']}{role}")
    return out


def balanced(examples: list[dict], word: str, k: int) -> list[dict]:
    """The ``k`` best examples, keeping at least one whose phrase is longer than the word and one whose
    phrase is the word, when both exist (examples are given best first)."""
    if len(examples) <= k:
        return examples
    w = _norm(word)
    longer = [e for e in examples if _norm(e["phrase"]) != w]
    bare = [e for e in examples if _norm(e["phrase"]) == w]
    chosen = list(examples[:k])
    required = [pool[0] for pool in (longer, bare) if pool and not any(e in pool for e in chosen)]
    def is_longer(e: dict) -> bool:
        return _norm(e["phrase"]) != w

    for r in required:
        for i in range(len(chosen) - 1, -1, -1):  # replace the weakest example of a group with more than one
            group = is_longer(chosen[i])
            if sum(is_longer(x) == group for x in chosen) > 1:
                chosen[i] = r
                break
    return chosen


def remark_example(marked_concept: str, word: str) -> str | None:
    """An example sentence where ``[[ ]]`` encloses the annotated concept, rewritten so that only the
    word is marked, as in the question (the concept is then said after the arrow)."""
    m = re.search(r"\[\[(.+?)\]\]", marked_concept)
    if not m:
        return None
    inner = m.group(1)
    w = re.search(re.escape(word), inner, re.I)
    if not w:
        return None
    s = marked_concept[: m.start()] + inner + marked_concept[m.end():]
    at = m.start() + w.start()
    return mark(s, at, at + len(w.group(0)))


def prompt(sentence: str, start: int, end: int, *, knowledge: list[str] | None = None,
           reader: list[str] | None = None, examples: list[dict] | None = None, language: str = "") -> str:
    """The question: rules, roles, optional knowledge (vocabularies), optional evidence of the reader,
    optional examples, the sentence with the word marked, the JSON schema."""
    word = sentence[start:end]
    parts = ["You read sentences of clinical and biomedical texts the way an expert annotator does."
             + (f" The sentence is in {language}; copy the phrase in that language." if language else ""),
             RULES, "Roles of the marked word:\n" + roles_text()]
    if knowledge:
        parts.append(f"What the vocabularies know about the words around «{word}» (every reading they have, "
                     "the anatomical one included; knowledge, not a hint):\n" + "\n".join(knowledge))
    if reader:
        parts.append("What a reading model built for this word (its readings with their share of activation "
                     "after integration, and the evidence it used; it can be wrong, check it against the "
                     "sentence):\n" + "\n".join(reader))
    if examples:
        parts.append("Annotated sentences of other documents (same notation):\n" + "\n".join(example_lines(examples)))
    parts.append(f"Sentence: {mark(sentence, start, end)}")
    parts.append("Answer with JSON only, no other text:\n" + schema())
    return "\n\n".join(parts)


ROLE_SCHEMA = '{"role": "<one of: %s>"}'


def role_prompt(sentence: str, start: int, end: int, *, reader: list[str] | None = None, language: str = "") -> str:
    """The question of the fine-tuned model (point 2): the role only. The gold set gives the role of every
    item but not always the whole phrase; teaching a phrase it does not have would teach the bare word."""
    full = prompt(sentence, start, end, reader=reader, language=language)
    head, _, _ = full.rpartition("Answer with JSON only")
    return head + "Answer with JSON only, no other text:\n" + ROLE_SCHEMA % ", ".join(ROLES)


def read_role(text: str) -> str:
    """The role in an answer of the role-only question; ``uncertain`` when unreadable or not in the schema."""
    found = re.findall(r"\{[^{}]*\}", text or "")  # the last object: reasoning models think first
    try:
        role = str(json.loads(found[-1]).get("role", "")).strip().lower().replace(" ", "_")
    except Exception:  # noqa: BLE001
        return "uncertain"
    return role if role in ROLES else "uncertain"


def kind_of_type(kind: str) -> str:
    """The NCIt-like kind of an answer type (``phrase_probe`` vocabulary)."""
    return dict(zip(TYPES, (
        "anatomy", "disease", "procedure", "device", "food", "protein", "chemical", "organism",
        "process", "property", "conceptual",
    ), strict=True)).get(kind.lower(), "conceptual")


__all__ = [
    "CLOSE", "OPEN", "ROLES", "SITE_ROLES", "TYPES", "Asked", "Checked", "ask", "balanced", "check",
    "covering", "kind_of_type", "mark", "next_words", "parse", "prompt", "remark_example", "schema",
]
