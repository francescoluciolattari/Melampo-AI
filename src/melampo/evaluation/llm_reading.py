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
    "landmark": "the structure only places something else: what the sentence describes is a node, a fluid, a mass, a device or another organ, which is not this structure nor a part of it, and the structure says where it is (nodes along the aortic arch, fluid between the bowel loops, the appendix lateral to the cecum, the prostate protruding into the bladder, ureters dilated up to the bladder, the parietal pleura at the level of a lung lobe); a finding IN or OF the structure (a nodule in the lobe, a granuloma in the lower lobe, calcification of the arch) is structure, not landmark",
    "procedure_site": "the structure is the site of a procedure or exam named in the phrase or by the word that governs it, also outside the phrase (liver biopsy, CT of the chest, liberation of the caecum and of the appendix)",
    "device_site": "the structure is where a device is or goes (vena cava filter, biliary stent)",
    "inherent_location": "the phrase names a measure, property, function or process of the structure (brain weight, liver function)",
    "source_of": "the structure is where a substance, cell or sample comes from (liver enzymes, bone marrow cells)",
    "inside_a_name": "the word is part of the name of something that is not in the body: a gene, protein, chemical, drug, organism, organisation, product (Heart and Soul gene, American Heart Association); the word does not refer to the body here",
    "not_a_body_site": "the word does not name a place in a body here (heart of the matter, a cheese)",
    "other": "none of the above",
    "uncertain": "the sentence does not let you decide",
}
SITE_ROLES = frozenset({"structure", "landmark", "procedure_site", "device_site", "inherent_location", "source_of"})
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
    answer = {k: (v if isinstance(v, (list, dict)) else str(v).strip() if v is not None else "") for k, v in answer.items()}
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
3. A body structure written inside a longer name of another kind of thing is part of that name; the role
   says what the structure is for that name. A disease, finding or lesion named after where it is (pleural
   effusion, pulmonary edema, osteosarcoma of the femur, macular pucker) still refers to that structure:
   the structure is where the finding is. So does a procedure, device or measure named after it (liver
   biopsy, biliary stent, heart rate). Only the name of something that is not in the body (a gene, a
   protein, a chemical, an organism, an organisation, a product) makes the word "inside_a_name".
4. Say "uncertain" when the sentence does not let you decide. A wrong certain answer costs more than
   an uncertain one.
5. More than one role can be true at once, and the sentence decides which: a measure of an organ that is
   also a finding in it, the place of a collection that is also where it comes from, a site examined that
   is also where a finding is, a part of an organ that places a finding in that organ. Give every role the
   sentence supports, the most important first."""


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


# -- the reader's method, asked of the model (14 October 2026) ------------------------------------------
# The Kintsch reader builds the possible readings, gathers the evidence for each and integrates them. The
# model is asked to do the same with its own knowledge of the language: the method, never the reader's
# answers (a model shown the reader's answers copies its errors, 13 October 2026), so that the two readers
# stay independent and their agreement keeps its value.
METHOD_STEPS = """Read the way the reading model of this project does, step by step, and write each step down:
1. The phrase that contains the marked word, and its head noun.
2. The governor: the word the phrase depends on, also outside the phrase (across "and", after "of"/"di",
   the noun or verb that governs it), and what that word names: an exam, a procedure, a device, a finding or
   disease, a measure or property, or something else. Empty when nothing governs it.
3. The predicate: what the sentence says about the structure. If it says nothing about it and uses it only to
   place something else (a node, a fluid, a mass, another organ), say what that thing is: the structure is
   then a landmark.
4. The readings: every role that is possible here, each with the evidence for it and against it, taken
   from steps 1-3 and the sentence.
5. The roles: those the evidence supports, the most important first (one or more)."""

METHOD_SCHEMA = (
    '{"phrase": "<copied exactly from the sentence>", "head": "<head noun>", "type": "<one of: %s>", '
    '"governor": {"word": "<the governing word, or empty>", "names": "<exam, procedure, device, finding or disease, '
    'measure or property, other, or empty>"}, "predicate": "<what the sentence says about the structure>", '
    '"placed": "<what the structure places, when it is only a landmark; else empty>", '
    '"readings": [{"role": "<one of the roles>", "for": "<evidence for>", "against": "<evidence against>"}], '
    '"roles": ["<the roles the evidence supports, one or more of: %s>"]}'
)


def method_prompt(sentence: str, start: int, end: int, *, language: str = "") -> str:
    """The question with the reader's method (steps written down), no reader output."""
    full = prompt(sentence, start, end, language=language)
    head, _, _ = full.rpartition("Sentence: ")
    return (head + METHOD_STEPS + "\n\nSentence: " + mark(sentence, start, end) + "\n\nAnswer with JSON only, "
            "no other text:\n" + METHOD_SCHEMA % (", ".join(TYPES), ", ".join(ROLES)))


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


# -- of what is the finding said: the question asked from the finding (1 November 2026) -----------------------
# The radiologist starts from what is abnormal and asks what it belongs to; the model was asked the role of the
# structure, and derived from two roles ("structure" and "landmark") an outcome it was never asked (owner probe,
# 31 October 2026: 11 of 26 findings of another structure kept). Here the outcome is asked, from the finding, and
# the model is told which structure the words were linked to, so that it can say when the link is wrong.
OWNER_RELATIONS = ("same", "part", "reference", "wrong", "hypothesis", "uncertain")
OWNER_STEPS = """Read it the way a radiologist does, in this order, and write each step:
1. finding: what the sentence describes, measures or states as abnormal, with its head noun ("dilated
   ureters", "hypertrophy of the transitional zone", "enlarged nodes", "a foreign body", "a cyst"); "none" if it
   describes nothing.
2. bearer: the body structure that finding is of or in, the one a radiologist would code it to, from the
   words of the sentence, the exam and your knowledge of anatomy; "unknown" if the sentence does not say.
3. linked: how the linked structure stands to the bearer:
   - "same": it is the bearer;
   - "part": it is a part of the bearer, or the bearer is a part of it (a lobe of the lung, the wall of the
     bladder, a segment of the liver);
   - "reference": it only says where the finding is, which belongs to another structure (nodes along the aorta,
     ureters dilated up to the bladder, the prostate protruding into the bladder, the parietal pleura at the
     level of a lung lobe, fluid around the spleen);
   - "wrong": the marked words name another structure here, not the linked one (in a prostate MRI the "middle
     lobe" is the prostate's, not the lung's);
   - "hypothesis": it is named only as a possible explanation (a differential diagnosis, "suggestive of");
   - "uncertain": you cannot tell.
4. answer: "yes" if the finding is of the linked structure (same or part), "no" if it is not (reference, wrong,
   hypothesis), "uncertain" otherwise."""
OWNER_SCHEMA = ('{"finding": "<what is described>", "bearer": "<the structure it is of>", '
                '"linked": "<one of: %s>", "answer": "<yes, no or uncertain>", "why": "<one sentence>"}')


def owner_prompt(sentence: str, start: int, end: int, linked: str, *, exam: str = "", section: str = "",
                 language: str = "") -> str:
    """The question "is the finding of the linked structure?", asked from the finding."""
    parts = ["You read sentences of radiology reports the way a radiologist does."]
    if exam:
        parts.append(f"Exam: {exam}")
    if section:
        parts.append(f"Section of the report: {section}")
    parts.append(f"Sentence{f' ({language})' if language else ''}: {mark(sentence, start, end)}")
    parts.append(f"An automatic reader linked the marked words to this body structure: {linked}.")
    parts.append(OWNER_STEPS)
    parts.append("Answer with JSON only, no other text:\n" + OWNER_SCHEMA % ", ".join(OWNER_RELATIONS))
    return "\n\n".join(parts)


def read_owner(text: str) -> dict:
    """The answer of ``owner_prompt``: the last JSON object; the answer folded to yes, no or uncertain."""
    found = re.findall(r"\{[^{}]*\}", text or "")
    try:
        answer = json.loads(found[-1])
    except Exception:  # noqa: BLE001
        return {"answer": "uncertain", "linked": "uncertain", "valid": False}
    said = str(answer.get("answer", "")).strip().lower()
    linked = str(answer.get("linked", "")).strip().lower()
    answer["answer"] = said if said in ("yes", "no") else "uncertain"
    answer["linked"] = linked if linked in OWNER_RELATIONS else "uncertain"
    if answer["answer"] == "yes" and answer["linked"] in ("reference", "wrong", "hypothesis"):
        answer["answer"] = "uncertain"      # the two steps contradict each other
    if answer["answer"] == "no" and answer["linked"] in ("same", "part"):
        answer["answer"] = "uncertain"
    answer["valid"] = True
    return answer


# -- what the sentence says of the structure (1 November 2026) ------------------------------------------------
# The question "is the finding of this structure?" makes the model pick the main finding of the sentence (the
# slipped L3, the spinal pain, the longer left kidney) and lose what the sentence says of the linked structure
# (L4 is the level below, the thyroid was operated, the right kidney is 10.5 cm). A radiologist asks, of every
# structure named, what the sentence says of it. The model copies that piece from the sentence and says its
# type; the reader checks that the piece is in the sentence and that the type agrees with what it knows.
STATEMENT_TYPES = ("finding", "involved", "history", "normal", "measure", "reference", "other_structure",
                   "hypothesis", "nothing", "uncertain")
STATEMENT_STEPS = """Read it the way a radiologist does: what does the sentence say of the linked structure?
1. piece: copy from the sentence, exactly as written, the words that say it (the fewest that carry the
   statement; include the structure's own words when they are part of it). "" if it says nothing of it.
2. type, the first that applies in this order:
   - "finding": something abnormal of it or in it, or of one of its parts (a lesion in it, a thickened wall,
     an enlarged organ); a finding stated as probable of it ("a suspicious lesion of it", "probably a cyst of
     it") is still a finding;
   - "involved": the abnormality is of another structure but reaches it: extends to it or into it, presses on
     it, narrows it, displaces it, or is at the level or the joint it forms (a disc bulge at the level two
     vertebrae form, a tumour of a neighbour invading it);
   - "history": a fact of its past: a past disease or a surgery ("status post resection of it", "history of
     surgery of it");
   - "normal": it is normal, or something abnormal is said to be absent from it ("no lesions", "not enlarged"),
     also when measures go with it;
   - "measure": only a measure of it, with no judgement ("it measures 11 cm");
   - "reference": it only says where something of another structure is (enlarged nodes along the aortic arch,
     fluid around an organ): the finding is of that other structure;
   - "other_structure": the marked words name a different structure from the linked one here (in a prostate
     MRI the "middle lobe" is the prostate's, not the lung's);
   - "hypothesis": it is named only as one possible explanation, or as something to be excluded ("differential
     diagnosis: ...", "possibly arising from it");
   - "nothing": it is named without anything said of it: a title, a heading, the name of the exam, an earlier
     exam it is compared with;
   - "uncertain": you cannot tell.
3. of: for "reference" and "other_structure", the structure the finding is of, or the one the words name."""
STATEMENT_SCHEMA = ('{"piece": "<copied exactly from the sentence>", "type": "<one of: %s>", '
                    '"of": "<the other structure, or empty>", "why": "<one sentence>"}')


def statement_prompt(sentence: str, start: int, end: int, linked: str, *, linked_names: list[str] | None = None,
                     exam: str = "", section: str = "", language: str = "") -> str:
    """The question "what does the sentence say of the linked structure?". ``linked``: the class as the linker
    means it (its usual name); ``linked_names``: its other names, so that the model knows what the class is
    (the left hip of the atlas is the hip bone, not the joint)."""
    parts = ["You read sentences of radiology reports the way a radiologist does."]
    if exam:
        parts.append(f"Exam: {exam}")
    if section:
        parts.append(f"Section of the report: {section}")
    parts.append(f"Sentence{f' ({language})' if language else ''}: {mark(sentence, start, end)}")
    names = [n for n in (linked_names or []) if n and n != linked]
    also = f" (also called: {', '.join(names[:4])})" if names else ""
    parts.append(f"An automatic reader linked the marked words to this structure of an anatomical atlas: "
                 f"{linked}{also}. Read the sentence about that structure as the atlas means it.")
    parts.append(STATEMENT_STEPS)
    parts.append("Answer with JSON only, no other text:\n" + STATEMENT_SCHEMA % ", ".join(STATEMENT_TYPES))
    return "\n\n".join(parts)


def in_sentence(sentence: str, piece: str) -> bool:
    """Is the piece in the sentence, as written (spaces, case and the mark folded)? A piece that is not there
    was not read from the sentence."""
    want = _norm(piece.replace(OPEN, "").replace(CLOSE, ""))
    return bool(want) and want in _norm(sentence)


def read_statement(text: str, sentence: str) -> dict:
    """The answer of ``statement_prompt``: the last JSON object, its type one of ``STATEMENT_TYPES``, and whether
    its piece is in the sentence (``found``). A type that needs a piece without one found is ``uncertain``."""
    found = re.findall(r"\{[^{}]*\}", text or "")
    try:
        answer = json.loads(found[-1])
    except Exception:  # noqa: BLE001
        return {"type": "uncertain", "piece": "", "of": "", "found": False, "valid": False}
    kind = str(answer.get("type", "")).strip().lower().replace(" ", "_")
    answer["type"] = kind if kind in STATEMENT_TYPES else "uncertain"
    answer["piece"] = str(answer.get("piece") or "").strip()
    answer["of"] = str(answer.get("of") or "").strip()
    answer["found"] = in_sentence(sentence, answer["piece"])
    if answer["type"] not in ("nothing", "uncertain") and not answer["found"]:
        answer["type"] = "uncertain"      # the piece was not read from the sentence
    answer["valid"] = True
    return answer


# -- what the sentence says of the structure, claim by claim (2 November 2026) --------------------------------
# One label from an ordered list made the models disagree wherever a sentence says two things ("hyperechoic and
# not enlarged") or something between two labels ("likely calcified cartilages"): the boundaries were set in the
# question, before reading. Here nothing is chosen: the model says what the sentence states of the structure,
# claim by claim, each with the words copied from the sentence, and the interpreter attaches every claim the
# readers agree on. What is not in the words is not a claim ("said, not inferred").
# 10 November 2026: one category of uncertainty instead of two ("certainty", "alternatives"). On ctrate3 the
# report's doubt was lost in 11 of 43 items: a sentence such as "a hypodense lesion (cyst?)" says two things, a
# lesion that is stated and that it is a cyst, which is only a hypothesis; the question asked whether "what is
# observed" is only probable, and the lesion is not, so a model reading well left it empty; one hypothesis fitted
# neither list, so the models put it in different lists or in none. Now what is seen and what the sentence says it
# is are apart, and every uncertainty is said with the words it applies to. Also: what is said of another person,
# and what the patient reports, which are not the exam's observation.
# 11 November 2026 (2): on ctrate5 the uncertainty was given to 14 of 293 stated findings, 13 by one model: what
# is seen but not identified and the limits of the exam put also in the uncertainty, "stable" read as probable,
# "ground-glass-like" as a doubt, a source of knowledge as a level. The literature on hedging tells them apart:
# approximators change what is said, shields the commitment to it (Prince, Frader and Bosk 1982; Hyland 1996:
# attribute and reliability hedges); attribution shields name a source; speculation has a cue and a scope
# (BioScope, Vincze et al. 2008); factuality is the commitment of a source to an event (FactBank, Sauri and
# Pustejovsky); time and experiencer are apart (ConText, Harkema et al. 2009).
# 11 November 2026: on ctrate4 the uncertainty was given to 62 of 250 findings the report states (24.8%): what is
# seen but not identified, the limits of the exam, an attribution, the degree of what is seen had no place of
# their own, and "when in doubt, it is uncertain" pushed them into the uncertainty. Now each has its place, and
# every uncertainty has its level on a published scale (Panicek and Hricak, AJR 2016).
CLAIM_LISTS = ("observed", "normal", "measure", "past", "uncertainty", "device", "not_assessable",
               "indication", "procedure", "other_person", "reported", "undetermined", "limits")
UNCERTAINTY_LEVELS = ("high", "probable", "possible", "less_likely", "unlikely")
# the lists of the questions before 10 November 2026, read into "uncertainty" (stored answers, replays)
_FORMER_UNCERTAINTY = ("certainty", "alternatives")
CLAIM_STEPS = """Say what the sentence states of the linked structure, using only its words: every entry is copied
exactly from the sentence (you may shorten a long piece with "..." between copied parts). Do not add what the
sentence does not say: a finding it does not state is not there. What you know of the structure serves only to
compare what the sentence says with the usual (observed, below). Leave a list empty when the sentence says nothing
of that kind.
- marked: the marked words, copied.
- same: about the marked words only, not about what the sentence describes: "yes" if here they name the linked
  structure or a part of it, "no" if they name a different structure (then named: the structure they name),
  "unsure".
- observed: everything the sentence says of an aspect of it or of what it contains (its state of filling or
  distension, its content, its wall, size, shape, density, echogenicity, signal, position, a lesion), each compared
  as a radiologist does with how that aspect of this structure is in a healthy person in this exam. A number is
  not compared here: a measure goes in measure, where it is compared with the reference values. For each:
  words; aspect: the aspect, in a few words; usual: how that aspect is in a healthy person in this exam, in a few
  words of your own (a physiological state, such as a hollow organ more or less filled, is usual); differs: "yes"
  if the words describe something different from the usual, even if mild, "no" if they describe the usual,
  "unsure" if it cannot be told; said_to_be: what the sentence says it is or what it is due to ("cyst" in "a
  hypodense lesion (cyst?)", "infectious process" in "densities evaluated in favor of an infectious process"),
  copied, or "": these words are not part of words, which hold only what is seen; of: "it" if it is of the linked
  structure or one of its parts, otherwise the
  structure it is of, copied from the sentence. Words that only say where something is ("at the level of", "at
  the base of", "in the topography of", "in the region of", "in the projection of", "projected on", "near",
  "beside") place it and do not make it the structure's own, unless the sentence says that it is in it or of it
  or that the structure itself is touched; when the sentence says what it is ("a stone in the right kidney", "may
  correspond to a lymph node", "pneumothorax?"), it is of the structure that is; what the sentence says of
  another subject ("the celiac trunk is patent, with a narrowing at its origin from the aorta") is of that
  subject. A variant of the structure (an accessory spleen, a lobe of Riedel) is of it; relation: the words that relate it to the linked structure ("in",
  "of", "between", "adjacent to", "extends into", "compresses"), copied, or ""; reaches: for what is of another
  structure, "yes" if the sentence says that the linked structure itself is touched by it (it extends into it,
  infiltrates, compresses, narrows, displaces or deforms it) or that it is of a joint or a space the linked
  structure forms with another (a disc between two vertebrae); "no" if the sentence says only where it is, how
  high or how far it goes, without the linked structure being touched; refers_to: when the words say "its",
  "it", "they", "the same", "this" (or "sua", "esso", "stesso"), what they refer to, copied from the sentence or
  from the previous sentence, else "".
- normal: words saying that it, or something of it, is normal or that something is absent from it.
- measure: every measure of it, of the structure itself: the size of a lesion in it belongs to that lesion
  (in observed) and is never compared with the structure's reference. For each: words; reference: the id of the reference value that applies to
  this measure (the same structure, aspect, technique and population), or "" if none applies or none is given;
  against: "below", "within" or "above" the normal range of that reference, or "" without a reference. A measure
  is never compared with a value you remember.
- past: words of a fact of its past (a past disease, a surgery).
- uncertainty: only how far the writer commits to the truth of something the sentence says (a hedge that shields
  it, Prince, Frader and Bosk 1982; a speculation, BioScope 2008): a doubt, a hypothesis, alternatives joined by
  "or", a differential, something probable, possible, suspected, questioned ("?"), "suggests", "may", "might",
  "in favor of" a hypothesis, "thought to be", "considered", "can be seen in", "cannot be excluded", "to be
  evaluated for", a condition ("if there is"), and also "consistent with", "compatible with", "in line with".
  For each: words: the words that make it uncertain, copied; about: the words of what is uncertain, copied (often
  what the sentence says it is, while what is seen is certain: in "a hypodense lesion (cyst?)" the lesion is
  seen, the cyst is uncertain); level: how likely the writer says it is, on the scale of Panicek and Hricak (AJR
  2016): "high" (more than 90%, as "consistent with"), "probable" (about 75%, as "suspicious for",
  "probably"), "possible" (about 50%, as "possibly", "may", "?"), "less_likely" (about 25%), "unlikely" (less
  than 10%), read from the whole sentence ("may be consistent with" is not "consistent with"; "suspicious ...
  consistent with" is not "consistent with"). A question mark applies to what it follows or closes, a
  parenthesis with "?" to the words in it. What a word does in this sentence decides, not the word. These are not
  uncertainty, and none of their words goes in uncertainty: a word stays in one place only.
  * what is seen but not identified ("nonspecific", "uncharacterized", "cannot be characterized"): its nature is
    open, it is not doubted; it goes in undetermined;
  * how far the exam could see ("as far as can be observed", "within the limits of non-contrast CT"): it goes in
    limits;
  * an approximation of what is seen, its degree, extent or likeness ("slight", "faint", "approximately", "about",
    "ground-glass-like", "tends to"): it changes what is said, not the commitment to it; it stays in what is seen;
  * where the knowledge comes from ("when evaluated together with US", "according to", "it was understood
    that"): a source, not a doubt; what the writer then states is certain;
  * a change in time ("stable", "unchanged", "new", "decreased", "regressed"): a statement about time, certain;
  * what something is attributed to ("increased cardiothoracic ratio in favor of the heart", "secondary to",
    "due to"): stated, certain.
- undetermined: words saying that what is seen is not identified or not characterised ("nonspecific", "cannot be
  characterized"): what is seen stays certain, only its nature is open.
- limits: words saying how far the exam could see, or what limited it, copied. A limit qualifies what the exam
  could not see (most often an absence: "no lesion, as far as can be observed"); what was seen stays certain.
- other_person: words of what is said of a person other than the patient (a relative's disease).
- reported: words of what the patient reports or the referral says, not observed in the exam.
- device: words naming a device in or at it (a tube, a catheter, a stent, a prosthesis, an implanted valve),
  even when the sentence says that its shadow or image is seen: a device is not an observation of it.
- not_assessable: words saying it cannot be assessed.
- indication: words saying why the exam was asked (the referral, the clinical question, a suspicion to verify):
  in a section of indication everything the sentence says goes here, and nothing in observed. The question of
  the referral is not a finding of the exam.
- procedure: words naming a procedure done on it, now or before (a biopsy, an embolization, a resection).
- conclusion: what the report's conclusion, when given, says of what is observed here: "same" if it repeats it
  of this structure, "other" if it gives it to another structure (then conclusion_of: that structure, copied
  from the conclusion), "none" if it does not mention it or no conclusion is given."""
CLAIM_SCHEMA = ('{"marked": "...", "same": "yes|no|unsure", "named": "", '
                '"observed": [{"words": "...", "aspect": "...", "usual": "...", "differs": "yes|no|unsure", '
                '"said_to_be": "", "of": "it", "relation": "", "reaches": "no", "refers_to": ""}], "normal": [], '
                '"measure": [{"words": "...", "reference": "", "against": ""}], "past": [], '
                '"uncertainty": [{"words": "...", "about": "...", "level": "high|probable|possible|less_likely|unlikely"}], '
                '"device": [], "not_assessable": [], "indication": [], "procedure": [], "other_person": [], '
                '"reported": [], "undetermined": [], "limits": [], '
                '"conclusion": "same|other|none", "conclusion_of": "", "why": "<one sentence>"}')
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")


def claims_prompt(sentence: str, start: int, end: int, linked: str, *, linked_names: list[str] | None = None,
                  exam: str = "", section: str = "", language: str = "", references: list[str] | None = None,
                  passages: list[str] | None = None, section_kind: str = "", conclusion: str = "",
                  previous: str = "", manual: str = "", examples: list[str] | None = None) -> str:
    """The question "what does the sentence state of the linked structure?", claim by claim. ``references``:
    the reference values that apply (``reference_values.Row.line``); ``passages``: paragraphs of their sources;
    ``manual``: what the sources say of reading a sentence (``reading_memory``, 11 November 2026); ``examples``:
    the most similar sentences already read and verified."""
    parts = ["You read sentences of radiology reports the way a radiologist does."]
    if exam:
        parts.append(f"Exam (context only: the sentence prevails): {exam}")
    if section or section_kind:
        kind = f" (a section of {section_kind})" if section_kind else ""
        parts.append(f"Section of the report: {section or section_kind}{kind}")
    if previous:
        parts.append(f"Previous sentence (context only): {previous}")
    parts.append(f"Sentence{f' ({language})' if language else ''}: {mark(sentence, start, end)}")
    if conclusion:
        parts.append(f"Conclusion of the report (context only): {conclusion}")
    names = [n for n in (linked_names or []) if n and n != linked]
    also = f" (also called: {', '.join(names[:4])})" if names else ""
    parts.append(f"An automatic reader linked the marked words to this structure of an anatomical atlas: "
                 f"{linked}{also}.")
    if references:
        parts.append("Reference values (compare a measure only with these, and cite the id):\n"
                     + "\n".join(references))
    else:
        parts.append("Reference values: none for this structure and exam (a measure is only a measure).")
    if passages:
        parts.append("Paragraphs of the sources of these values (for reading them, not to be copied):\n"
                     + "\n".join(passages))
    if manual:
        parts.append("Reading manual (what the sources say of reading what a sentence states; it guides the "
                     "steps below):\n" + manual.strip())
    if examples:
        parts.append("Sentences already read and verified, the most like this one (how to read, not what to copy: "
                     "this sentence decides):\n" + "\n".join(examples))
    parts.append(CLAIM_STEPS)
    parts.append("Answer with JSON only, no other text:\n" + CLAIM_SCHEMA)
    return "\n\n".join(parts)


def copied(sentence: str, words: str) -> bool:
    """Are the words copied from the sentence? A piece shortened with "..." is copied when every part is in the
    sentence, in order."""
    words = str(words or "").replace(OPEN, "").replace(CLOSE, "")
    parts = [p for p in _ELLIPSIS.split(words) if p.strip()]
    if not parts:
        return False
    text, at = _norm(sentence), 0
    for part in parts:
        found = text.find(_norm(part), at)
        if found < 0:
            return False
        at = found + len(_norm(part))
    return True


def read_claims(text: str, sentence: str, mention: str, previous: str = "", conclusion: str = "") -> dict:
    """The answer of ``claims_prompt``, checked against the sentence: entries whose words are not in it are
    dropped (``dropped`` counts them); an answer about other words than the marked ones is not valid. What a
    pronoun refers to must be copied from the sentence or the ``previous`` one, what the ``conclusion`` gives the
    finding to from the conclusion; otherwise it is left empty."""
    found = re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", text or "", re.S)
    try:
        answer = json.loads(found[-1])
    except Exception:  # noqa: BLE001
        return {"valid": False, "same": "unsure", "dropped": 0, **{k: [] for k in CLAIM_LISTS}}
    out: dict = {"valid": True, "dropped": 0, "why": str(answer.get("why") or "")}
    marked = str(answer.get("marked") or "").replace(OPEN, "").replace(CLOSE, "")
    out["marked"] = marked
    if marked and _norm(marked) != _norm(mention) and _norm(mention) not in _norm(marked):
        out["valid"] = False            # it read other words than the marked ones
    same = str(answer.get("same") or "").strip().lower()
    out["same"] = same if same in ("yes", "no", "unsure") else "unsure"
    out["named"] = str(answer.get("named") or "")
    observed, usual = [], []
    for entry in answer.get("observed") or []:
        if not isinstance(entry, dict) or not copied(sentence, entry.get("words", "")):
            out["dropped"] += 1
            continue
        of = str(entry.get("of") or "it").strip()
        if of.lower() not in ("it", "") and not copied(sentence, of):
            of = "?"                    # of another structure, not named in the sentence: unknown
        relation = str(entry.get("relation") or "")
        reaches = str(entry.get("reaches") or "").strip().lower()
        differs = str(entry.get("differs") or "yes").strip().lower()
        if differs == "no":
            usual.append(str(entry["words"]))   # compared with the usual and found the same: a normal claim
            continue
        if differs != "yes":
            out["state_unsure"] = True          # the model cannot tell it from the usual: a person decides
        observed.append({"words": str(entry["words"]), "of": "it" if of.lower() in ("it", "") else of,
                         "relation": relation if copied(sentence, relation) else "",
                         "reaches": reaches if reaches in ("yes", "no") else "no",
                         "aspect": str(entry.get("aspect") or ""), "usual": str(entry.get("usual") or ""),
                         "said_to_be": _copied_from(entry.get("said_to_be"), sentence),
                         "refers_to": _copied_from(entry.get("refers_to"), sentence, previous)})
    out["observed"] = observed
    out["measure_refs"] = []
    out["uncertain_about"] = []
    for key in CLAIM_LISTS[1:]:
        kept = []
        entries = list(answer.get(key) or [])
        if key == "uncertainty":
            entries += [e for k in _FORMER_UNCERTAINTY for e in (answer.get(k) or [])]
        for entry in entries:
            words = entry.get("words", "") if isinstance(entry, dict) else str(entry)
            if copied(sentence, words):
                if words in kept:
                    continue
                kept.append(words)
                if key == "uncertainty" and isinstance(entry, dict):
                    level = str(entry.get("level") or "").strip().lower()
                    out["uncertain_about"].append({"words": words, "about": _copied_from(entry.get("about"),
                                                                                         sentence),
                                                   # an uncertainty without a readable level is "possible": it stays
                                                   # uncertain, never certain
                                                   "level": level if level in UNCERTAINTY_LEVELS else "possible"})
                if key == "measure" and isinstance(entry, dict) and str(entry.get("reference") or "").strip():
                    out["measure_refs"].append({"words": words, "reference": str(entry["reference"]).strip(),
                                                "against": str(entry.get("against") or "").strip().lower()})
            else:
                out["dropped"] += 1
        out[key] = kept
    out["normal"] = usual + [w for w in out["normal"] if w not in usual]
    # non-contradiction (11 November 2026): words the answer itself puts where the nature is open or the exam's
    # limit is are not, in the same answer, an uncertainty; the place it chose for them stands (ctrate5: 10 of the
    # 13 uncertainties one model gave to stated findings were such words, put in both places)
    elsewhere = [w.lower() for k in ("undetermined", "limits") for w in out.get(k, [])]
    if elsewhere and out.get("uncertainty"):
        def tokens(text: str) -> str:
            return " ".join(re.findall(r"\w+", text.lower()))

        def placed(words: str) -> bool:
            # the same words, whole words only: "or" is not in "for"
            w = tokens(words)
            for e in map(tokens, elsewhere):
                if w and e and (re.search(r"\b" + re.escape(w) + r"\b", e) or re.search(r"\b" + re.escape(e) + r"\b", w)):
                    return True
            return False

        dropped = [w for w in out["uncertainty"] if placed(w)]
        if dropped:
            out["uncertainty"] = [w for w in out["uncertainty"] if w not in dropped]
            out["uncertain_about"] = [u for u in out["uncertain_about"] if u["words"] not in dropped]
            out["contradictions"] = dropped
    said = str(answer.get("conclusion") or "none").strip().lower()
    out["conclusion"] = said if said in ("same", "other", "none") else "none"
    out["conclusion_of"] = _copied_from(answer.get("conclusion_of"), conclusion)
    if out["conclusion"] == "other" and not out["conclusion_of"]:
        out["conclusion"] = "none"      # another structure the conclusion does not name is not read
    return out


def _copied_from(words, *texts: str) -> str:
    words = str(words or "").strip()
    return words if words and any(t and copied(t, words) for t in texts) else ""


__all__ = [
    "CLAIM_LISTS", "claims_prompt", "copied", "read_claims",
    "STATEMENT_TYPES", "in_sentence", "read_statement", "statement_prompt",
    "OWNER_RELATIONS", "owner_prompt", "read_owner",
    "CLOSE", "METHOD_SCHEMA", "METHOD_STEPS", "OPEN", "ROLES", "method_prompt", "SITE_ROLES", "TYPES", "Asked", "Checked", "ask", "balanced", "check",
    "covering", "kind_of_type", "mark", "next_words", "parse", "prompt", "remark_example", "schema",
]
