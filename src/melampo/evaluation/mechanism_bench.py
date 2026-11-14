"""The vetting bench, version 3 (14 November 2026): a mechanism judged by the medical sources and by a reading
that understands them, and a link stated with how certain it is.

Version 2 judged a candidate's mechanism against a curated graph of thirty edges, one name per case. Real,
correct answers failed it: "iron overload causing hepatic fibrosis" against "hepatic iron deposition", "light
chain cast nephropathy" against "Bence Jones proteinuria", "reduced LDL receptor expression" against "reduced LDL
receptor activity"; and its restraint cases asked candidates to deny links medicine documents (a monoclonal
protein interfering with thyroid assays). The bench measured agreement with its own vocabulary, as it said.

Three changes, asked for by the owner:

1. **The judge is every source loaded, read with understanding.** Each case lists the mechanisms medicine accepts
   for it -- more than one where more than one is established -- each a concept with a short description, and
   each looked up in every ontology given (HPO, GO, MONDO, ChEBI...: its identifier, definition, synonyms and the
   terms below it). A claim is right, deterministically, when it names one of them or a synonym; otherwise a
   judge model reads it the way a clinician would, aimed at the sources: it is shown the accepted concepts with
   their ontology entries and the ontology terms closest to the claim's words, says which term the claim means
   and whether it is the same mechanism, a more specific one, a step of the same chain, or a different one.
   When the judge maps the claim to a term the ontology places under an accepted one, the ontology decides.
   Two judges of families other than the candidates' must agree; a disagreement is a doubt, reported apart,
   never counted right or wrong.
2. **The link is stated with its certainty** -- ``known`` (established, documented), ``possible`` (rare,
   debated, conjectured) or ``none`` -- so a prudent answer is not counted as an invention. With it the
   restraint cases need no fixture of their own beyond the cases without an established link: there, ``none``
   and ``possible`` are right, and ``known`` is an error unless both judges find the link established (then the
   case, not the candidate, is wrong, and the result says so).
3. **Errors are the confident ones**: a ``known`` link with a wrong mechanism, or a ``known`` link where none is
   established. A right mechanism stated as only ``possible`` is right but hedged; ``none`` on an established
   link is a miss. Each is counted apart.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..memory.concept_resolution import SCOPE_EXACT, OntologyTerm, TermIndex, normalise_surface, parse_obo

LINK_KNOWN, LINK_POSSIBLE, LINK_NONE = "known", "possible", "none"
LINKS = (LINK_KNOWN, LINK_POSSIBLE, LINK_NONE)
RIGHT_RELATIONS = ("same", "more_specific", "step_of_chain")

# outcomes
RIGHT = "right"                      # known, and the mechanism is an accepted one (or none/possible where no link)
RIGHT_HEDGED = "right_hedged"        # the right mechanism, stated only as possible
MISS = "miss"                        # none, where a link is established
WRONG_CONFIDENT = "wrong_confident"  # known, with a mechanism that is not an accepted one: an error
WRONG_HEDGED = "wrong_hedged"        # possible, with a wrong mechanism
INVENTED = "invented"                # known, where no link is established and the judges find none: an error
CASE_HAS_A_LINK = "case_has_a_link"  # known, where the case says none, and both judges find the link established
DOUBT = "doubt"                      # the judges disagree, or a judge did not answer
MALFORMED = "malformed"
ERRORS = (WRONG_CONFIDENT, INVENTED)


@dataclass(frozen=True)
class Accepted:
    """A mechanism medicine accepts for a case: a concept, how it is described, and the ontology labels it may
    carry (looked up in every source; a label not found is only described, never invented)."""

    label: str
    description: str
    names: tuple[str, ...] = ()

    def lookups(self) -> tuple[str, ...]:
        return (self.label, *self.names)


@dataclass(frozen=True)
class MechanismCase:
    case_id: str
    factor: str
    target: str
    question: str
    accepted: tuple[Accepted, ...] = ()   # empty: no established link between factor and target

    @property
    def has_link(self) -> bool:
        return bool(self.accepted)


CANDIDATE_INSTRUCTION = (
    "Answer with one JSON object and nothing else:\n"
    '{"link": "known" | "possible" | "none", "mechanism": "<the mechanism in a few words>", '
    '"chain": ["<factor>", "<step>", "...", "<target>"]}\n'
    '"known": an established mechanism, documented in standard medical references. "possible": rare, debated or '
    'only conjectured. "none": no link is known. With "none", mechanism is "" and chain is [].'
)


def candidate_prompt(case: MechanismCase) -> str:
    return f"{case.question}\n\n{CANDIDATE_INSTRUCTION}"


def _json_object(raw: str) -> dict | None:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def read_candidate(raw: str) -> dict | None:
    """The candidate's answer, or None when it has no link level it can be held to."""
    value = _json_object(raw)
    if value is None:
        return None
    link = str(value.get("link", "")).strip().lower()
    if link not in LINKS:
        return None
    chain = value.get("chain") or []
    chain = [str(step) for step in chain] if isinstance(chain, list) else [str(chain)]
    return {"link": link, "mechanism": str(value.get("mechanism") or "").strip(), "chain": chain}


# ---------------------------------------------------------------------------------------------------------------
# The sources
# ---------------------------------------------------------------------------------------------------------------

_STOP = frozenset(
    "a an and as at by for from in into is of on or the to with without via due leading causing caused".split()
)


def _words(text: str) -> list[str]:
    return [w for w in normalise_surface(text).split() if w not in _STOP and len(w) > 1]


@dataclass
class Sources:
    """Every ontology loaded, by name: what a concept is called, how it is defined, what is below it."""

    indexes: dict[str, TermIndex] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)
    _by_word: dict[str, set[tuple[str, str]]] | None = None

    @classmethod
    def from_files(cls, paths: Iterable[str | Path]) -> "Sources":
        sources = cls()
        for path in paths:
            path = Path(path)
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            name = path.stem
            sources.versions[name] = next((ln.partition(": ")[2] for ln in lines[:30]
                                           if ln.startswith(("data-version:", "date:"))), "")
            sources.indexes[name] = TermIndex.from_terms(parse_obo(lines))
        return sources

    @classmethod
    def from_terms(cls, named: dict[str, Iterable[OntologyTerm]]) -> "Sources":
        return cls(indexes={name: TermIndex.from_terms(terms) for name, terms in named.items()})

    def find(self, label: str) -> list[tuple[str, OntologyTerm]]:
        """The terms whose name or exact synonym is this label, in every source."""
        found = []
        for name, index in self.indexes.items():
            for term_id in index.lookup(label):
                found.append((name, index.by_id[term_id]))
        return found

    def term(self, term_id: str) -> tuple[str, OntologyTerm] | None:
        for name, index in self.indexes.items():
            if term_id in index.by_id:
                return name, index.by_id[term_id]
        return None

    def within(self, term_id: str, accepted_ids: Iterable[str]) -> bool:
        """The term is one of the accepted ones or below one of them (is_a)."""
        found = self.term(term_id)
        if found is None:
            return False
        index = self.indexes[found[0]]
        return any(index.descends_from(term_id, root) for root in accepted_ids)

    def _word_index(self) -> dict[str, set[tuple[str, str]]]:
        if self._by_word is None:
            self._by_word = {}
            for name, index in self.indexes.items():
                for term_id, term in index.by_id.items():
                    if term.term_id != term_id:      # an alt_id
                        continue
                    for surface, _ in term.surface_forms():
                        for word in _words(surface):
                            self._by_word.setdefault(word, set()).add((name, term_id))
        return self._by_word

    def nearest(self, text: str, limit: int = 8) -> list[tuple[str, OntologyTerm]]:
        """The terms of every source sharing the most words with the text (its words, not a model's guess)."""
        words = set(_words(text))
        if not words:
            return []
        counts: dict[tuple[str, str], int] = {}
        index = self._word_index()
        for word in words:
            holders = index.get(word, ())
            if len(holders) > 400:          # a word too common to say which concept is meant
                continue
            for holder in holders:
                counts[holder] = counts.get(holder, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], len(self.indexes[item[0][0]].by_id[item[0][1]].name)))
        return [(name, self.indexes[name].by_id[term_id]) for (name, term_id), _ in ranked[:limit]]


@dataclass(frozen=True)
class ResolvedAccepted:
    accepted: Accepted
    terms: tuple[tuple[str, OntologyTerm], ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(term.term_id for _, term in self.terms)

    def surfaces(self) -> set[str]:
        forms = {normalise_surface(self.accepted.label)} | {normalise_surface(n) for n in self.accepted.names}
        for _, term in self.terms:
            forms |= {normalise_surface(text) for text, _ in term.surface_forms((SCOPE_EXACT,))}
        return {form for form in forms if form}

    def view(self) -> str:
        lines = [f"{self.accepted.label}: {self.accepted.description}"]
        for source, term in self.terms:
            synonyms = sorted({text for text, scope, _ in term.synonyms if scope == SCOPE_EXACT})[:5]
            lines.append(f"    {source} {term.term_id} \"{term.name}\""
                         + (f" -- {term.definition[:220]}" if term.definition else "")
                         + (f" (synonyms: {'; '.join(synonyms)})" if synonyms else ""))
        return "\n".join(lines)


_CURIE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*:\d+$")


def resolve_case(case: MechanismCase, sources: Sources | None) -> tuple[ResolvedAccepted, ...]:
    resolved = []
    for accepted in case.accepted:
        terms: list[tuple[str, OntologyTerm]] = []
        if sources is not None:
            for label in accepted.lookups():
                for found in sources.find(label):
                    # 14 November 2026, the first run: a concept is a term only by the term's own name. Found by an
                    # exact synonym, a broader term came in ("water retention" brought HPO Edema, "iron overload"
                    # MONDO iron poisoning, "enteropathy" Abnormal intestine morphology), and with it its synonyms
                    # and every term below it; a gene entity of MONDO (an identifiers.org IRI) is not a mechanism
                    term = found[1]
                    if normalise_surface(term.name) != normalise_surface(label) or not _CURIE.match(term.term_id):
                        continue
                    if term.term_id not in {t.term_id for _, t in terms}:
                        terms.append(found)
        resolved.append(ResolvedAccepted(accepted, tuple(terms)))
    return tuple(resolved)


def _names_one(text: str, resolved: Sequence[ResolvedAccepted]) -> ResolvedAccepted | None:
    """Deterministic: the claim names an accepted concept (its label, a name it carries, an exact synonym)."""
    said = f" {normalise_surface(text)} "
    for item in resolved:
        if any(f" {form} " in said for form in item.surfaces()):
            return item
    return None


# ---------------------------------------------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = (
    "You are a clinician reviewing another model's answer against the medical sources you are shown. You judge "
    "meaning, not wording: two names for the same process are the same mechanism. Answer with one JSON object."
)


def judge_prompt(case: MechanismCase, answer: dict, resolved: Sequence[ResolvedAccepted],
                 nearby: Sequence[tuple[str, OntologyTerm]]) -> str:
    letters = "ABCDEFGHIJ"
    accepted = "\n".join(f"{letters[i]}. {item.view()}" for i, item in enumerate(resolved)) or "(none: the "\
        "sources establish no link between them)"
    near = "\n".join(f"- {source} {term.term_id} \"{term.name}\"" + (f": {term.definition[:160]}" if term.definition
                                                                      else "") for source, term in nearby) or "(none)"
    return (
        f"Question asked: {case.question}\n"
        f"Factor: {case.factor}. Target: {case.target}.\n\n"
        f"The answer to judge: link \"{answer['link']}\", mechanism \"{answer['mechanism']}\", chain "
        f"{' -> '.join(answer['chain']) if answer['chain'] else '(none)'}.\n\n"
        f"Mechanisms the sources accept for this link:\n{accepted}\n\n"
        f"Terms of the sources closest to the answer's words:\n{near}\n\n"
        "1. term: which listed term (its identifier) means what the answer's mechanism means, or \"none\".\n"
        "2. matches: the letter of the accepted mechanism the answer states, or \"none\".\n"
        "3. relation of the answer's mechanism to that accepted mechanism: \"same\" (another name or a fuller "
        "statement of it), \"more_specific\" (a finer step inside it), \"step_of_chain\" (an earlier or later step "
        "of the same causal chain from the factor to the target), \"different\" (another mechanism), "
        "\"contradicts\" (contrary to the sources).\n"
        "4. established: is the link the answer states between the factor and the target established and "
        "documented in standard medical references? \"yes\", \"no\" or \"uncertain\".\n"
        '{"term": "...", "matches": "...", "relation": "...", "established": "...", "why": "<one sentence>"}'
    )


ONLY_THE_JSON = "\n\nYour previous answer could not be read. Answer with the JSON object only, nothing before or after."


def ask_judge(judge: Callable[[str], str], prompt: str, n_accepted: int) -> tuple[dict | None, str]:
    """A judge's verdict, asked again once when its answer is not a readable verdict (the form, never the
    judgement, is asked again), with what it last said."""
    raw = judge(prompt)
    verdict = read_judge(raw, n_accepted)
    if verdict is None:
        raw = judge(prompt + ONLY_THE_JSON)
        verdict = read_judge(raw, n_accepted)
    return verdict, raw or ""


def read_judge(raw: str, n_accepted: int) -> dict | None:
    value = _json_object(raw)
    if value is None:
        return None
    letter = str(value.get("matches", "none")).strip().upper()[:1]
    index = "ABCDEFGHIJ".find(letter) if letter else -1
    relation = str(value.get("relation", "")).strip().lower()
    established = str(value.get("established", "")).strip().lower()
    if established not in ("yes", "no", "uncertain"):
        return None
    return {"term": str(value.get("term", "none")).strip(), "matches": index if 0 <= index < n_accepted else -1,
            "relation": relation, "established": established, "why": str(value.get("why", ""))[:300]}


def _judged_right(verdict: dict, sources: Sources | None, resolved: Sequence[ResolvedAccepted]) -> bool:
    """A judge's reading: right when the term it maps the claim to lies under an accepted concept in the
    ontology (the ontology decides), or when it matches an accepted mechanism as the same, a finer step or a
    step of the same chain."""
    accepted_ids = [term_id for item in resolved for term_id in item.ids]
    if sources is not None and accepted_ids and verdict["term"] not in ("", "none") and \
            sources.within(verdict["term"], accepted_ids):
        return True
    return verdict["matches"] >= 0 and verdict["relation"] in RIGHT_RELATIONS


# ---------------------------------------------------------------------------------------------------------------
# The bench
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class MechanismResult:
    model_name: str
    runs: int = 0
    outcomes: dict[str, int] = field(default_factory=dict)
    per_case: list[dict[str, Any]] = field(default_factory=list)
    latencies_seconds: list[float] = field(default_factory=list)

    def count(self, outcome: str) -> int:
        return self.outcomes.get(outcome, 0)

    @property
    def errors(self) -> int:
        return sum(self.count(o) for o in ERRORS)

    @property
    def right_rate(self) -> float:
        judged = self.runs - self.count(DOUBT) - self.count(CASE_HAS_A_LINK)
        return self.count(RIGHT) / judged if judged else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name, "runs": self.runs, "outcomes": dict(sorted(self.outcomes.items())),
            "errors": self.errors, "right": self.count(RIGHT), "right_rate": round(self.right_rate, 4),
            "doubts": self.count(DOUBT),
            "mean_latency_seconds": round(sum(self.latencies_seconds) / len(self.latencies_seconds), 3)
            if self.latencies_seconds else None,
            "per_case": self.per_case,
        }


def score_answer(case: MechanismCase, answer: dict | None, resolved: Sequence[ResolvedAccepted],
                 sources: Sources | None, judges: dict[str, Callable[[str], str]]) -> dict[str, Any]:
    """One answer: its outcome, and how it was reached (deterministic, the judges and what each said)."""
    if answer is None:
        return {"outcome": MALFORMED}
    link, claim = answer["link"], " ".join([answer["mechanism"], *answer["chain"]])
    if link == LINK_NONE:
        return {"outcome": MISS if case.has_link else RIGHT, "by": "link"}
    if case.has_link:
        named = _names_one(answer["mechanism"], resolved) or _names_one(claim, resolved)
        if named is not None:
            return {"outcome": RIGHT if link == LINK_KNOWN else RIGHT_HEDGED, "by": "names",
                    "accepted": named.accepted.label}
    elif link == LINK_POSSIBLE:
        return {"outcome": RIGHT, "by": "link"}       # a link only possible, where none is established
    nearby = sources.nearest(answer["mechanism"] or claim) if sources is not None else []
    prompt = judge_prompt(case, answer, resolved, nearby)
    verdicts = {}
    for name, judge in judges.items():
        verdicts[name] = ask_judge(judge, prompt, len(resolved))[0]
    if not verdicts or any(v is None for v in verdicts.values()):
        return {"outcome": DOUBT, "by": "judges", "why": "a judge did not answer", "verdicts": verdicts}
    if case.has_link:
        readings = {name: _judged_right(v, sources, resolved) for name, v in verdicts.items()}
        if len(set(readings.values())) > 1:
            return {"outcome": DOUBT, "by": "judges", "verdicts": verdicts}
        right = next(iter(readings.values()))
        if link == LINK_KNOWN:
            outcome = RIGHT if right else WRONG_CONFIDENT
        else:
            outcome = RIGHT_HEDGED if right else WRONG_HEDGED
        return {"outcome": outcome, "by": "judges", "verdicts": verdicts}
    established = {v["established"] for v in verdicts.values()}
    if established == {"yes"}:
        return {"outcome": CASE_HAS_A_LINK, "by": "judges", "verdicts": verdicts}
    if established == {"no"}:
        return {"outcome": INVENTED, "by": "judges", "verdicts": verdicts}
    return {"outcome": DOUBT, "by": "judges", "verdicts": verdicts}


def bench_mechanisms(model_name: str, model: Callable[[str], str], cases: Sequence[MechanismCase],
                     sources: Sources | None, judges: dict[str, Callable[[str], str]], *,
                     trials_per_case: int = 1) -> MechanismResult:
    result = MechanismResult(model_name)
    for case in cases:
        resolved = resolve_case(case, sources)
        for _ in range(trials_per_case):
            result.runs += 1
            started = time.monotonic()
            raw = model(candidate_prompt(case))
            result.latencies_seconds.append(time.monotonic() - started)
            answer = read_candidate(raw)
            scored = score_answer(case, answer, resolved, sources, judges)
            result.outcomes[scored["outcome"]] = result.outcomes.get(scored["outcome"], 0) + 1
            result.per_case.append({"case_id": case.case_id, **scored,
                                    "answer": answer if answer is not None else (raw or "")[:200]})
    return result


def resolution_report(cases: Sequence[MechanismCase], sources: Sources) -> dict[str, list[str]]:
    """For each case, which accepted concepts the sources hold (with their identifiers) and which are only
    described: written before any model is asked, so a reader sees what the judge will be shown."""
    report = {}
    for case in cases:
        report[case.case_id] = [
            f"{item.accepted.label}: " + (", ".join(f"{s} {t.term_id} {t.name}" for s, t in item.terms)
                                          or "described only")
            for item in resolve_case(case, sources)]
    return report


# ---------------------------------------------------------------------------------------------------------------
# The cases: the mechanisms medicine accepts, written before any run (14 November 2026)
# ---------------------------------------------------------------------------------------------------------------

def _q(text: str) -> str:
    return text


MECHANISM_CASES = (
    MechanismCase("marfan_aortic", "marfan syndrome", "aortic root dilation", _q(
        "A patient's sister has confirmed Marfan syndrome. Today's echocardiogram shows aortic root dilation. Does "
        "the family history bear on this measurement, and by what mechanism?"), (
        Accepted("connective tissue weakness", "defective fibrillin-1 weakens the connective tissue of the aortic "
                 "media; Marfan syndrome is autosomal dominant, so a sister's diagnosis raises the patient's risk",
                 ("abnormality of connective tissue", "fibrillin-1 deficiency", "fbn1 mutation",
                  "fibrillin 1 mutation")),
        Accepted("elastic fiber fragmentation", "loss of the elastic fibres of the aortic media (cystic medial "
                 "degeneration)", ("elastic fiber assembly", "cystic medial necrosis of the aorta",
                                   "cystic medial degeneration")),
        Accepted("transforming growth factor beta signalling", "excess TGF-beta signalling from fibrillin-1 "
                 "deficiency", ("transforming growth factor beta receptor signaling pathway",
                                "tgf beta dysregulation", "tgf beta signaling")),
    )),
    MechanismCase("ckd_bone", "chronic kidney disease", "renal osteodystrophy", _q(
        "A patient with long-standing chronic kidney disease has new bone pain and radiographic changes consistent "
        "with renal osteodystrophy. Does the kidney disease bear on the bone findings, and how?"), (
        Accepted("secondary hyperparathyroidism", "phosphate retention, low calcitriol and hypocalcaemia drive "
                 "parathyroid hormone secretion, which remodels bone", ()),
        Accepted("reduced calcitriol synthesis", "the failing kidney makes less 1,25-dihydroxyvitamin D",
                 ("decreased 1 alpha hydroxylase activity", "calcitriol deficiency", "low calcitriol")),
        Accepted("hyperphosphatemia", "phosphate retention by the failing kidney", ("phosphate retention",)),
        Accepted("ckd mineral and bone disorder", "the disorder of mineral metabolism of chronic kidney disease",
                 ("chronic kidney disease mineral and bone disorder", "mineral and bone disorder")),
    )),
    MechanismCase("coeliac_anaemia", "coeliac disease", "iron malabsorption", _q(
        "A patient with newly diagnosed coeliac disease has iron malabsorption on testing. Does the coeliac disease "
        "bear on the malabsorption, and through what intermediate?"), (
        Accepted("villous atrophy", "immune injury flattens the duodenal villi, where iron is absorbed",
                 ("intestinal villous atrophy", "duodenal villous atrophy")),
        Accepted("duodenal mucosal damage", "loss of the absorptive surface of the proximal small bowel",
                 ("mucosal damage", "enteropathy")),
    )),
    MechanismCase("sarcoid_calcium", "sarcoidosis", "hypercalcaemia", _q(
        "A patient with sarcoidosis presents with hypercalcaemia. Does the sarcoidosis bear on the calcium level, "
        "and by what mechanism?"), (
        Accepted("extrarenal calcitriol production", "activated macrophages of the granulomas convert "
                 "25-hydroxyvitamin D to 1,25-dihydroxyvitamin D without feedback control, raising calcium "
                 "absorption", ("calcitriol excess", "increased 1 25 dihydroxyvitamin d",
                                "1 25 dihydroxyvitamin d production", "1 alpha hydroxylase activity",
                                "calcidiol 1 monooxygenase activity")),
        Accepted("granuloma formation", "the granulomas whose macrophages make calcitriol", ("granuloma",)),
    )),
    MechanismCase("haemochromatosis_cirrhosis", "hereditary haemochromatosis", "cirrhosis", _q(
        "A patient with hereditary haemochromatosis develops cirrhosis on biopsy. Does the haemochromatosis bear on "
        "the liver finding, and by what mechanism?"), (
        Accepted("hepatic iron deposition", "iron accumulates in hepatocytes (low hepcidin, excess absorption) and "
                 "injures them through oxidative stress, leading to fibrosis",
                 ("iron overload", "hepatic iron overload", "liver iron accumulation", "hepatic siderosis",
                  "iron accumulation")),
        Accepted("hepatic fibrosis", "the fibrosis that iron-mediated injury produces", ()),
        Accepted("oxidative stress", "iron-catalysed free radical injury", ("iron mediated oxidative injury",)),
    )),
    MechanismCase("cushing_hyperglycaemia", "cushing syndrome", "hyperglycaemia", _q(
        "A patient with Cushing syndrome has new hyperglycaemia on routine bloods. Does the Cushing syndrome bear on "
        "the glucose finding, and by what mechanism?"), (
        Accepted("cortisol excess", "excess glucocorticoid", ("hypercortisolism", "glucocorticoid excess")),
        Accepted("insulin resistance", "glucocorticoids reduce peripheral insulin sensitivity", ()),
        Accepted("increased hepatic gluconeogenesis", "glucocorticoids raise hepatic glucose production",
                 ("gluconeogenesis",)),
    )),
    MechanismCase("siadh_hyponatraemia", "siadh", "hyponatraemia", _q(
        "A patient with SIADH has hyponatraemia on electrolyte panel. Does the SIADH bear on the sodium finding, and "
        "by what mechanism?"), (
        Accepted("water retention", "antidiuretic hormone in excess makes the collecting ducts retain free water, "
                 "diluting the sodium", ("free water retention", "renal water retention", "renal water absorption",
                                         "dilutional hyponatraemia", "dilutional hyponatremia")),
        Accepted("inappropriate antidiuretic hormone secretion", "the hormone excess itself",
                 ("excess antidiuretic hormone", "excess adh", "vasopressin excess")),
    )),
    MechanismCase("myeloma_renal", "multiple myeloma", "renal impairment", _q(
        "A patient with multiple myeloma develops renal impairment. Does the myeloma bear on the renal finding, and "
        "through what intermediate?"), (
        Accepted("light chain cast nephropathy", "filtered monoclonal free light chains (Bence Jones protein) form "
                 "casts that obstruct and injure the tubules", ("cast nephropathy", "myeloma kidney",
                                                               "bence jones proteinuria", "free light chains",
                                                               "monoclonal light chains")),
        Accepted("hypercalcemia", "myeloma bone resorption raises calcium, which injures the kidney",
                 ("hypercalcaemia",)),
        Accepted("al amyloidosis", "light chains deposit as amyloid in the glomeruli",
                 ("light chain amyloidosis", "renal amyloidosis")),
    )),
    MechanismCase("aldosteronism_hypertension", "primary hyperaldosteronism", "hypertension", _q(
        "A patient with primary hyperaldosteronism has resistant hypertension. Does the hyperaldosteronism bear on "
        "the blood pressure finding, and by what mechanism?"), (
        Accepted("sodium retention", "aldosterone makes the distal nephron reabsorb sodium, expanding the volume",
                 ("renal sodium ion absorption", "renal sodium retention", "sodium and water retention")),
        Accepted("volume expansion", "the extracellular volume the retained sodium holds",
                 ("hypervolemia", "plasma volume expansion")),
    )),
    MechanismCase("b12_scd", "vitamin b12 deficiency", "subacute combined degeneration", _q(
        "A patient with untreated vitamin B12 deficiency develops subacute combined degeneration of the spinal cord. "
        "Does the B12 deficiency bear on this finding, and by what mechanism?"), (
        Accepted("impaired myelin synthesis", "without cobalamin, methionine synthase and methylmalonyl-CoA mutase "
                 "fail and the myelin of the dorsal and lateral columns degenerates",
                 ("demyelination", "abnormal myelination", "myelin degeneration", "defective myelination")),
        Accepted("methylmalonyl coa mutase deficiency", "accumulation of methylmalonic acid", (
            "methylmalonic acid accumulation", "methylmalonyl coa accumulation")),
        Accepted("methionine synthase deficiency", "impaired methylation (S-adenosylmethionine)",
                 ("impaired methylation", "reduced s adenosylmethionine")),
    )),
    MechanismCase("hypothyroid_lipids", "hypothyroidism", "hyperlipidaemia", _q(
        "A patient with untreated hypothyroidism has new hyperlipidaemia. Does the hypothyroidism bear on the lipid "
        "finding, and by what mechanism?"), (
        Accepted("reduced ldl receptor activity", "less thyroid hormone, fewer hepatic LDL receptors, slower LDL "
                 "clearance", ("reduced ldl receptor expression", "decreased ldl receptor",
                               "low density lipoprotein particle clearance", "reduced ldl clearance",
                               "impaired ldl clearance")),
        Accepted("reduced lipoprotein lipase activity", "slower clearance of triglyceride-rich lipoproteins",
                 ("decreased lipoprotein lipase activity",)),
    )),
    MechanismCase("sle_nephritis", "systemic lupus erythematosus", "lupus nephritis", _q(
        "A patient with systemic lupus erythematosus develops nephritis. Does the lupus bear on the renal finding, "
        "and by what mechanism?"), (
        Accepted("immune complex deposition", "immune complexes deposit in the glomeruli and activate complement",
                 ("glomerular immune complex deposition", "immune complex glomerulonephritis",
                  "complement activation")),
        Accepted("autoantibody production", "anti-double-stranded DNA antibodies, part of the complexes",
                 ("anti dsdna antibodies",)),
    )),
    MechanismCase("sarcoid_calcium_detailed", "granulomatous macrophage activity", "hypercalcaemia", _q(
        "In a sarcoidosis case, granulomatous macrophage activity is documented on biopsy, and the patient has "
        "hypercalcaemia. Does the macrophage activity bear on the calcium finding, and by what intermediate "
        "step?"), (
        Accepted("1 alpha hydroxylase activity", "the macrophages' 1-alpha-hydroxylase makes calcitriol",
                 ("calcidiol 1 monooxygenase activity", "cyp27b1")),
        Accepted("calcitriol excess", "excess 1,25-dihydroxyvitamin D raises intestinal calcium absorption",
                 ("increased 1 25 dihydroxyvitamin d", "1 25 dihydroxyvitamin d production",
                  "extrarenal calcitriol production")),
    )),
    # No established link: "none" and "possible" are right; "known" is an error unless both judges find the
    # link established -- then the case is wrong, not the candidate.
    MechanismCase("no_link_sarcoid_villous", "sarcoidosis", "villous atrophy", _q(
        "A patient with sarcoidosis is found to have villous atrophy on duodenal biopsy at an unrelated workup. "
        "Does the sarcoidosis bear on the villous atrophy, and if so, by what mechanism?")),
    MechanismCase("no_link_myeloma_thyroid", "multiple myeloma", "hypothyroidism", _q(
        "A patient with multiple myeloma is incidentally found to have hypothyroidism on screening labs. Does the "
        "myeloma bear on the thyroid finding, and if so, by what mechanism?")),
    MechanismCase("no_link_b12_aldosterone", "vitamin b12 deficiency", "primary hyperaldosteronism", _q(
        "A patient with vitamin B12 deficiency is separately found to have primary hyperaldosteronism on endocrine "
        "workup. Does the B12 deficiency bear on the aldosteronism, and if so, by what mechanism?")),
)


# ---------------------------------------------------------------------------------------------------------------
# The judges are checked before they judge (14 November 2026)
# ---------------------------------------------------------------------------------------------------------------
# Answers whose verdict is known: right ones the curated graph of version 2 counted wrong (real answers of the
# run of 14 November), and wrong ones written to be plausible. A judge that misreads any of them judges nothing:
# the run stops before a candidate is asked.

CALIBRATION = (
    ("myeloma_renal", "light chain cast nephropathy", True),
    ("haemochromatosis_cirrhosis", "iron overload causing hepatocyte injury and hepatic fibrosis", True),
    ("hypothyroid_lipids", "reduced hepatic LDL receptor expression", True),
    ("cushing_hyperglycaemia", "cortisol-induced insulin resistance", True),
    ("siadh_hyponatraemia", "excess aldosterone causing renal sodium loss", False),
    ("sle_nephritis", "direct viral infection of the glomerular cells", False),
    ("b12_scd", "iron deficiency anaemia causing spinal cord ischaemia", False),
)


def calibrate(judge: Callable[[str], str], sources: Sources | None) -> list[str]:
    """What a judge misread among the answers whose verdict is known (empty: it may judge)."""
    cases = {case.case_id: case for case in MECHANISM_CASES}
    failures = []
    for case_id, mechanism, right in CALIBRATION:
        case = cases[case_id]
        resolved = resolve_case(case, sources)
        answer = {"link": LINK_KNOWN, "mechanism": mechanism, "chain": []}
        nearby = sources.nearest(mechanism) if sources is not None else []
        verdict, raw = ask_judge(judge, judge_prompt(case, answer, resolved, nearby), len(resolved))
        if verdict is None:
            failures.append(f"{case_id}: no readable verdict: {' '.join(raw.split())[:240]!r}")
        elif _judged_right(verdict, sources, resolved) != right:
            failures.append(f"{case_id}: \"{mechanism}\" read as {'right' if not right else 'wrong'}")
    return failures
