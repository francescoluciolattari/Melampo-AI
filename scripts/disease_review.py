"""A systematic review of every disease, by machine: all its manifestations from all the sources, and their weights
(22 November 2026).

    python3 scripts/disease_review.py --mondo mondo.obo --hpo hp.obo --hpoa phenotype.hpoa --doid-owl doid.owl \
        --symp symp.obo --diagnoses "acute appendicitis,pneumonia" --out out/profiles.json \
        [--saturation 5] [--budget 5] [--resume previous]

It takes the place of the two runs that came before it (disease-profiles, which listed the manifestations from a
few texts, and disease-frequencies, which weighed them from a few others): one review per disease, the method of a
systematic review (PRISMA) carried out with the readers and the judges of Melampo.

1. **The names.** MONDO's name and exact synonyms; Europe PMC expands MeSH's synonyms by itself.
2. **The curated layer.** HPO's annotations (Orphanet's included, with their frequencies) and the Disease
   Ontology: the manifestations already verified, from which the review starts.
3. **Every source, in layers**, each read to its end: the sources of the index (``data/sources/index.yml``, the
   ones the configuration mask turns on: books, PDFs, BMJ...), then Europe PMC: guidelines; JAMA's Rational
   Clinical Examination; systematic reviews and meta-analyses of diagnosis; studies of diagnostic accuracy;
   descriptive series of the presentation; narrative reviews; case reports. By relevance, page after page; a title
   on another cause or form (COVID-19), prognosis, treatment, risk factors or epidemiology is left out.
4. **Screening.** Before a text is read, its title and abstract must name the disease, and a screener (Gemma 4,
   no reasoning) must say it is about the presentation or the diagnosis of the disease in general.
5. **Interpretation and verification** (corrected 22 November 2026: the interpretation is the interpreters', the
   judges only verify). The two interpreters, as for the reports (Nemotron 3 Ultra + Muse Spark Open, full
   reasoning), read the whole text and give, for everything it states: the words of the text, the general name
   (without the values of one patient), what it is (manifestation, test, score, outcome, risk factor, form of the
   disease), whether it is one thing or a list, where it is found (history, examination, laboratory, imaging,
   endoscopy, intraoperative, histopathology, course), whether it is pathological or physiological, the HPO term
   by name, which known manifestation it is, and the measures. A code is added only when the HPO term they name is
   HPO's own name or exact synonym: never an approximate term, never a code written by a model. What the two
   interpreters read differently stays a doubt (a category read differently is not a datum; an HPO term read
   differently gives no code). The checks are the rules of ``disease_frequencies.checked`` (the sentence in the
   text, the numbers in the sentence). The judges, independent of the interpreters (Claude Sonnet 5, then Gemma 4,
   no reasoning), verify only: does the sentence state it, with these numbers, for this disease in general?
   Corrections of 23 November 2026, after the second run: (A) a sign, a laboratory value beyond a cut-off, an
   imaging finding is a manifestation, with its accuracy; a test is an examination as a whole (CT, ultrasound,
   MRI), for the next exam. (B) A statement of direction without a number ("makes appendicitis less likely") is
   recorded and weighs by a provisional likelihood ratio, to be calibrated at step 9. (C) A measure comparing two
   forms of the disease (complicated against uncomplicated) does not weigh the diagnosis. (D) A result stated
   more than once in one article, with the same numbers, counts once; every sentence stays a citation.
   Corrections of 24 November 2026, after the third run: (E) one format for every model, the measures only inside one
   "measures" object; another form is not read and is counted for its model; (F) a measure is one value the text states as one value, not a bound or a range; a refused
   measure leaves the manifestation; (G) each interpreter's reading is its own record: a measure or a direction
   weighs only when both read it in the same source, a link read by one interpreter alone is not made; (H) at the
   end, the two interpreters group the names of one manifestation and join an absence to its thing, and only what
   both group is joined; (I) a direction is verified as a claim of its own; (J) a page that does not answer is
   tried again, every layer's hits are written, an unreadable layer is said.
   On 25 November 2026 the reading moved to the one procedure of reading (``melampo.reading``: the same card,
   comparison and coding as a patient's documents); it replaces the reader's prompt, the union by identical names
   and the grouping at the end (H): readings of one finding are paired in the sentence, names across texts meet
   in the thesaurus.
6. **When to stop: a parameter** (``--saturation``, the owner's rule of 22 November 2026). The review of a disease
   stops when that many texts in a row, read and judged, add no manifestation; the least is 5 (a smaller value
   is raised to 5), and 0 means no limit: every text of every layer is read, to the end of the sources. No other
   limit is set beforehand; the budget stops the run, which resumes where it stopped.
7. **Completeness.** A curated manifestation no text has stated yet is searched by itself (the disease and the
   manifestation together), up to ``--targeted`` texts each.
8. **Weights**, in the order of ``disease_frequencies``: measured likelihood ratios by setting, then frequencies
   pooled by a random-effects model, then words of frequency; what is left without weight is said.
9. **What is written**, as it goes: ``profiles.json`` (the diseases with their manifestations, each with its
   sources, evidence, frequency and accuracy), ``tests.json`` (how well each exam detects the disease),
   ``scores.json`` (clinical scores), ``curves.json`` (texts read and new manifestations, for every disease), the
   logs from which a run resumes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rs = _script("reference_sources")
df = _script("disease_frequencies")

from melampo.clinical import case_graph as cg  # noqa: E402
from melampo.memory import disease_profiles as dp  # noqa: E402
from melampo.reading import Thesaurus, read_text  # noqa: E402
from melampo.reading import thesaurus as th  # noqa: E402
from melampo.reading.card import CATEGORIES, COMPARES, DIRECTIONS, MEASURE_FIELDS, NATURES, OBSERVED_BY  # noqa: E402,F401
from melampo.reading.normalize import HpoIndex  # noqa: E402
from melampo.reading.prompt import reading_prompt  # noqa: E402

SATURATION_LEAST = 5
SCREENER = "google/gemma-4-31b-it"
JUDGED = "v6:"         # the judges verify only (22 November 2026): the interpretation is the interpreters'
QUESTION_ROOM = 16000  # tokens of the question of the codes for a text, its reasoning included (the look at the other's
                       # findings has the room of a reading: it may give a card for each)
JUDGED_CLAIM = "v7:"   # a claim with its numbers or its direction is verified as such (24 November 2026)
LAYERS = (
    ("guideline", '(PUB_TYPE:"guideline" OR PUB_TYPE:"practice-guideline" OR TITLE:"guideline" OR '
                  'TITLE:"guidelines" OR TITLE:"consensus" OR TITLE:"recommendations")'),
    ("rational_clinical_examination", 'TITLE:"does this patient have"'),
    ("systematic_review_of_diagnosis", '(PUB_TYPE:"systematic-review" OR TITLE:"meta-analysis" OR '
                                       'TITLE:"systematic review") AND (TITLE:"diagnos" OR TITLE:"clinical" OR '
                                       'TITLE:"signs" OR TITLE:"symptoms" OR TITLE:"accuracy")'),
    ("diagnostic_accuracy", '(ABSTRACT:"sensitivity" OR ABSTRACT:"likelihood ratio" OR ABSTRACT:"diagnostic accuracy")'
                            ' AND (TITLE:"clinical" OR TITLE:"signs" OR TITLE:"symptoms" OR TITLE:"examination" OR '
                            'TITLE:"findings" OR TITLE:"diagnos")'),
    ("descriptive_series", '(TITLE:"clinical features" OR TITLE:"clinical characteristics" OR TITLE:"presentation" OR '
                           'TITLE:"presenting" OR TITLE:"symptoms" OR TITLE:"signs" OR TITLE:"manifestations")'),
    ("narrative_review", 'PUB_TYPE:"review"'),
    ("case_report", 'PUB_TYPE:"case-reports"'),
)
PAGE = 50


def saturation_of(value: int) -> int:
    """0 is no limit (to the end of the sources); any other value is at least ``SATURATION_LEAST``."""
    return 0 if value <= 0 else max(SATURATION_LEAST, value)


def names_of(name: str, mondo: dp.Terms, mondo_id: str) -> list[str]:
    out = [name] + ([mondo.names[mondo_id]] if mondo_id in mondo.names else []) + list(mondo.synonyms.get(mondo_id, []))
    seen, kept = set(), []
    for n in out:
        if n and dp.norm(n) not in seen and len(n) > 3:
            seen.add(dp.norm(n))
            kept.append(n)
    return kept[:8]


def layer_query(names: list[str], layer: str, since: str = "") -> str:
    title = " OR ".join(f'TITLE:"{n}"' for n in names)
    after = f" AND FIRST_PDATE:[{since} TO 3000-12-31]" if since else ""
    return f"({title}) AND {dict(LAYERS)[layer]} AND HAS_ABSTRACT:y{after}"


ATTEMPTS = (0, 10, 30, 90)   # seconds before each attempt at one page of Europe PMC


def records(names: list[str], layer: str, get=None, since: str = "", report: dict | None = None, wait=time.sleep):
    """Every record of one layer, page after page by Europe PMC's cursor, to the end; with ``since``, only those
    first published on that day or after. ``report`` receives the hits Europe PMC counts, the pages read and
    whether a page could not be read (24 November 2026, declared after the second run: a page that did not answer
    ended its layer in silence, and four layers of the review gave no text without a trace of why)."""
    get = get or rs.get
    report = report if report is not None else {}
    report.update(hits=None, pages=0, unreachable=False)
    cursor = "*"
    while True:
        q = {"query": layer_query(names, layer, since), "format": "json", "resultType": "core", "pageSize": PAGE,
             "cursorMark": cursor}
        page = None
        for pause in ATTEMPTS:
            wait(pause)
            try:
                page = json.loads(get(f"{rs.EPMC}/search?{rs.urllib.parse.urlencode(q)}") or "null")
            except json.JSONDecodeError:
                page = None
            if isinstance(page, dict) and "resultList" in page:
                break
            page = None
        if page is None:
            report["unreachable"] = True
            return
        report["pages"] += 1
        if report["hits"] is None:
            report["hits"] = int(page.get("hitCount") or 0)
        results = page.get("resultList", {}).get("result", [])
        yield from results
        nxt = page.get("nextCursorMark")
        if not results or not nxt or nxt == cursor:
            return
        cursor = nxt


def off_topic(title: str, disease: str) -> bool:
    return any(t.lower() in title.lower() for t in df.OFF_TOPIC if t.lower() not in disease.lower())


def screen_prompt(disease: str, record: dict) -> str:
    abstract = " ".join(re.sub(r"<[^>]+>", " ", str(record.get("abstractText", ""))).split())[:3000]
    return (f"Disease: {disease}\nTitle: {record.get('title', '')}\nAbstract: {abstract}\n\n"
            f"Is this publication about the manifestations (symptoms, signs, laboratory or imaging findings, "
            f"complications), the presentation or the diagnosis of {disease} in general, in humans? Answer no when "
            "it is about one particular cause or form only, about treatment, prognosis, risk factors, epidemiology, "
            'or another disease.\nAnswer only JSON: {"about": "yes"} or {"about": "no"}.')


READING = "r6"         # the one procedure of reading: the same card as a patient's report; r6 (27 November 2026): the part of the document


def judge_prompt(disease: str, item: dict) -> str:
    """The judges (independent of the interpreters, no reasoning) verify only: does the sentence state it?"""
    measure = df.claim(disease, item) if any(k in item for k in ("with_disease", "sensitivity", "specificity",
                                                                 "lr_positive", "lr_negative", "verbal")) else ""
    return (f"Disease: {disease}\nStated: {item['manifestation']} ({item.get('category', 'manifestation')})\n"
            f"Sentence (or table row) from a published text: \"{item['quote']}\"\n"
            f"Claim: {claim_of(disease, item, measure)}.\n"
            "Answer yes when the sentence states this (and the numbers, if any) for this disease in general; no when "
            "it does not, when it belongs to another disease, to one particular cause or form only, or to a group the "
            "claim does not name.\n"
            'Answer only JSON: {"states": "yes"} or {"states": "no"}.')


def claim_of(disease: str, item: dict, measure: str = "") -> str:
    """What the judges verify. A statement of direction is a claim of its own: "vomiting before pain, when present,
    makes appendicitis less likely" (24 November 2026, declared after the second run: written "in appendicitis,
    vomiting before pain; ... less likely", it claimed what the sentence denies, and the judges refused it)."""
    d = item.get("direction")
    if d:
        return (f"{item['manifestation']}, when {d['when']}, makes {disease} {d['effect'].replace('_', ' ')}"
                + (f" ({d['words']})" if d.get("words") else ""))
    return f"in {disease}, {item['manifestation']}" + (f"; {measure}" if measure else "")


def read_verdict(answer: str) -> dict:
    return {"states": dp.read_judgment(answer)}


def replaced(lines: list[str]) -> dict[str, tuple[str, str]]:
    """MONDO's obsolete terms by their name and exact synonyms: name -> (the term it was, the term that replaces
    it) (26 November 2026, declared after the seventh run: "acute appendicitis" no longer found by its name went to
    a broader term by another synonym)."""
    out: dict[str, tuple[str, str]] = {}
    current, names, by, obsolete = "", [], "", False

    def close():
        if current and obsolete and by:
            for n in names:
                out.setdefault(dp.norm(n), (current, by))
    for line in lines + ["[Term]"]:
        if line.startswith("[Term]"):
            close()
            current, names, by, obsolete = "", [], "", False
        elif line.startswith("id: MONDO:"):
            current = line[4:].strip()
        elif line.startswith("name: "):
            names.append(line[6:].strip().removeprefix("obsolete "))
        elif line.startswith("is_obsolete: true"):
            obsolete = True
        elif line.startswith("replaced_by: MONDO:"):
            by = line[len("replaced_by: "):].strip()
        elif line.startswith("synonym: ") and " EXACT" in line:
            m = re.match(r'synonym: "(.*?)"', line)
            if m:
                names.append(m.group(1))
    return out


def any_synonym(lines: list[str]) -> dict[str, str]:
    """MONDO's terms by any of their synonyms (exact, related, narrow, broad), only where one term has it."""
    found: dict[str, set[str]] = {}
    current, obsolete = "", False
    for line in lines + ["[Term]"]:
        if line.startswith("[Term]"):
            current, obsolete = "", False
        elif line.startswith("id: MONDO:"):
            current = line[4:].strip()
        elif line.startswith("is_obsolete: true"):
            obsolete = True
        elif current and not obsolete and line.startswith("synonym: "):
            m = re.match(r'synonym: "(.*?)"', line)
            if m:
                found.setdefault(dp.norm(m.group(1)), set()).add(current)
    return {k: next(iter(v)) for k, v in found.items() if len(v) == 1}


def measures_of(item: dict) -> dict:
    """The measures of an item, read only where the format puts them: inside "measures"."""
    nested = item.get("measures")
    return {k: nested[k] for k in MEASURE_FIELDS if k in nested} if isinstance(nested, dict) else {}


def item_checked(item: dict, sections: list[tuple[str, str]]) -> tuple[dict | None, str]:
    """A manifestation with or without a measure, by the rules fixed before the run. Its measures are read only inside
    "measures" (the one format). A measure refused (a range, a bound, a number not in the sentence) leaves the manifestation, stated by the sentence, without it: the reason is
    kept as a doubt (K, 24 November 2026, declared after the second run: the manifestation was lost with it)."""
    quote = " ".join(str(item.get("quote", "")).split())
    name = str(item.get("name") or item.get("manifestation") or item.get("words") or "").strip()
    if not name or not quote or not df.in_text(quote, sections):
        return None, "quote_not_in_text"
    item = {"quote": quote, **measures_of(item), "manifestation": name}
    m, why = df.checked(item, sections)
    if m is not None:
        return m, ""
    bare = {"manifestation": name, "quote": quote, "setting": str(item.get("setting", "") or "not stated").lower(),
            "comparator": "", "design": str(item.get("design", "other")).lower(), "pooled": False, "patients": None}
    if why == "no_measure":       # a manifestation stated without a measure is a manifestation
        return bare, ""
    return {**bare, "measure_refused": why}, why


def dedupe(measures: list[dict]) -> list[dict]:
    """(D) A result stated more than once in one article, with the same numbers, counts once; every sentence stays
    a citation of the one kept."""
    out: dict[str, dict] = {}
    for m in measures:
        src = m.get("source") or {}
        wd = m.get("with_disease") or {}
        key = json.dumps([src.get("pmcid") or src.get("pmid") or src.get("doi") or src.get("title") or "",
                          *(m.get(k) for k in ("sensitivity", "specificity", "lr_positive", "lr_negative")),
                          wd.get("count"), wd.get("total"), wd.get("proportion")], sort_keys=True)
        if key in out:
            out[key].setdefault("also_stated", []).append(m["quote"])
        else:
            out[key] = dict(m)
    return list(out.values())


def for_weights(measures: list[dict], names: list[str]) -> list[dict]:
    """(C) Only what compares patients with the disease against patients without it weighs the diagnosis: a
    comparison between forms of the same disease (complicated against uncomplicated) does not."""
    kept = []
    for m in measures:
        if m.get("compares") == "forms_of_the_disease":
            continue
        comparator = dp.norm(m.get("comparator", ""))
        if comparator and any(dp.norm(n) in comparator for n in names) and "without" not in comparator \
                and "non " not in comparator and "no " not in f" {comparator} ":
            continue
        kept.append(m)
    return kept


def source_of(m: dict) -> str:
    src = m.get("source") or {}
    return str(src.get("pmcid") or src.get("pmid") or src.get("doi") or src.get("id") or src.get("title") or "")


def numbers_of(m: dict) -> dict:
    wd = m.get("with_disease") or {}
    out = {k: m.get(k) for k in ("sensitivity", "specificity", "lr_positive", "lr_negative")}
    out.update(count=wd.get("count"), total=wd.get("total"), proportion=wd.get("proportion"),
               verbal=(m.get("verbal") or {}).get("class"))
    return {k: v for k, v in out.items() if v is not None}


def agreed_measures(measures: list[dict]) -> list[dict]:
    """The measures both interpreters read in the same source with the same numbers, one each (a measure one
    interpreter alone read stays a citation, a doubt); a source whose agreed results for one manifestation differ
    (not one contained in another) weighs nothing for it: which result is the manifestation's is a doubt."""
    groups: dict[tuple, list[dict]] = {}
    for m in measures:
        groups.setdefault((source_of(m), json.dumps(numbers_of(m), sort_keys=True)), []).append(m)
    agreed_: dict[str, list[dict]] = {}
    for (src, _), g in groups.items():
        readers = sorted({x.get("reader") for x in g if x.get("reader")})
        if len(readers) < len(rs.READERS):
            continue
        first = {**g[0], "readers": readers,
                 "also_stated": sorted({x["quote"] for x in g[1:]} - {g[0]["quote"]})}
        agreed_.setdefault(src, []).append(first)
    out = []
    for src, ms in agreed_.items():
        widest = [m for m in ms if not any(numbers_of(m).items() < numbers_of(o).items() for o in ms)]
        if len(widest) == 1:
            out.append(widest[0])
    return out


def agreed_directions(records: list[dict]) -> tuple[list[dict], int]:
    """The statements of direction both interpreters read in the same source (one per source), and how many were
    read by one alone."""
    by: dict[tuple, set] = {}
    words: dict[tuple, str] = {}
    for m in records:
        d = m.get("direction")
        if not d:
            continue
        k = (source_of(m), d["when"], d["effect"])
        by.setdefault(k, set()).add(m.get("reader") or "")
        words[k] = " ".join(sorted(set((words.get(k, "") + " " + d.get("words", "")).split())))
    agreed_ = [{"when": k[1], "effect": k[2], "words": words[k]} for k, rd in by.items()
               if len(rd - {""}) >= len(rs.READERS)]
    return agreed_, sum(1 for rd in by.values() if len(rd - {""}) < len(rs.READERS))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("mondo", "hpo", "hpoa", "doid-owl", "symp"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--diagnoses", default="", help="comma-separated names (with --update: empty = all reviewed)")
    ap.add_argument("--saturation", type=int, default=SATURATION_LEAST,
                    help=f"texts in a row adding nothing before a disease's review stops; at least "
                         f"{SATURATION_LEAST}; 0 = no limit, to the end of the sources")
    ap.add_argument("--targeted", type=int, default=3, help="texts searched for a curated manifestation not found")
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--research", action="store_true", help="also the research-only sources of the index")
    ap.add_argument("--out", default="out/profiles.json")
    ap.add_argument("--knowledge", default="", help="the folder of the knowledge already reviewed (the knowledge "
                                                     "branch): its measures and reviews are the start")
    ap.add_argument("--update", action="store_true",
                    help="review again the diseases of --knowledge, only what was published after their last review")
    args = ap.parse_args(argv)
    saturation = saturation_of(args.saturation)
    knowledge = Path(args.knowledge) if args.knowledge else None
    prev_measures = rs.jsonl(knowledge / "measures.jsonl") if knowledge else []
    prev_reviews = json.loads((knowledge / "reviews.json").read_text("utf-8")) if knowledge and (
        knowledge / "reviews.json").exists() else {}
    diagnoses = [d.strip() for d in args.diagnoses.split(",") if d.strip()] or (
        sorted(prev_reviews) if args.update else [])
    today = __import__("datetime").date.today().isoformat()
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    logs = {n: out.with_name(n) for n in ("screened.jsonl", "readings.jsonl", "verdicts.jsonl", "texts.jsonl",
                                          "questions.jsonl")}
    if args.resume:
        for name, path in logs.items():
            src = Path(args.resume) / name
            if src.exists() and src.resolve() != path.resolve():
                path.write_text(src.read_text("utf-8"), "utf-8")
    screened = {(r["disease"], r["id"]): r["about"] for r in rs.jsonl(logs["screened.jsonl"])}
    readings = {(r["disease"], r["text"], r["reader"]): r for r in rs.jsonl(logs["readings.jsonl"])}
    verdicts = {r["key"]: r["votes"] for r in rs.jsonl(logs["verdicts.jsonl"])}
    texts_kept = {(r["disease"], r["id"]): r for r in rs.jsonl(logs["texts.jsonl"])}
    questions = {r["key"]: r["answer"] for r in rs.jsonl(logs["questions.jsonl"])}
    read = lambda p: Path(p).read_text("utf-8", errors="replace").splitlines()  # noqa: E731
    mondo, hpo, symp = (dp.Terms.from_obo(read(args.mondo)), dp.Terms.from_obo(read(args.hpo)),
                        dp.Terms.from_obo(read(args.symp)))
    hpoa = read(args.hpoa)
    hpo_index = HpoIndex(hpo)
    # the thesaurus: the knowledge's, what a resumed run had added, and what this run adds (25 November 2026)
    thesaurus = Thesaurus()
    thesaurus.merge(Thesaurus.load(knowledge / th.FILE if knowledge else None))
    if args.resume:
        thesaurus.merge(Thesaurus.load(Path(args.resume) / th.FILE))
    from melampo.memory import literature_connectors as lc
    from melampo.memory import sources_index as si
    from melampo.memory.symptom_sources import parse_doid_owl

    mondo_lines = read(args.mondo)
    mondo_by_any_synonym = any_synonym(mondo_lines)
    mondo_replaced = replaced(mondo_lines)
    doid_symptoms = {c.doid: c.symptom_ids for c in parse_doid_owl(Path(args.doid_owl)) if not c.deprecated}
    index = si.SourcesIndex.load(si.SOURCES)
    router, lock = rs.Router(key, args.budget), threading.Lock()

    def log(name: str, row: dict) -> None:
        with lock, logs[name].open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    profiles, tests, scores, curves, summary_rows, stopped = [], {}, {}, {}, [], ""
    all_measures, reviews = [], {}
    for name in diagnoses:
        since = prev_reviews.get(name, {}).get("date", "") if args.update else ""
        exact = mondo.find(name, prefix="MONDO:")
        was, by = mondo_replaced.get(dp.norm(name), ("", ""))
        mondo_id = exact or by or mondo_by_any_synonym.get(dp.norm(name), "")
        row_mondo = "name or exact synonym" if exact else (
            f"obsolete {was}, replaced by {by}" if by else "other synonym" if mondo_id else "not found")
        xrefs = mondo.xrefs.get(mondo_id, set())
        ids = {x.replace("Orphanet:", "ORPHA:") for x in xrefs if x.startswith(("OMIM:", "Orphanet:", "ORPHA:"))}
        curated = dp.from_hpoa(hpoa, ids, hpo) + [
            m for d in sorted(x for x in xrefs if x.startswith("DOID:"))
            for m in dp.from_doid(d, doid_symptoms.get(d, ()), symp, hpo)]
        names = names_of(name, mondo, mondo_id)
        known: dict[str, str] = {}             # key -> the name the judges see
        for m in curated:
            known.setdefault(m.concept.curie if m.concept and m.concept.code else dp.norm(m.words), m.words)
        found_keys: set[str] = set()           # keys stated by a text (the curated ones count once stated)
        accepted: list[dict] = []
        # what earlier reviews accepted is the start: the weights are computed again on every measure, old and new
        for m in (x for x in prev_measures if x.get("disease") == name):
            if m.get("category") == "manifestation":
                accepted.append(m)
                known.setdefault(m["key"], m["manifestation"])
                found_keys.add(m["key"])
            elif m.get("category") == "test":
                tests.setdefault(name, []).append(m)
            elif m.get("category") == "score":
                scores.setdefault(name, []).append(m)
        curve, quiet, read_count, row = [], 0, 0, {"diagnosis": name, "mondo": mondo_id, "mondo_by": row_mondo,
                                                     "curated": len(curated),
                                                     "layers": {}, "screened_out": 0, "since": since}
        new_measures: list[dict] = []
        saturated: set[str] = set()

        def stream():
            """The texts in the order of the layers: the index's sources, then Europe PMC layer by layer."""
            try:
                if since:                # the sources of the index carry no date: read in the full review
                    raise StopIteration
                reading = lc.read_sources(name, names, index, research=args.research, get=rs.get)
                row["index"] = {"documents": len(reading.documents), "per_source": dict(reading.per_source),
                                "skipped": [f"{x.source}: {x.reason}"[:200] for x in reading.skipped][:20]}
                print(json.dumps({"disease": name, "index": row["index"]}), flush=True)
                for d in reading.documents:
                    ident = str(d.citation.get("id") or d.citation.get("pmcid") or d.citation.get("doi") or "")
                    yield "sources_index", {"id": ident or hashlib.sha1(str(d.citation).encode()).hexdigest()[:12],
                                            "read": "source", "source": dict(d.citation),
                                            "sections": [list(s) for s in d.sections]}, None
            except StopIteration:
                pass
            except Exception as error:  # noqa: BLE001
                print(json.dumps({"sources_index": f"{type(error).__name__}: {error}"[:300]}), flush=True)
            seen = set()
            for layer, _ in LAYERS:
                report: dict = {}
                for r in records(names, layer, since=since, report=report):
                    if layer in saturated:       # a saturated layer is not paged to its end
                        break
                    ident = r.get("pmcid") or r.get("pmid") or r.get("doi")
                    if not ident or ident in seen:
                        continue
                    seen.add(ident)
                    yield layer, None, r
                row.setdefault("layers_found", {})[layer] = report.get("hits")
                if report.get("unreachable"):
                    row.setdefault("layers_unreachable", []).append(layer)
                print(json.dumps({"disease": name, "layer": layer, **report}), flush=True)

        def judge(item: dict) -> list[dict]:
            measure = df.claim(name, item) if any(k in item for k in MEASURE_FIELDS[:5] + ("verbal",)) else ""
            if measure or item.get("direction"):    # what is verified is the claim with its numbers or direction
                k = JUDGED_CLAIM + json.dumps([name, dp.norm(item["manifestation"]), item["quote"],
                                               claim_of(name, item, measure)])
            else:
                k = JUDGED + json.dumps([name, dp.norm(item["manifestation"]), item["quote"]])
            if k in verdicts:
                return verdicts[k]
            votes = []
            for _, model in rs.JUDGES:
                answer, _ = router.chat(model, judge_prompt(name, item), reasoning=False, max_tokens=60)
                votes.append(read_verdict(answer))
                if votes[-1]["states"] != "yes":
                    break
            if votes and votes[-1]["states"]:
                verdicts[k] = votes
                log("verdicts.jsonl", {"key": k, "votes": votes})
            return votes

        def prepared(layer: str, record: dict, *, screen: bool = True) -> dict | None:
            """The text of a record that passes the screening, or None."""
            ident = record.get("pmcid") or record.get("pmid") or record.get("doi")
            title = str(record.get("title", ""))
            if not ident or off_topic(title, name) or not any(
                    dp.norm(n) in dp.norm(title + " " + str(record.get("abstractText", ""))) for n in names):
                row["screened_out"] += 1
                return None
            if screen and (name, ident) not in screened:
                answer, _ = router.chat(SCREENER, screen_prompt(name, record), reasoning=False, max_tokens=20)
                about = "yes" if re.search(r'"about"\s*:\s*"yes"', answer) else "no"
                screened[(name, ident)] = about
                log("screened.jsonl", {"disease": name, "id": ident, "about": about, "layer": layer,
                                       "title": title[:200]})
            if screen and screened[(name, ident)] != "yes":
                row["screened_out"] += 1
                return None
            text = texts_kept.get((name, ident)) or rs.text_of(record)
            if text is None:
                return None
            text["layer"] = layer
            if (name, text["id"]) not in texts_kept:
                texts_kept[(name, text["id"])] = text
                log("texts.jsonl", {"disease": name, **text})
            return text

        def process(layer: str, text: dict) -> int:
            """Read by the two interpreters with the one procedure of reading (``melampo.reading``: the same as for
            a patient's report), checked, judged; returns how many manifestations it added."""
            nonlocal read_count
            sections = [tuple(s) for s in text["sections"]]
            prompt = reading_prompt("publication", sections, name)

            def read_one(reader: str) -> str:
                kept = readings.get((name, text["id"], reader))
                if kept is not None and kept.get("reading") == READING and kept.get("finish") == "stop":
                    return kept["answer"]
                answer, finish = router.chat(rs.READERS[reader], prompt, reasoning=True, max_tokens=rs.READER_ROOM)
                if finish == "error":      # the provider broke off: asked once more (25 November 2026)
                    answer, finish = router.chat(rs.READERS[reader], prompt, reasoning=True,
                                                 max_tokens=rs.READER_ROOM)
                if finish != "stop":       # an answer cut off is not a reading: not kept, said
                    row.setdefault("not_read", {}).setdefault(reader, {}).setdefault(finish or "none", 0)
                    row["not_read"][reader][finish or "none"] += 1
                    return ""
                r = {"disease": name, "text": text["id"], "reader": reader, "finish": finish, "reading": READING,
                     "answer": answer}
                readings[(name, text["id"], reader)] = r
                log("readings.jsonl", r)
                return answer

            def question(reader: str):
                def chat(q: str) -> str:
                    qk = hashlib.sha1((rs.READERS[reader] + "\n" + q).encode()).hexdigest()
                    if qk not in questions:
                        room = rs.READER_ROOM if q.startswith("You and another reader") else QUESTION_ROOM
                        answer, finish = router.chat(rs.READERS[reader], q, reasoning=True, max_tokens=room)
                        if finish != "stop":     # an answer cut off is no answer: not kept
                            row.setdefault("not_answered", 0)
                            row["not_answered"] += 1
                            return ""
                        questions[qk] = answer
                        log("questions.jsonl", {"disease": name, "reader": reader, "key": qk, "answer": answer})
                    return questions[qk]
                return chat

            with ThreadPoolExecutor(2) as pool:
                answers = dict(zip(rs.READERS, pool.map(read_one, rs.READERS)))
            source_id = text["source"].get("pmid") or text["source"].get("pmcid") or text["id"]
            got = read_text(sections, "publication", {r: question(r) for r in rs.READERS}, subject=name,
                            hpo=hpo_index, thesaurus=thesaurus, answers=answers,
                            evidence={"source": str(source_id), "disease": name})
            for reader, counts in got["format"].items():
                fmt = row.setdefault("format", {}).setdefault(reader, {})
                for k, v in counts.items():
                    fmt[k] = fmt.get(k, 0) + v
            row["questions"] = row.get("questions", 0) + len(got["asked"])
            new_here = 0
            for card_ in got["cards"]:
                category = card_["category"]
                if not category:
                    row["doubtful_category"] = row.get("doubtful_category", 0) + 1
                    continue                  # the interpreters read it as different things: a doubt, not a datum
                if category == "manifestation" and card_["specific"] is not True:
                    continue                  # a list or a general phrase is not one manifestation
                concept = card_["concept"]
                kk = concept["concept"] or dp.norm(card_["name"])
                both = card_["paired_by"] != "one_interpreter"
                taken = []
                for reader in card_["readers"]:
                    own = card_["readings"][reader]
                    measure, _ = item_checked({"name": card_["name"], "quote": own["quote"],
                                               "measures": own.get("measures") or None}, sections)
                    if measure is None:
                        continue
                    m = {**measure, "manifestation": card_["name"], "words_read": card_.get("words") or "",
                         "category": category, "reader": reader, "readers": list(card_["readers"]),
                         "interpreters_agree": both and not card_["doubts"], "paired_by": card_["paired_by"],
                         "doubts": list(card_["doubts"]) + ([f"measure_refused:{measure['measure_refused']}"]
                                                            if measure.get("measure_refused") else [])}
                    m.pop("measure_refused", None)
                    if own.get("direction") and not any(x in m for x in ("with_disease", "sensitivity",
                                                                          "specificity", "lr_positive",
                                                                          "lr_negative")):
                        m["direction"] = own["direction"]
                    votes = judge(m)
                    if len(votes) != len(rs.JUDGES) or any(v["states"] != "yes" for v in votes):
                        continue
                    m["source"] = {**text["source"], "read": text["read"], "layer": layer}
                    m["disease"], m["reviewed"] = name, today
                    m["observed_by"], m["nature"] = card_["observed_by"] or "", card_["nature"] or ""
                    m["compares"] = card_.get("compares") or ""
                    m["concept"], m["code"], m["coded_by"] = concept["concept"], concept["code"], concept["how"]
                    m["description"] = {k: card_.get(k) for k in ("what", "interpretation", "threshold", "site",
                                                                  "side", "time")}
                    taken.append(m)
                if not taken:
                    continue
                if category == "manifestation":
                    if kk not in found_keys:
                        found_keys.add(kk)
                        new_here += kk not in known
                    known.setdefault(kk, card_["name"])
                    for m in taken:
                        m["key"] = kk
                    accepted.extend(taken)
                    new_measures.extend(taken)
                elif category == "test":
                    tests.setdefault(name, []).extend(taken)
                    new_measures.extend(taken)
                elif category == "score":
                    scores.setdefault(name, []).extend(taken)
                    new_measures.extend(taken)
            read_count += 1
            row["layers"][layer] = row["layers"].get(layer, 0) + 1
            curve.append({"text": text["id"], "layer": layer, "new": new_here, "known": len(known)})
            print(json.dumps({"disease": name, "text": text["id"], "layer": layer, "new": new_here,
                              "spent": round(router.cost, 4)}), flush=True)
            return new_here

        try:
            # the saturation is counted layer by layer (corrected 22 November 2026, after the first run: counted on
            # the whole review, five guideline abstracts in a row that added nothing stopped it before the studies of
            # diagnostic accuracy and the series of patients, where the weights are, had been read)
            current_layer = ""
            for layer, text, record in stream():
                if layer != current_layer:
                    current_layer, quiet = layer, 0
                if layer in saturated:
                    continue
                if record is not None:
                    text = prepared(layer, record)
                    if text is None:
                        continue
                quiet = 0 if process(layer, text) else quiet + 1
                if saturation and quiet >= saturation:
                    saturated.add(layer)
                    row.setdefault("saturated_layers", []).append(layer)
            # completeness: each curated manifestation no text has stated yet, searched with the disease
            curated_keys = {m.concept.curie if m.concept and m.concept.code else dp.norm(m.words) for m in curated}
            for kk in sorted(curated_keys - found_keys):
                words = known.get(kk, "")
                q = " OR ".join(f'TITLE_ABS:"{n}"' for n in names[:3])
                for record in rs.epmc(f'({q}) AND ABSTRACT:"{words}" AND HAS_ABSTRACT:y', args.targeted):
                    if kk in found_keys:
                        break
                    text = prepared("targeted", record, screen=False)
                    if text is not None:
                        row["targeted"] = row.get("targeted", 0) + 1
                        process("targeted", text)
        except rs.OverBudget as error:
            stopped = str(error)
        # the profile: the curated manifestations and those the texts stated, each with its sources and weight
        by_key: dict[str, list[dict]] = {}
        for m in accepted:
            by_key.setdefault(m["key"], []).append(m)
        manifestations = []
        for m in curated:
            k = m.concept.curie if m.concept and m.concept.code else dp.norm(m.words)
            if k in by_key:
                m.citations = list(m.citations) + [x["source"] for x in by_key[k]][:10]
            manifestations.append(m)
        done = {m.concept.curie if m.concept and m.concept.code else dp.norm(m.words) for m in curated}
        for k, ms in by_key.items():
            if k in done:
                continue
            first = ms[0]
            observed = next((x.get("observed_by") for x in ms if x.get("observed_by")), "")
            manifestations.append(cg.Manifestation(
                f"r{len(manifestations) + 1}", known.get(k, first["manifestation"]),
                cg.Concept("HPO", k, hpo.names.get(k, "")) if k.startswith("HP:") else (
                    cg.Concept("MELAMPO", k, known.get(k, "")) if k.startswith("MLP:") else None),
                typicality=dp.typicality_of(first["quote"]),
                # where it is found (history, examination, laboratory, imaging, ... intraoperative, histopathology):
                # the diagnosis can tell what a patient can show before surgery from what only surgery shows
                verified_by=[{"kind": observed}] if observed else [],
                nature=next((x.get("nature") for x in ms if x.get("nature")), ""),
                citations=list({(source_of(x), x["quote"]): {
                    **x["source"], "quote": x["quote"], "words_read": x.get("words_read", ""),
                    "interpreters_agree": x.get("interpreters_agree", False), "doubts": x.get("doubts", []),
                    **({"absent": True} if x.get("absent_of") else {})} for x in ms}.values())[:10]))
            done.add(k)
        profile = dp.merge_profile(mondo_id or name, name, manifestations)
        for man in profile.manifestations:
            k = man.concept.curie if man.concept and man.concept.code else dp.norm(man.words)
            records_ = by_key.get(k, [])
            measured = [x for x in records_ if not x.get("absent_of") and any(
                f in x for f in ("with_disease", "sensitivity", "specificity", "lr_positive", "lr_negative", "verbal"))]
            agreed_ms = agreed_measures(measured)
            row["measures_agreed"] = row.get("measures_agreed", 0) + len(agreed_ms)
            row["measures_one_interpreter"] = row.get("measures_one_interpreter", 0) + len(
                {(source_of(x), json.dumps(numbers_of(x), sort_keys=True)) for x in measured}) - len(agreed_ms)
            ms = for_weights(dedupe(agreed_ms), names)
            directions, alone = agreed_directions(records_)
            row["directions_agreed"] = row.get("directions_agreed", 0) + len(directions)
            row["directions_one_interpreter"] = row.get("directions_one_interpreter", 0) + alone
            if directions:
                man.direction = direction_summary(directions)
            if ms:
                w = df.weigh(ms)
                if "frequency" in w and (w.get("frequency_basis") != "verbal" or man.frequency is None):
                    man.frequency, man.frequency_basis = w["frequency"], w["frequency_basis"]
                if "accuracy" in w:
                    man.accuracy = w["accuracy"]
            elif man.frequency is not None and not man.frequency_basis:
                man.frequency_basis = "curated"
            man.evidence = dp.evidence_of(man) or man.evidence
        profiles.append(profile.as_dict())
        curves[name] = curve
        ms_ = profile.manifestations
        row.update(texts_read=read_count, stopped_by=("budget" if stopped else
                                                      "sources_unreachable" if row.get("layers_unreachable") else
                                                      "saturation" if row.get("saturated_layers") else
                                                      "end_of_sources"),
                   manifestations=len(ms_), from_texts=len(found_keys), curated_not_found=sum(
                       1 for k in known if k not in found_keys and k in {
                           m.concept.curie if m.concept and m.concept.code else dp.norm(m.words) for m in curated}),
                   weighted_measured=sum(1 for m in ms_ if m.accuracy or m.frequency_basis in (
                       "pooled_studies", "one_study", "curated")),
                   weighted_verbal=sum(1 for m in ms_ if m.frequency_basis == "verbal"),
                   typicality_only=sum(1 for m in ms_ if not m.accuracy and m.frequency is None and m.typicality),
                   without_weight=sum(1 for m in ms_ if not m.accuracy and m.frequency is None and not m.typicality),
                   cost_so_far=round(router.cost, 4))
        summary_rows.append(row)
        print(json.dumps(row), flush=True)
        all_measures += [x for x in prev_measures if x.get("disease") == name] + new_measures
        if not stopped:            # a review cut by the budget is not dated: the next run reads it again
            reviews[name] = {"date": today, "since": since, "texts_read": read_count,
                             "stopped_by": row["stopped_by"], "saturation": saturation or "none"}
        write(out, profiles, tests, scores, curves, summary_rows, saturation, stopped, router, all_measures, reviews)
        thesaurus.save(out.with_name(th.FILE))
        if stopped:
            break
    write(out, profiles, tests, scores, curves, summary_rows, saturation, stopped, router, all_measures, reviews)
    thesaurus.save(out.with_name(th.FILE))
    return 0


def direction_summary(directions: list[dict]) -> dict:
    """(B) The statements of direction of one manifestation: for present and for absent, the effect all sources
    agree on, with how many said it and whether any said it strongly; sources that disagree leave a doubt."""
    out: dict = {}
    for when in ("present", "absent"):
        said = [d for d in directions if d["when"] == when]
        if not said:
            continue
        effects = {d["effect"] for d in said}
        out[when] = {"effect": effects.pop() if len(effects) == 1 else "doubt", "sources": len(said),
                     "strong": any(w in d["words"] for d in said for w in STRONG)}
    return out


STRONG = ("strong", "markedly", "highly", "greatly", "very", "most")


def write(out: Path, profiles, tests, scores, curves, rows, saturation, stopped, router, measures=(),
          reviews=None) -> None:
    """Everything, at once after each disease: an interrupted run loses nothing. ``measures.jsonl`` holds every
    measure accepted for these diseases, old and new, from which the profile is computed again; ``reviews.json``
    the day of each review, from which the next update starts."""
    out.write_text(json.dumps({"profiles": profiles}, indent=1, ensure_ascii=False), "utf-8")
    out.with_name("measures.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in measures),
                                               "utf-8")
    out.with_name("reviews.json").write_text(json.dumps(reviews or {}, indent=1, ensure_ascii=False), "utf-8")
    group = lambda d: {k: [{"name": v[0]["manifestation"], **df.weigh(dedupe(agreed_measures(v))),  # noqa: E731
                            "measures": len(v)} for v in _by_name(ms).values()] for k, ms in d.items()}
    out.with_name("tests.json").write_text(json.dumps(group(tests), indent=1, ensure_ascii=False), "utf-8")
    out.with_name("scores.json").write_text(json.dumps(group(scores), indent=1, ensure_ascii=False), "utf-8")
    out.with_name("curves.json").write_text(json.dumps(curves, indent=1, ensure_ascii=False), "utf-8")
    summary = {"saturation": saturation or "none: to the end of the sources", "stopped": stopped,
               "cost_usd": round(router.cost, 4), "pages_not_read": dict(rs.FAILED), "rows": rows}
    out.with_name("summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), "utf-8")


def _by_name(ms: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for m in ms:
        out.setdefault(dp.norm(m["manifestation"]), []).append(m)
    return out


if __name__ == "__main__":
    sys.exit(main())
