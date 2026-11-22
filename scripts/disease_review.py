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
5. **Reading.** Two readers as for the reports (Nemotron 3 Ultra + Muse Spark Open, full reasoning) list every
   manifestation the text states, with its sentence, and every measure of it (frequency, sensitivity,
   specificity, likelihood ratios). The rules of ``disease_frequencies.checked``: the sentence in the text, the
   numbers in the sentence. Two judges independent of the readers (Claude Sonnet 5, then Gemma 4) both say it
   states it, say what it is (manifestation, test, score, outcome, risk factor) and which known manifestation it
   is; no person.
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

SATURATION_LEAST = 5
SCREENER = "google/gemma-4-31b-it"
CATEGORIES = ("manifestation", "test", "score", "outcome", "risk_factor")
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


def records(names: list[str], layer: str, get=None, since: str = ""):
    """Every record of one layer, page after page by Europe PMC's cursor, to the end; with ``since``, only those
    first published on that day or after."""
    get = get or rs.get
    cursor = "*"
    while True:
        q = {"query": layer_query(names, layer, since), "format": "json", "resultType": "core", "pageSize": PAGE,
             "cursorMark": cursor}
        page = json.loads(get(f"{rs.EPMC}/search?{rs.urllib.parse.urlencode(q)}") or "{}")
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


def read_prompt(disease: str, sections: list[tuple[str, str]]) -> str:
    text = "\n\n".join(f"## {t}\n{x}" for t, x in sections)
    return (f"You read a medical publication. List every MANIFESTATION of {disease} it states: symptoms, signs, "
            "laboratory values, imaging findings, complications of the disease itself; and with each, any MEASURE "
            "it gives of how often it is present.\nFor each answer an object with: manifestation (its name), "
            "with_disease {count, total, percent}, sensitivity, specificity, lr_positive, lr_negative (as written, "
            "null when absent), comparator (the patients without the disease the specificity was measured in), "
            "setting (emergency department, primary care, hospital, surgical, population, other), patients, "
            "design (meta_analysis, systematic_review, prospective_cohort, retrospective_cohort, cross_sectional, "
            "case_control, case_series, case_report, review, guideline, other), pooled (true for a meta-analysis's "
            "pooled value), verbal (a word of frequency such as \"often\" when the text gives no number, else \"\"), "
            "and quote: ONE sentence or ONE table row copied word for word from the text, that states it.\n"
            "List a manifestation even when the text gives no measure of it. Do not compute anything the text does "
            "not state. Answer only a JSON list, [] when there is none.\n\n" + text)


def judge_prompt(disease: str, item: dict, known: list[str]) -> str:
    measure = df.claim(disease, item) if any(k in item for k in ("with_disease", "sensitivity", "specificity",
                                                                 "lr_positive", "lr_negative", "verbal")) else ""
    names = "\n".join(f"- {n}" for n in known[:400])
    return (f"Disease: {disease}\nStated: {item['manifestation']}\nSentence (or table row) from a published text: "
            f"\"{item['quote']}\"\nClaim: {item['manifestation']} is a manifestation of {disease}"
            + (f"; {measure}" if measure else "") + ".\n\n"
            "1. states: yes when the sentence states it (and the numbers, if any) for this disease in general; no "
            "when it does not, when it belongs to another disease, to one particular cause or form only, or to a "
            "group the claim does not name.\n"
            "2. category: manifestation (a symptom, sign, laboratory value or imaging finding of the patient, or a "
            "complication of the disease itself), test (how well an exam or assay detects the disease), score (a "
            "clinical score or rule), outcome (death, a complication of treatment, recurrence), risk_factor (what "
            "makes the disease more likely, not what it shows).\n"
            "3. same_as: the name in the list below that is the same manifestation, copied exactly, or \"none\".\n"
            f"Known manifestations of {disease}:\n{names}\n\n"
            'Answer only JSON: {"states": "yes" or "no", "category": "...", "same_as": "..."}.')


def read_verdict(answer: str) -> dict:
    v = df.read_verdict(answer)
    m = re.search(r'"category"\s*:\s*"([a-z_]+)"', answer or "")
    if m and m.group(1) in CATEGORIES:
        v["category"] = m.group(1)
    return v


def item_checked(item: dict, sections: list[tuple[str, str]]) -> tuple[dict | None, str]:
    """A manifestation with or without a measure, by the rules fixed before the run."""
    quote = " ".join(str(item.get("quote", "")).split())
    name = str(item.get("manifestation", "")).strip()
    if not name or not quote or not df.in_text(quote, sections):
        return None, "quote_not_in_text"
    m, why = df.checked(item, sections)
    if m is not None:
        return m, ""
    if why == "no_measure":       # a manifestation stated without a measure is a manifestation
        return {"manifestation": name, "quote": quote, "setting": str(item.get("setting", "") or "not stated").lower(),
                "comparator": "", "design": str(item.get("design", "other")).lower(), "pooled": False,
                "patients": None}, ""
    return None, why


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
    logs = {n: out.with_name(n) for n in ("screened.jsonl", "readings.jsonl", "verdicts.jsonl", "texts.jsonl")}
    if args.resume:
        for name, path in logs.items():
            src = Path(args.resume) / name
            if src.exists() and src.resolve() != path.resolve():
                path.write_text(src.read_text("utf-8"), "utf-8")
    screened = {(r["disease"], r["id"]): r["about"] for r in rs.jsonl(logs["screened.jsonl"])}
    readings = {(r["disease"], r["text"], r["reader"]): r for r in rs.jsonl(logs["readings.jsonl"])}
    verdicts = {r["key"]: r["votes"] for r in rs.jsonl(logs["verdicts.jsonl"])}
    texts_kept = {(r["disease"], r["id"]): r for r in rs.jsonl(logs["texts.jsonl"])}
    read = lambda p: Path(p).read_text("utf-8", errors="replace").splitlines()  # noqa: E731
    mondo, hpo, symp = (dp.Terms.from_obo(read(args.mondo)), dp.Terms.from_obo(read(args.hpo)),
                        dp.Terms.from_obo(read(args.symp)))
    hpoa = read(args.hpoa)
    from melampo.memory import literature_connectors as lc
    from melampo.memory import sources_index as si
    from melampo.memory.symptom_sources import parse_doid_owl

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
        mondo_id = mondo.find(name, prefix="MONDO:") or ""
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
        curve, quiet, read_count, row = [], 0, 0, {"diagnosis": name, "mondo": mondo_id, "curated": len(curated),
                                                     "layers": {}, "screened_out": 0, "since": since}
        new_measures: list[dict] = []

        def stream():
            """The texts in the order of the layers: the index's sources, then Europe PMC layer by layer."""
            try:
                if since:                # the sources of the index carry no date: read in the full review
                    raise StopIteration
                reading = lc.read_sources(name, names, index, research=args.research, get=rs.get)
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
                for r in records(names, layer, since=since):
                    ident = r.get("pmcid") or r.get("pmid") or r.get("doi")
                    if not ident or ident in seen:
                        continue
                    seen.add(ident)
                    yield layer, None, r

        def judge(item: dict) -> list[dict]:
            k = df.VERDICTS + json.dumps([name, dp.norm(item["manifestation"]), item["quote"]])
            if k in verdicts:
                return verdicts[k]
            votes = []
            for _, model in rs.JUDGES:
                answer, _ = router.chat(model, judge_prompt(name, item, sorted(set(known.values()))),
                                        reasoning=False, max_tokens=200)
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
            """Read by the two readers, checked, judged; returns how many manifestations it added."""
            nonlocal read_count
            sections = [tuple(s) for s in text["sections"]]

            def read_one(reader: str) -> dict | None:
                kept = readings.get((name, text["id"], reader))
                if kept is not None:
                    return kept
                answer, finish = router.chat(rs.READERS[reader], read_prompt(name, sections), reasoning=True,
                                             max_tokens=rs.READER_ROOM)
                if not finish:
                    return None
                r = {"disease": name, "text": text["id"], "reader": reader, "finish": finish,
                     "items": rs.read_items(answer)}
                readings[(name, text["id"], reader)] = r
                log("readings.jsonl", r)
                return r

            with ThreadPoolExecutor(2) as pool:
                got = [f.result() for f in [pool.submit(read_one, r) for r in rs.READERS]]
            items: dict[str, dict] = {}
            for r in got:
                for it in (r or {}).get("items", []):
                    m, _ = item_checked(it, sections)
                    if m is None:
                        continue
                    k = json.dumps([dp.norm(m["manifestation"]), m["quote"]])
                    items.setdefault(k, {**m, "readers": []})["readers"].append(r["reader"])
            new_here = 0
            for m in items.values():
                votes = judge(m)
                if len(votes) != len(rs.JUDGES) or any(v["states"] != "yes" for v in votes):
                    continue
                cats = {v["category"] for v in votes}
                if len(cats) != 1 or "" in cats:
                    continue
                m["category"] = cats.pop()
                m["source"] = {**text["source"], "read": text["read"], "layer": layer}
                m["disease"], m["reviewed"] = name, today
                same = {v["same_as"] for v in votes}
                same_name = same.pop() if len(same) == 1 else ""
                if m["category"] == "manifestation":
                    kk = next((k for k, v in known.items() if v == same_name), "") if same_name else ""
                    kk = kk or hpo.find(m["manifestation"], prefix="HP:") or dp.norm(m["manifestation"])
                    if kk not in found_keys:
                        found_keys.add(kk)
                        new_here += kk not in known
                    known.setdefault(kk, m["manifestation"])
                    m["key"] = kk
                    accepted.append(m)
                    new_measures.append(m)
                elif m["category"] == "test":
                    tests.setdefault(name, []).append(m)
                    new_measures.append(m)
                elif m["category"] == "score":
                    scores.setdefault(name, []).append(m)
                    new_measures.append(m)
            read_count += 1
            row["layers"][layer] = row["layers"].get(layer, 0) + 1
            curve.append({"text": text["id"], "layer": layer, "new": new_here, "known": len(known)})
            print(json.dumps({"disease": name, "text": text["id"], "layer": layer, "new": new_here,
                              "spent": round(router.cost, 4)}), flush=True)
            return new_here

        try:
            for layer, text, record in stream():
                if saturation and quiet >= saturation:
                    break
                if record is not None:
                    text = prepared(layer, record)
                    if text is None:
                        continue
                quiet = 0 if process(layer, text) else quiet + 1
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
            manifestations.append(cg.Manifestation(
                f"r{len(manifestations) + 1}", known.get(k, first["manifestation"]),
                cg.Concept("HPO", k, known.get(k, "")) if k.startswith("HP:") else None,
                typicality=dp.typicality_of(first["quote"]),
                citations=[{**x["source"], "quote": x["quote"]} for x in ms][:10]))
            done.add(k)
        profile = dp.merge_profile(mondo_id or name, name, manifestations)
        for man in profile.manifestations:
            k = man.concept.curie if man.concept and man.concept.code else dp.norm(man.words)
            ms = [x for x in by_key.get(k, []) if any(f in x for f in ("with_disease", "sensitivity", "specificity",
                                                                      "lr_positive", "lr_negative", "verbal"))]
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
        row.update(texts_read=read_count, stopped_by=("saturation" if saturation and quiet >= saturation else
                                                      "budget" if stopped else "end_of_sources"),
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
        if stopped:
            break
    write(out, profiles, tests, scores, curves, summary_rows, saturation, stopped, router, all_measures, reviews)
    return 0


def write(out: Path, profiles, tests, scores, curves, rows, saturation, stopped, router, measures=(),
          reviews=None) -> None:
    """Everything, at once after each disease: an interrupted run loses nothing. ``measures.jsonl`` holds every
    measure accepted for these diseases, old and new, from which the profile is computed again; ``reviews.json``
    the day of each review, from which the next update starts."""
    out.write_text(json.dumps({"profiles": profiles}, indent=1, ensure_ascii=False), "utf-8")
    out.with_name("measures.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in measures),
                                               "utf-8")
    out.with_name("reviews.json").write_text(json.dumps(reviews or {}, indent=1, ensure_ascii=False), "utf-8")
    group = lambda d: {k: [{"name": v[0]["manifestation"], **df.weigh(v), "measures": len(v)}  # noqa: E731
                           for v in _by_name(ms).values()] for k, ms in d.items()}
    out.with_name("tests.json").write_text(json.dumps(group(tests), indent=1, ensure_ascii=False), "utf-8")
    out.with_name("scores.json").write_text(json.dumps(group(scores), indent=1, ensure_ascii=False), "utf-8")
    out.with_name("curves.json").write_text(json.dumps(curves, indent=1, ensure_ascii=False), "utf-8")
    summary = {"saturation": saturation or "none: to the end of the sources", "stopped": stopped,
               "cost_usd": round(router.cost, 4), "rows": rows}
    out.with_name("summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), "utf-8")


def _by_name(ms: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for m in ms:
        out.setdefault(dp.norm(m["manifestation"]), []).append(m)
    return out


if __name__ == "__main__":
    sys.exit(main())
