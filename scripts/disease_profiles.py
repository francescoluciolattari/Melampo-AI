"""The pilot of the disease profiles (15 November 2026): every link from a publication or a curated source, the
reading measured against curated profiles, the real cost.

    python3 scripts/disease_profiles.py --mondo mondo.obo --hpo hp.obo --hpoa phenotype.hpoa \
        --doid-owl doid.owl --symp symp.obo --out profiles.json [--budget 1.5] [--gold 5] [--statpearls]

The ten diagnoses (``PILOT``) were written before the run. For each, a manifestation enters the profile only with
a source that can be checked:

1. **Curated sources, no model**: the HPO annotations (which carry Orphanet's), the Disease Ontology's axioms.
2. **Publications**: open reviews and guidelines from Europe PMC under CC0, CC BY or CC BY-SA (NC and ND left
   out), their presentation, diagnosis and imaging sections; the reading pair lists manifestations with their
   sentence; an invented sentence is refused; the judges decide what the words do not.
3. **Wikipedia only as an index**: its presentation and diagnosis sections give candidate manifestations and the
   publications they cite; a candidate enters only with a sentence of a cited publication open for use, judged
   like any other, and the citation is the publication's (PMID, PMCID, DOI). Wikipedia is never cited.

Each manifestation carries its level of evidence: ``established`` when two independent sources state it,
``reported`` when one does. No person confirms anything. The reading itself is measured against curated profiles
(``--gold N``): N rare diseases with at least 15 curated phenotypes (HPO annotations, Orphanet included) are
read from their open publications only, and every term read is compared with the curated profile (the term, one
above it, or one at most two steps below). What the curators do not have is counted: an upper bound of the
reading's error, since the curated profiles are not complete.

OpenRouter reports the cost of every call (``usage.cost``); the run stops before the budget is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from melampo.memory import disease_profiles as dp
from melampo.memory.symptom_sources import parse_doid_owl

PILOT = ("pneumonia", "pulmonary embolism", "acute appendicitis", "acute cholecystitis", "acute pancreatitis",
         "diverticulitis", "cirrhosis", "nephrolithiasis", "ischemic stroke", "aortic dissection")
READERS = {"nemotron-3-ultra": "nvidia/nemotron-3-ultra-550b-a55b", "gemma-4-31b": "google/gemma-4-31b-it"}
JUDGES = {"muse-spark-open-reasoning": ("meta/muse-spark-open", True), "claude-sonnet-5": ("anthropic/claude-sonnet-5",
                                                                                           False)}
CONFIRMING_JUDGE = "claude-sonnet-5"
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
WIKIPEDIA = "https://en.wikipedia.org/w/api.php"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MAX_CITED = 25             # publications of a Wikipedia article followed per diagnosis


class OverBudget(RuntimeError):
    pass


class OpenRouter:
    def __init__(self, key: str, budget: float):
        self.key, self.budget, self.cost, self.calls = key, budget, 0.0, Counter()
        self.tokens = Counter()
        self.last: dict = {}

    def chat(self, model: str, prompt: str, *, reasoning: bool = False, max_tokens: int = 3000) -> str:
        if self.cost >= self.budget:
            raise OverBudget(f"{self.cost:.4f} USD spent, budget {self.budget}")
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                "max_tokens": max_tokens, "usage": {"include": True},
                "reasoning": {"enabled": True, "max_tokens": 1200} if reasoning else {"enabled": False}}
        got: dict = {}
        for attempt in range(3):
            req = urllib.request.Request(OPENROUTER, data=json.dumps(body).encode(), method="POST",
                                         headers={"Authorization": f"Bearer {self.key}",
                                                  "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=240) as r:
                    got = json.loads(r.read())
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
                if isinstance(error, urllib.error.HTTPError) and error.code == 400 and "reasoning" in body:
                    body.pop("reasoning")          # a provider that does not take the reasoning switch
                    continue
                if attempt == 2:
                    print(f"  {model}: {error}", file=sys.stderr)
                    return ""
                time.sleep(5 * (attempt + 1))
        usage = got.get("usage") or {}
        self.cost += float(usage.get("cost") or 0)
        self.calls[model] += 1
        self.tokens[f"{model}:in"] += int(usage.get("prompt_tokens") or 0)
        self.tokens[f"{model}:out"] += int(usage.get("completion_tokens") or 0)
        choices = got.get("choices") or [{}]
        message = choices[0].get("message") or {}
        content = str(message.get("content") or "")
        self.last = {"model": model, "finish_reason": choices[0].get("finish_reason", ""),
                     "content_chars": len(content), "reasoning_chars": len(str(message.get("reasoning") or ""))}
        return content


def get(url: str, accept: str = "application/json") -> str:
    time.sleep(0.2)
    req = urllib.request.Request(url, headers={"User-Agent": "melampo-disease-profiles/1.0", "Accept": accept})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read().decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return ""


CLINICAL_TITLE = ('(TITLE:"diagnosis" OR TITLE:"clinical" OR TITLE:"presentation" OR TITLE:"imaging" '
                  'OR TITLE:"evaluation" OR TITLE:"approach")')


MIN_SECTION_CHARACTERS = 6000     # less than this of clinical text is not enough to read a disease from
MAX_REVIEWS = 3


def reviews_for(name: str, at_most: int = 3) -> list[tuple[dict, list[tuple[str, str]]]]:
    """Open reviews (CC BY or CC0) on the diagnosis, those whose title speaks of its diagnosis, clinical picture or
    imaging first, systematic reviews and meta-analyses left out, until their clinical sections reach
    ``MIN_SECTION_CHARACTERS`` (at most ``MAX_REVIEWS``). (Corrected 15 November 2026: the first pilot took the most
    cited review whatever it was about; the second took one review only, and four diagnoses had 500 to 1,300
    characters of clinical text, from a meta-analysis or a guideline's questions.)"""
    base = (f'TITLE:"{name}" AND PUB_TYPE:"review" AND OPEN_ACCESS:y AND IN_EPMC:y '
            f'AND (LICENSE:"cc by" OR LICENSE:"cc0") AND NOT TITLE:"meta-analysis" AND NOT TITLE:"systematic"')
    results, seen = [], set()
    for query in (f"{base} AND {CLINICAL_TITLE}", base):
        q = {"query": query, "format": "json", "resultType": "core", "pageSize": 15, "sort": "CITED desc"}
        got = json.loads(get(f"{EPMC}/search?{urllib.parse.urlencode(q)}") or "{}")
        results += got.get("resultList", {}).get("result", [])
    out, total = [], 0
    for r in results:
        pmcid = r.get("pmcid", "")
        if not pmcid or pmcid in seen or not dp.open_for_use(str(r.get("license", ""))):
            continue
        seen.add(pmcid)
        sections = dp.sections_from_jats(get(f"{EPMC}/{pmcid}/fullTextXML", "application/xml"))
        size = sum(len(x) for _, x in sections)
        if size < 1500:                    # a review that says almost nothing of the presentation is skipped
            continue
        out.append(({"source": "Europe PMC review", "id": pmcid, "pmid": r.get("pmid", ""), "doi": r.get("doi", ""),
                     "title": r.get("title", ""), "licence": r.get("license", ""), "year": r.get("pubYear", "")},
                    sections))
        total += size
        if total >= MIN_SECTION_CHARACTERS or len(out) >= at_most:
            break
    return out


def wikipedia_for(name: str) -> tuple[dict, list[tuple[str, str]], list[dict[str, str]]]:
    """The English Wikipedia article of the diagnosis (redirects followed), its presentation and diagnosis
    sections and the publications those sections cite. An index only: what enters a profile is the publication's
    sentence, never Wikipedia's."""
    q = {"action": "parse", "page": name, "prop": "text|revid", "redirects": 1, "format": "json",
         "formatversion": 2}
    got = json.loads(get(f"{WIKIPEDIA}?{urllib.parse.urlencode(q)}") or "{}").get("parse") or {}
    html = str(got.get("text") or "")
    sections = dp.wikipedia_sections(html)
    if not sections:
        return {}, [], []
    title = got.get("title", name)
    return ({"index": "Wikipedia", "title": title, "revision": got.get("revid", "")}, sections,
            dp.wikipedia_references(html))


def publication(ref: dict[str, str]) -> tuple[dict, str]:
    """A cited publication through Europe PMC: its identifiers, title, licence, and its body text when it is open
    for use (CC0, CC BY, CC BY-SA) and in PubMed Central; otherwise the reason it is not read."""
    if ref.get("pmcid"):
        query = f'PMCID:{ref["pmcid"]}'
    elif ref.get("pmid"):
        query = f'EXT_ID:{ref["pmid"]} AND SRC:MED'
    else:
        query = f'DOI:"{ref["doi"]}"'
    q = {"query": query, "format": "json", "resultType": "core", "pageSize": 1}
    hits = json.loads(get(f"{EPMC}/search?{urllib.parse.urlencode(q)}") or "{}").get("resultList", {}).get(
        "result", [])
    if not hits:
        return {**ref, "read": "not_found"}, ""
    r = hits[0]
    meta = {"source": "publication cited by Wikipedia", "id": r.get("pmcid") or r.get("pmid") or r.get("doi", ""),
            "pmid": r.get("pmid", ""), "pmcid": r.get("pmcid", ""), "doi": r.get("doi", ""),
            "title": r.get("title", ""), "journal": (r.get("journalInfo") or {}).get("journal", {}).get("title", ""),
            "year": r.get("pubYear", ""), "licence": r.get("license", "")}
    if not meta["pmcid"]:
        return {**meta, "read": "not_in_pmc"}, ""
    if not dp.open_for_use(meta["licence"]):
        return {**meta, "read": "licence_not_open"}, ""
    text = dp.body_text(get(f"{EPMC}/{meta['pmcid']}/fullTextXML", "application/xml"))
    return ({**meta, "read": "read" if text else "no_text"}, text)


def statpearls_for(name: str) -> tuple[dict, list[tuple[str, str]]]:
    q = {"db": "books", "term": f'statpearls[book] AND "{name}"[title]', "retmode": "json", "retmax": 3}
    ids = json.loads(get(f"{EUTILS}/esearch.fcgi?{urllib.parse.urlencode(q)}") or "{}").get(
        "esearchresult", {}).get("idlist", [])
    if not ids:
        return {}, []
    doc = json.loads(get(f"{EUTILS}/esummary.fcgi?db=books&retmode=json&id={ids[0]}") or "{}").get(
        "result", {}).get(ids[0], {})
    nbk = doc.get("chapteraccessionid") or doc.get("accessionid") or ""
    page = get(f"https://www.ncbi.nlm.nih.gov/books/{nbk}/?report=printable", "text/html") if nbk else ""
    sections = dp.html_sections(page, ("History and Physical", "Evaluation"))
    return ({"source": "StatPearls", "id": nbk, "title": doc.get("title", ""),
             "licence": "CC BY-NC-ND 4.0 (research only)"}, sections) if sections else ({}, [])


class Pilot:
    """The reading of one run: the judges, what each diagnosis's profile holds, every link with its verdict."""

    def __init__(self, router: OpenRouter, hpo: dp.Terms, answers_dir: Path):
        self.router, self.hpo, self.answers_dir = router, hpo, answers_dir
        self.links: list[dict] = []

    def read(self, name: str, source: dict, sections: list[tuple[str, str]], row: dict) -> dict:
        """The pair's reading of a text: the items each reader lists, kept (public text) to see why one is empty."""
        prompt = dp.extraction_prompt(name, sections)
        readings = {}
        for r, m in READERS.items():
            answer = self.router.chat(m, prompt, max_tokens=6000)
            readings[r] = dp.read_extraction(answer)
            self.answers_dir.mkdir(parents=True, exist_ok=True)
            label = source.get("id") or source.get("index", "")
            stem = re.sub(r"[^A-Za-z0-9]+", "_", f"{name}__{label}__{r}")
            (self.answers_dir / f"{stem}.txt").write_text(answer, "utf-8")
            row.setdefault("readers", {})[f"{label}:{r}"] = {
                **self.router.last, "items": len(readings[r]), "section_chars": sum(len(x) for _, x in sections)}
        return readings

    def ask(self, judge: str, name: str, link: dp.Link) -> str:
        model, reasoning = JUDGES[judge]
        return dp.read_judgment(self.router.chat(model, dp.judge_prompt(name, link.__dict__), reasoning=reasoning,
                                                 max_tokens=1600 if reasoning else 60))

    def decide(self, name: str, link: dp.Link) -> str:
        """``names`` (words, confirmed by one judge), ``judges`` (both say yes), or why not: a link enters only
        when the sentence is in the text and both readings of it agree."""
        if link.verdict == "names":
            link.judges = {CONFIRMING_JUDGE: self.ask(CONFIRMING_JUDGE, name, link)}
            if link.judges[CONFIRMING_JUDGE] == "yes":
                return "names"
            link.verdict = ""
        if link.verdict == "":
            votes = dict(link.judges)
            for j in (CONFIRMING_JUDGE, *[x for x in JUDGES if x != CONFIRMING_JUDGE]):
                if j not in votes:
                    votes[j] = self.ask(j, name, link)
                if votes[j] != "yes":
                    break
            link.judges = votes
            if len(votes) == len(JUDGES):
                return dp.judged(link, votes)
            return "one_judge_no" if "no" in votes.values() else "doubt"
        return link.verdict

    def record(self, name: str, link: dp.Link, how: str) -> None:
        self.links.append({"diagnosis": name, "how": how, **{k: v for k, v in link.__dict__.items() if k != "source"},
                           "source": link.source.get("id", ""), "licence": link.source.get("licence", "")})


def profile_of(pilot: Pilot, name: str, words: set[str], *, structured: list, wiki, reviews, row: dict) -> dp.cg.Disease:
    """One diagnosis: curated sources, then the reviews, then Wikipedia's candidates through the publications it
    cites. A manifestation already in the profile is not judged again; another source that states it in its own
    sentence (words, confirmed by one judge) corroborates it."""
    hpo, parts, verdicts = pilot.hpo, list(structured), Counter()
    in_profile: dict[str, int] = {}
    for m in structured:
        in_profile[m.concept.curie if m.concept and m.concept.code else dp.norm(m.words)] = len(in_profile)

    def key_of(manifestation: str) -> str:
        return hpo.find(manifestation, prefix="HP:") or dp.norm(manifestation)

    def take(link: dp.Link, how: str) -> bool:
        key = key_of(link.manifestation)
        if key in in_profile:
            if link.verdict == "names":            # corroboration: its own sentence, one judge
                link.judges = {CONFIRMING_JUDGE: pilot.ask(CONFIRMING_JUDGE, name, link)}
                link.verdict = "corroborated" if link.judges[CONFIRMING_JUDGE] == "yes" else "not_corroborated"
            else:
                link.verdict = "already_in_profile"
        elif link.verdict in ("names", ""):
            link.verdict = pilot.decide(name, link)
        verdicts[link.verdict] += 1
        pilot.record(name, link, how)
        if link.verdict in ("names", "judges", "corroborated"):
            parts.append(dp.to_manifestation(link, hpo, len(parts)))
            in_profile.setdefault(key, len(in_profile))
            return True
        return False

    for source, sections in reviews:
        readings = pilot.read(name, source, sections, row)
        for link in dp.merge_readings(readings, sections, source, hpo, words):
            take(link, "review")
    _, sections, refs = wiki
    if sections:
        readings = pilot.read(name, {"index": "Wikipedia"}, sections, row)
        candidates: dict[str, dict] = {}
        for reader, items in readings.items():
            for item in items:
                candidates.setdefault(key_of(item["manifestation"]), {**item, "readers": []})["readers"].append(reader)
        publications = []
        for ref in refs[:MAX_CITED]:
            pub, text = publication(ref)
            row.setdefault("cited", Counter())[pub.get("read", "")] += 1
            if text:
                publications.append((pub, text))
        for key, item in candidates.items():
            if key in in_profile:
                continue
            found = False
            for pub, text in publications:
                for sentence in dp.supporting_sentences(text, item["manifestation"], hpo, words, pub["title"])[:2]:
                    link = dp.Link(item["manifestation"], sentence, pub, readers=item["readers"],
                                   typicality={r: item["typicality"] for r in item["readers"]}, kind=item["kind"],
                                   modality=item["modality"], region=item["region"])
                    link.verdict = "names" if dp.names_the_disease(sentence, words) or dp.names_the_disease(
                        pub["title"], words) else ""
                    if take(link, "wikipedia_index"):
                        found = True
                        break
                if found:
                    break
            if not found:
                verdicts["no_publication_states_it"] += 1
    row["verdicts"] = dict(verdicts)
    return dp.merge_profile(row.get("mondo") or name, name, parts)


def rare_diseases_with_curation(hpoa: list[str], mondo: dp.Terms, at_least: int = 15) -> list[tuple[str, str, set]]:
    """(MONDO id, name, curated HPO terms) of the diseases whose curated profile (HPO annotations, Orphanet
    included) has ``at_least`` phenotypes, in a fixed order (the MONDO id), the same in every run."""
    curated: dict[str, set[str]] = {}
    for line in hpoa:
        parts = line.rstrip("\n").split("\t")
        if len(parts) > 10 and parts[10] == "P" and parts[2].strip().upper() != "NOT":
            curated.setdefault(parts[0], set()).add(parts[3])
    by_xref: dict[str, str] = {}
    for term, xrefs in mondo.xrefs.items():
        if term.startswith("MONDO:") and term in mondo.names:
            for x in xrefs:
                by_xref.setdefault(x.replace("Orphanet:", "ORPHA:"), term)
    out: dict[str, set[str]] = {}
    for disease, terms in curated.items():
        m = by_xref.get(disease)
        if m:
            out.setdefault(m, set()).update(terms)
    return [(m, mondo.names[m], t) for m, t in sorted(out.items()) if len(t) >= at_least]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("mondo", "hpo", "hpoa", "doid-owl", "symp"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--diagnoses", default=",".join(PILOT))
    ap.add_argument("--budget", type=float, default=1.5, help="USD; the run stops before passing it")
    ap.add_argument("--gold", type=int, default=5, help="rare diseases read to measure the reading (0: none)")
    ap.add_argument("--gold-share", type=float, default=0.4, help="share of the budget kept for the measure")
    ap.add_argument("--statpearls", action="store_true", help="also StatPearls (CC BY-NC-ND, research only)")
    ap.add_argument("--out", default="profiles.json")
    args = ap.parse_args(argv)
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 1
    read = lambda p: Path(p).read_text("utf-8", errors="replace").splitlines()
    mondo, hpo, symp = dp.Terms.from_obo(read(args.mondo)), dp.Terms.from_obo(read(args.hpo)), \
        dp.Terms.from_obo(read(args.symp))
    hpoa = read(args.hpoa)
    doid_symptoms = {c.doid: c.symptom_ids for c in parse_doid_owl(Path(args.doid_owl)) if not c.deprecated}
    gold_budget = args.budget * args.gold_share if args.gold else 0.0
    router = OpenRouter(key, args.budget - gold_budget)
    pilot = Pilot(router, hpo, Path(args.out).with_name("answers"))
    profiles, rows, stopped = [], [], ""
    for name in [d.strip() for d in args.diagnoses.split(",") if d.strip()]:
        row: dict = {"diagnosis": name}
        mondo_id = mondo.find(name, prefix="MONDO:")
        row["mondo"] = mondo_id
        xrefs = mondo.xrefs.get(mondo_id, set())
        ids = {x.replace("Orphanet:", "ORPHA:") for x in xrefs if x.startswith(("OMIM:", "Orphanet:", "ORPHA:"))}
        structured = dp.from_hpoa(hpoa, ids, hpo)
        row["hpoa"] = len(structured)
        from_doid = [m for d in sorted(x for x in xrefs if x.startswith("DOID:"))
                     for m in dp.from_doid(d, doid_symptoms.get(d, ()), symp, hpo)]
        row["doid"] = len(from_doid)
        try:
            reviews = reviews_for(name, at_most=2)
            if args.statpearls:
                reviews.append(statpearls_for(name))
            reviews = [r for r in reviews if r[0]]
            wiki = wikipedia_for(name)
            index = f"Wikipedia index: {wiki[0].get('title', '')}, {len(wiki[2])} cited"
            row["texts"] = [r[0]["id"] for r in reviews] + ([index] if wiki[1] else [])
            words = dp.disease_words(name, mondo.synonyms.get(mondo_id, []),
                                     [s for _, ss in reviews for s in ss] + list(wiki[1]))
            profile = profile_of(pilot, name, words, structured=structured + from_doid, wiki=wiki, reviews=reviews,
                                 row=row)
        except OverBudget as error:
            stopped = str(error)
            profile = dp.merge_profile(mondo_id or name, name, structured + from_doid)
        profiles.append(profile.as_dict())
        ms = profile.manifestations
        row.update(manifestations=len(ms), with_concept=sum(1 for m in ms if m.concept and m.concept.code),
                   established=sum(1 for m in ms if m.evidence == "established"),
                   reported=sum(1 for m in ms if m.evidence == "reported"),
                   with_typicality_or_frequency=sum(1 for m in ms if m.typicality or m.frequency is not None),
                   cost_so_far=round(router.cost, 4))
        if "cited" in row:
            row["cited"] = dict(row["cited"])
        rows.append(row)
        print(json.dumps(row), flush=True)
        if stopped:
            break
    cost_profiles = router.cost
    gold = measure(pilot, router, hpoa, mondo, hpo, args.gold, gold_budget) if args.gold else {}
    summary = {
        "diagnoses": len(rows), "stopped": stopped, "cost_usd": round(router.cost, 4),
        "cost_profiles_usd": round(cost_profiles, 4),
        "cost_per_diagnosis_usd": round(cost_profiles / max(1, len(rows)), 4), "calls": dict(router.calls),
        "tokens": dict(router.tokens),
        "verdicts": dict(sum((Counter(r.get("verdicts", {})) for r in rows), Counter())),
        "manifestations": sum(r["manifestations"] for r in rows),
        "established": sum(r["established"] for r in rows), "reported": sum(r["reported"] for r in rows),
        "reading_against_curation": {k: v for k, v in gold.items() if k != "diseases"}, "rows": rows,
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "profiles": profiles, "links": pilot.links,
                                          "gold": gold}, indent=1, ensure_ascii=False))
    Path(args.out).with_name("profiles_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    return 0


def measure(pilot: Pilot, router: OpenRouter, hpoa: list[str], mondo: dp.Terms, hpo: dp.Terms, count: int,
            budget: float) -> dict:
    """The reading measured against curated profiles: rare diseases read from their open publications only (no
    curated source in the reading), every HPO term read compared with the curated profile."""
    router.budget += budget
    diseases, read_terms, agree, outside = [], 0, 0, []
    for mondo_id, name, curated in rare_diseases_with_curation(hpoa, mondo):
        if len(diseases) >= count:
            break
        reviews = [r for r in reviews_for(name, at_most=2) if r[0]]
        if not reviews:
            continue
        row: dict = {"diagnosis": name, "mondo": mondo_id, "curated": len(curated)}
        words = dp.disease_words(name, mondo.synonyms.get(mondo_id, []), [s for _, ss in reviews for s in ss])
        try:
            profile = profile_of(pilot, name, words, structured=[], wiki=({}, [], []), reviews=reviews, row=row)
        except OverBudget as error:
            row["stopped"] = str(error)
            diseases.append(row)
            break
        terms = [m for m in profile.manifestations if m.concept and m.concept.code.startswith("HP:")]
        ok = [m for m in terms if dp.agrees_with_curation(m.concept.code, curated, hpo)]
        row.update(accepted=len(profile.manifestations), with_hpo_term=len(terms), agree=len(ok))
        read_terms += len(terms)
        agree += len(ok)
        outside += [{"diagnosis": name, "term": m.concept.code, "words": m.words,
                     "quote": (m.citations[0] if m.citations else {}).get("quote", ""),
                     "source": (m.citations[0] if m.citations else {}).get("id", "")}
                    for m in terms if m not in ok]
        diseases.append(row)
        print(json.dumps({"gold": row}), flush=True)
    return {"diseases_read": len(diseases), "terms_read": read_terms, "agree_with_curation": agree,
            "outside_curation": len(outside),
            "outside_share_upper_bound_of_error": round(len(outside) / read_terms, 4) if read_terms else None,
            "outside": outside, "diseases": diseases}


if __name__ == "__main__":
    sys.exit(main())
