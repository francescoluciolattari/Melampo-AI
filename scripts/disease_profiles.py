"""The pilot of the disease profiles (15 November 2026): ten diagnoses, cheapest first, the real cost measured.

    python3 scripts/disease_profiles.py --mondo mondo.obo --hpo hp.obo --hpoa phenotype.hpoa \
        --doid-owl doid.owl --symp symp.obo --out profiles.json [--budget 1.5] [--statpearls]

The ten diagnoses (``PILOT``) were written before the run, from ``source_coverage.DIAGNOSES``. For each:
the structured sources (no model), one review under CC BY or CC0 from Europe PMC (its presentation, diagnosis
and imaging sections), StatPearls only with ``--statpearls`` (CC BY-NC-ND, research only); the reading pair on
the text; the words; the two judges on what the words did not settle (``memory.disease_profiles``). OpenRouter
reports the cost of every call (``usage.cost``); the run stops before the budget is passed.

Writes the profiles, a summary, and ``review_sample.tsv``: 60 links drawn with a fixed seed from those accepted
from text, for a person to mark right or wrong (the criterion: errors under 2%).
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
OPEN_LICENCES = ("cc by", "cc0", "cc-by", "cc by 4.0")
SAMPLE_SIZE, SEED = 60, 20261115


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


def reviews_for(name: str) -> list[tuple[dict, list[tuple[str, str]]]]:
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
        licence = str(r.get("license", "")).lower()
        pmcid = r.get("pmcid", "")
        if not pmcid or pmcid in seen or not any(licence.startswith(x) for x in OPEN_LICENCES):
            continue
        seen.add(pmcid)
        sections = dp.sections_from_jats(get(f"{EPMC}/{pmcid}/fullTextXML", "application/xml"))
        size = sum(len(x) for _, x in sections)
        if size < 1500:                    # a review that says almost nothing of the presentation is skipped
            continue
        out.append(({"source": "Europe PMC review", "id": pmcid, "title": r.get("title", ""),
                     "licence": r.get("license", ""), "year": r.get("pubYear", "")}, sections))
        total += size
        if total >= MIN_SECTION_CHARACTERS or len(out) >= MAX_REVIEWS:
            break
    return out


def wikipedia_for(name: str) -> tuple[dict, list[tuple[str, str]]]:
    """The English Wikipedia article of the diagnosis (redirects followed: "acute cholecystitis" -> its article),
    its presentation, diagnosis and imaging sections, with the revision read (CC BY-SA 4.0: commercial use
    allowed, adapted text shared alike; the facts taken from it are not text)."""
    q = {"action": "parse", "page": name, "prop": "text|revid", "redirects": 1, "format": "json",
         "formatversion": 2}
    got = json.loads(get(f"{WIKIPEDIA}?{urllib.parse.urlencode(q)}") or "{}").get("parse") or {}
    sections = dp.wikipedia_sections(str(got.get("text") or ""))
    if not sections:
        return {}, []
    title = got.get("title", name)
    return ({"source": "Wikipedia", "id": f"{title} (rev {got.get('revid', '')})", "title": title,
             "url": f"https://en.wikipedia.org/w/index.php?oldid={got.get('revid', '')}",
             "licence": "CC BY-SA 4.0"}, sections)


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("mondo", "hpo", "hpoa", "doid-owl", "symp"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--diagnoses", default=",".join(PILOT))
    ap.add_argument("--budget", type=float, default=1.5, help="USD; the run stops before passing it")
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
    router = OpenRouter(key, args.budget)
    answers_dir = Path(args.out).with_name("answers")
    profiles, rows, links_all = [], [], []
    stopped = ""
    for name in [d.strip() for d in args.diagnoses.split(",") if d.strip()]:
        row = {"diagnosis": name}
        mondo_id = mondo.find(name, prefix="MONDO:")
        xrefs = mondo.xrefs.get(mondo_id, set())
        ids = {x.replace("Orphanet:", "ORPHA:") for x in xrefs if x.startswith(("OMIM:", "Orphanet:", "ORPHA:"))}
        doids = sorted(x for x in xrefs if x.startswith("DOID:"))
        parts = dp.from_hpoa(hpoa, ids, hpo)
        row["hpoa"] = len(parts)
        from_doid = [m for d in doids for m in dp.from_doid(d, doid_symptoms.get(d, ()), symp, hpo)]
        row["doid"] = len(from_doid)
        parts += from_doid
        texts = [t for t in [wikipedia_for(name)] if t[0]] + reviews_for(name)
        if args.statpearls:
            texts.append(statpearls_for(name))
        row["texts"] = [t[0].get("id", "") for t in texts if t[0]]
        verdicts: Counter = Counter()
        try:
            for source, sections in texts:
                if not sections:
                    continue
                prompt = dp.extraction_prompt(name, sections)
                readings = {}
                for r, m in READERS.items():
                    answer = router.chat(m, prompt, max_tokens=6000)
                    readings[r] = dp.read_extraction(answer)
                    answers_dir.mkdir(parents=True, exist_ok=True)       # public text read: kept to see why
                    stem = f"{re.sub(r'[^a-z0-9]+', '_', name)}__{source.get('id', '')}__{r}"
                    (answers_dir / f"{stem}.txt").write_text(answer, "utf-8")
                    row.setdefault("readers", {})[f"{source.get('id', '')}:{r}"] = {
                        **router.last, "items": len(readings[r]), "section_chars": sum(len(x) for _, x in sections)}
                names = dp.disease_words(name, mondo.synonyms.get(mondo_id, []), sections)
                links = dp.merge_readings(readings, sections, source, hpo, names)
                for link in links:
                    if link.verdict == "names":
                        # the words name the manifestation and the disease; one judge confirms that the sentence
                        # says the disease shows it (15 November 2026, fourth run: a sentence naming Bartter
                        # syndrome and nephrolithiasis, a "coincidence" between pancreatitis and TTP)
                        model, reasoning = JUDGES[CONFIRMING_JUDGE]
                        vote = dp.read_judgment(router.chat(model, dp.judge_prompt(name, link.__dict__),
                                                            reasoning=reasoning, max_tokens=60))
                        link.judges = {CONFIRMING_JUDGE: vote}
                        if vote != "yes":
                            link.verdict = ""
                    if link.verdict == "":
                        votes = {j: link.judges.get(j) or dp.read_judgment(router.chat(
                                     m, dp.judge_prompt(name, link.__dict__), reasoning=r,
                                     max_tokens=1600 if r else 60)) for j, (m, r) in JUDGES.items()}
                        link.verdict = dp.judged(link, votes)
                    verdicts[link.verdict] += 1
                accepted = [l for l in links if l.verdict in ("names", "judges")]
                parts += [dp.to_manifestation(l, hpo, i) for i, l in enumerate(accepted)]
                links_all += [{"diagnosis": name, **{k: v for k, v in l.__dict__.items() if k != "source"},
                               "source": l.source.get("id", "")} for l in links]
        except OverBudget as error:
            stopped = str(error)
        profile = dp.merge_profile(mondo_id or name, name, parts)
        profiles.append(profile.as_dict())
        row.update(mondo=mondo_id, verdicts=dict(verdicts), manifestations=len(profile.manifestations),
                   with_concept=sum(1 for m in profile.manifestations if m.concept and m.concept.code),
                   with_typicality_or_frequency=sum(1 for m in profile.manifestations
                                                    if m.typicality or m.frequency is not None),
                   cost_so_far=round(router.cost, 4))
        rows.append(row)
        print(json.dumps(row), flush=True)
        if stopped:
            break
    accepted_text = [l for l in links_all if l["verdict"] in ("names", "judges")]
    sample = random.Random(SEED).sample(accepted_text, min(SAMPLE_SIZE, len(accepted_text)))
    with open(Path(args.out).with_name("review_sample.tsv"), "w", encoding="utf-8") as f:
        f.write("diagnosis\tmanifestation\tquote\tsource\tverified_by\tright(1/0)\n")
        for l in sample:
            quote = re.sub(r"\s+", " ", l["quote"])
            f.write(f"{l['diagnosis']}\t{l['manifestation']}\t{quote}\t{l['source']}\t{l['verdict']}\t\n")
    summary = {
        "diagnoses": len(rows), "stopped": stopped, "cost_usd": round(router.cost, 4),
        "cost_per_diagnosis_usd": round(router.cost / max(1, len(rows)), 4), "calls": dict(router.calls),
        "tokens": dict(router.tokens),
        "verdicts": dict(sum((Counter(r["verdicts"]) for r in rows), Counter())),
        "manifestations": sum(r["manifestations"] for r in rows),
        "with_text": sum(1 for r in rows if r["texts"]), "rows": rows,
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "profiles": profiles, "links": links_all},
                                         indent=1, ensure_ascii=False))
    Path(args.out).with_name("profiles_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
