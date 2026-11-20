"""Which readers read the literature best: the same texts, the same judges, three configurations (20 November 2026).

    python3 scripts/literature_reader_bench.py --mondo mondo.obo --hpo hp.obo --hpoa phenotype.hpoa \
        --out out/bench.json [--budget 5]

What the readers do here is the step that fills the disease branches of the case graph: from a publication, the
manifestations of a disease (symptoms, signs, laboratory and imaging findings, complications), each with the sentence
it comes from. The owner wants the deepest reading for reports and literature alike (Nemotron 3 Ultra and Muse Spark
Open with reasoning); the run of 20 November showed Nemotron with reasoning stopping before it answered. This bench
measures, before choosing. Written before the run:

* **Configurations.** A: Nemotron 3 Ultra + Gemma 4 31B, no reasoning. B: Nemotron 3 Ultra + Muse Spark Open with
  reasoning and room enough for Nemotron to finish (``ROOM_B``). C: the same pair with the reasoning capped
  (``CAP_C``), cheaper.
* **Texts.** Five common diagnoses (``COMMON``) and the first five rare diseases, in MONDO order, with at least 15
  curated phenotypes (HPO annotations, Orphanet's included) and an open review; two open reviews each (CC0, CC BY,
  CC BY-SA), their presentation, diagnosis and imaging sections. Read once, kept, the same for every configuration.
* **Judges.** Neither is a reader: Claude Sonnet 5 first, Gemini 3.1 Pro only when the first says yes; a link is
  accepted when both say yes. Every link any configuration proposes is judged once, so no configuration is judged
  differently. No person.
* **Measures, per configuration.** Precision (accepted / proposed); coverage (the accepted manifestations it found /
  all the accepted manifestations); coverage of the curated profiles (rare diseases: curated phenotypes found);
  invented quotes (a sentence not in the text); empty or interrupted answers; cost and time per text.
* **Budget.** Reading may spend at most ``READING_SHARE`` of it, text by text for every configuration together, so
  a run that stops early still compares the same texts; the rest goes to the judges, who take the links in an order
  fixed by a hash, so a judging cut short is a sample drawn alike from every configuration.
* **Rule.** The highest coverage wins, provided no quote is invented and its precision is not more than 2 points
  below the best precision; on equal terms, the cheaper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from melampo.memory import disease_profiles as dp  # noqa: E402

COMMON = ("pulmonary embolism", "acute appendicitis", "acute cholecystitis", "acute pancreatitis", "diverticulitis")
RARE_COUNT, TEXTS_PER_DISEASE = 5, 2
NEMOTRON, GEMMA, MUSE = "nvidia/nemotron-3-ultra-550b-a55b", "google/gemma-4-31b-it", "meta/muse-spark-open"
READING_SHARE = 0.6
ROOM_B = 60000            # output tokens, reasoning included: room for Nemotron to finish
CAP_C = (6000, 16000)     # reasoning tokens, output tokens
CONFIGS = {
    "A": {"nemotron": (NEMOTRON, None, 6000), "gemma": (GEMMA, None, 6000)},
    "B": {"nemotron": (NEMOTRON, {"enabled": True, "effort": "high"}, ROOM_B),
          "muse": (MUSE, {"enabled": True, "effort": "high"}, ROOM_B)},
    "C": {"nemotron": (NEMOTRON, {"enabled": True, "max_tokens": CAP_C[0]}, CAP_C[1]),
          "muse": (MUSE, {"enabled": True, "max_tokens": CAP_C[0]}, CAP_C[1])},
}
JUDGES = (("claude-sonnet-5", "anthropic/claude-sonnet-5", None, 60),
          ("gemini-3.1-pro", "google/gemini-3.1-pro-preview", {"enabled": True, "max_tokens": 2000}, 4000))
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
CLINICAL_TITLE = ('(TITLE:"diagnosis" OR TITLE:"clinical" OR TITLE:"presentation" OR TITLE:"imaging" '
                  'OR TITLE:"evaluation" OR TITLE:"approach")')
NEEDED = ("Terms", "sections_from_jats", "extraction_prompt", "read_extraction", "quote_in", "judge_prompt",
          "read_judgment", "open_for_use", "agrees_with_curation", "norm")


class OverBudget(RuntimeError):
    pass


class Router:
    def __init__(self, key: str, budget: float):
        self.key, self.budget, self.limit, self.cost = key, budget, budget, 0.0

    def chat(self, model: str, prompt: str, reasoning: dict | None, max_tokens: int) -> dict:
        if self.cost >= self.limit:
            raise OverBudget(f"{self.cost:.4f} USD spent, limit {self.limit:.2f} of a budget of {self.budget}")
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                "max_tokens": max_tokens, "usage": {"include": True},
                "reasoning": reasoning if reasoning else {"enabled": False}}
        started, got = time.time(), {}
        for attempt in range(3):
            req = urllib.request.Request(OPENROUTER, data=json.dumps(body).encode(), method="POST",
                                         headers={"Authorization": f"Bearer {self.key}",
                                                  "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=900) as r:
                    got = json.loads(r.read())
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
                if isinstance(error, urllib.error.HTTPError) and error.code == 400 and "reasoning" in body:
                    body.pop("reasoning")
                    continue
                if attempt == 2:
                    print(f"  {model}: {error}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
        usage = got.get("usage") or {}
        cost = float(usage.get("cost") or 0)
        self.cost += cost
        choice = (got.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return {"content": str(message.get("content") or ""), "finish_reason": choice.get("finish_reason", ""),
                "reasoning_chars": len(str(message.get("reasoning") or "")), "cost": cost,
                "seconds": round(time.time() - started, 1), "tokens_out": int(usage.get("completion_tokens") or 0)}


def get(url: str, accept: str = "application/json") -> str:
    time.sleep(0.2)
    req = urllib.request.Request(url, headers={"User-Agent": "melampo-literature-bench/1.0", "Accept": accept})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read().decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return ""


def open_reviews(name: str, count: int) -> list[dict]:
    """``count`` open reviews (CC0, CC BY, CC BY-SA) with at least 1,500 characters of clinical sections, those whose
    title speaks of diagnosis or presentation first, meta-analyses and systematic reviews left out."""
    base = (f'TITLE:"{name}" AND PUB_TYPE:"review" AND OPEN_ACCESS:y AND IN_EPMC:y '
            f'AND NOT TITLE:"meta-analysis" AND NOT TITLE:"systematic"')
    results = []
    for query in (f"{base} AND {CLINICAL_TITLE}", base):
        q = {"query": query, "format": "json", "resultType": "core", "pageSize": 15, "sort": "CITED desc"}
        results += json.loads(get(f"{EPMC}/search?{urllib.parse.urlencode(q)}") or "{}").get(
            "resultList", {}).get("result", [])
    out, seen = [], set()
    for r in results:
        pmcid = r.get("pmcid", "")
        if not pmcid or pmcid in seen or not dp.open_for_use(str(r.get("license", ""))):
            continue
        seen.add(pmcid)
        sections = dp.sections_from_jats(get(f"{EPMC}/{pmcid}/fullTextXML", "application/xml"))
        if sum(len(x) for _, x in sections) < 1500:
            continue
        out.append({"id": pmcid, "title": r.get("title", ""), "licence": r.get("license", ""), "sections": sections})
        if len(out) >= count:
            break
    return out


def rare_diseases(hpoa: list[str], mondo: dp.Terms, at_least: int = 15) -> list[tuple[str, str, set[str]]]:
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
        if by_xref.get(disease):
            out.setdefault(by_xref[disease], set()).update(terms)
    return [(m, mondo.names[m], t) for m, t in sorted(out.items()) if len(t) >= at_least]


def gather_texts(hpoa: list[str], mondo: dp.Terms, cache: Path) -> list[dict]:
    """The texts of the bench, read once and kept: a second run reads the same texts."""
    if cache.exists():
        return json.loads(cache.read_text("utf-8"))
    texts = []
    for name in COMMON:
        reviews = open_reviews(name, TEXTS_PER_DISEASE)
        texts.append({"diagnosis": name, "kind": "common", "curated": [], "reviews": reviews})
        print(json.dumps({"texts": name, "reviews": [r["id"] for r in reviews]}), flush=True)
    found = 0
    for mondo_id, name, curated in rare_diseases(hpoa, mondo):
        if found >= RARE_COUNT:
            break
        reviews = open_reviews(name, TEXTS_PER_DISEASE)
        if not reviews:
            continue
        found += 1
        texts.append({"diagnosis": name, "mondo": mondo_id, "kind": "rare", "curated": sorted(curated),
                      "reviews": reviews})
        print(json.dumps({"texts": name, "reviews": [r["id"] for r in reviews], "curated": len(curated)}), flush=True)
    cache.write_text(json.dumps(texts, ensure_ascii=False), "utf-8")
    return texts


def read_all(router: Router, texts: list[dict], answers: Path) -> dict:
    """Every configuration reads every text; what each reader answered is kept. When the reading share of the budget
    is spent, the text being read is dropped for every configuration, and the texts read so far are compared."""
    readings: dict = {c: [] for c in CONFIGS}
    answers.mkdir(parents=True, exist_ok=True)
    for t in texts:
        for review in t["reviews"]:
            done = {c: len(rows) for c, rows in readings.items()}
            try:
                read_one(router, t, review, readings, answers)
            except OverBudget as error:
                for c in readings:
                    del readings[c][done[c]:]
                print(json.dumps({"reading_stopped": str(error), "text_dropped": review["id"]}), flush=True)
                return readings
    return readings


def read_one(router: Router, t: dict, review: dict, readings: dict, answers: Path) -> None:
    sections = [tuple(s) for s in review["sections"]]
    prompt = dp.extraction_prompt(t["diagnosis"], sections)
    for config, readers in CONFIGS.items():
        for reader, (model, reasoning, room) in readers.items():
            got = router.chat(model, prompt, reasoning, room)
            items = dp.read_extraction(got["content"])
            stem = f"{config}__{reader}__{review['id']}"
            (answers / f"{stem}.txt").write_text(got["content"], "utf-8")
            readings[config].append({
                "diagnosis": t["diagnosis"], "text": review["id"], "reader": reader, "items": items,
                **{k: got[k] for k in ("finish_reason", "reasoning_chars", "cost", "seconds", "tokens_out")}})
            print(json.dumps({"read": stem, "items": len(items), "finish": got["finish_reason"],
                              "cost": round(got["cost"], 4), "seconds": got["seconds"],
                              "spent": round(router.cost, 4)}), flush=True)


def link_key(hpo: dp.Terms, diagnosis: str, item: dict) -> tuple[str, str]:
    return diagnosis, hpo.find(item["manifestation"], prefix="HP:") or dp.norm(item["manifestation"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("mondo", "hpo", "hpoa"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--out", default="out/bench.json")
    args = ap.parse_args(argv)
    missing = [n for n in NEEDED if not hasattr(dp, n)]
    if missing:
        print(f"memory/disease_profiles.py lacks {missing}: the bench needs them", file=sys.stderr)
        return 1
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    read = lambda p: Path(p).read_text("utf-8", errors="replace").splitlines()  # noqa: E731
    mondo, hpo, hpoa = dp.Terms.from_obo(read(args.mondo)), dp.Terms.from_obo(read(args.hpo)), read(args.hpoa)
    texts = gather_texts(hpoa, mondo, out.with_name("texts.json"))
    router = Router(key, args.budget)
    stopped = ""
    router.limit = args.budget * READING_SHARE
    readings = read_all(router, texts, out.with_name("answers"))
    router.limit = args.budget
    cost_reading = router.cost
    read_texts = {row["text"] for rows in readings.values() for row in rows}
    sections_of = {r["id"]: [tuple(s) for s in r["sections"]] for t in texts for r in t["reviews"]}

    # every link any configuration proposes, judged once by the same two judges
    proposals: dict[str, dict] = {}
    invented: Counter = Counter()
    by_config: dict[str, set[str]] = {c: set() for c in CONFIGS}
    for config, rows in readings.items():
        for row in rows:
            for item in row["items"]:
                if not dp.quote_in(item["quote"], sections_of[row["text"]]):
                    invented[config] += 1
                    continue
                diag, key_ = link_key(hpo, row["diagnosis"], item)
                lid = json.dumps([diag, key_, dp.norm(item["quote"])])
                proposals.setdefault(lid, {"diagnosis": diag, "key": key_, "manifestation": item["manifestation"],
                                           "quote": item["quote"], "text": row["text"]})
                by_config[config].add(lid)
    verdicts: dict[str, bool] = {}
    order = sorted(proposals, key=lambda lid: hashlib.sha1(lid.encode()).hexdigest())
    try:
        for lid in order:
            p = proposals[lid]
            votes = []
            for _, model, reasoning, room in JUDGES:
                got = router.chat(model, dp.judge_prompt(p["diagnosis"], p), reasoning, room)
                votes.append(dp.read_judgment(got["content"]))
                if votes[-1] != "yes":
                    break
            verdicts[lid] = votes == ["yes"] * len(JUDGES)
            p["votes"] = votes
    except OverBudget as error:
        stopped = str(error)
    judged = set(verdicts)
    accepted = {lid for lid, ok in verdicts.items() if ok}
    accepted_keys = {tuple(json.loads(lid)[:2]) for lid in accepted}
    curated = {t["diagnosis"]: set(t["curated"]) for t in texts if t["kind"] == "rare"}
    n_reviews = len(read_texts)
    table = {}
    for config in CONFIGS:
        mine = by_config[config] & judged
        mine_ok = mine & accepted
        keys = {tuple(json.loads(lid)[:2]) for lid in mine_ok}
        rows = readings[config]
        found_curated = total_curated = 0
        for diag, terms in curated.items():
            read_terms = {k for d, k in keys if d == diag and k.startswith("HP:")}
            total_curated += len(terms)
            found_curated += sum(1 for c in terms if any(dp.agrees_with_curation(t, {c}, hpo) for t in read_terms))
        table[config] = {
            "proposed_judged": len(mine), "accepted": len(mine_ok),
            "precision": round(len(mine_ok) / len(mine), 4) if mine else None,
            "coverage": round(len(keys) / len(accepted_keys), 4) if accepted_keys else None,
            "curated_coverage": round(found_curated / total_curated, 4) if total_curated else None,
            "invented_quotes": invented[config],
            "empty_or_interrupted": sum(1 for r in rows if not r["items"] or r["finish_reason"] == "length"),
            "answers": len(rows),
            "cost_per_text_usd": round(sum(r["cost"] for r in rows) / max(1, n_reviews), 4),
            "seconds_per_text": round(sum(r["seconds"] for r in rows) / max(1, n_reviews), 1),
        }
    eligible = {c: v for c, v in table.items() if v["invented_quotes"] == 0 and v["precision"] is not None}
    best_precision = max((v["precision"] for v in eligible.values()), default=0)
    eligible = {c: v for c, v in eligible.items() if v["precision"] >= best_precision - 0.02}
    winner = sorted(eligible, key=lambda c: (-(eligible[c]["coverage"] or 0), eligible[c]["cost_per_text_usd"]))
    result = {"stopped": stopped, "judged": len(judged), "proposals": len(proposals), "accepted": len(accepted),
              "cost_usd": round(router.cost, 4), "cost_reading_usd": round(cost_reading, 4),
              "texts": [{"diagnosis": t["diagnosis"], "kind": t["kind"], "reviews": [r["id"] for r in t["reviews"]],
                         "curated": len(t["curated"])} for t in texts],
              "configurations": {c: {r: [m, bool(rs), room] for r, (m, rs, room) in v.items()}
                                 for c, v in CONFIGS.items()},
              "table": table, "winner": winner[0] if winner else None,
              "rule": "highest coverage; no invented quote; precision within 2 points of the best; then the cheaper"}
    out.write_text(json.dumps({**result, "links": list(proposals.values()), "readings": readings}, indent=1,
                              ensure_ascii=False))
    out.with_name("bench_summary.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
