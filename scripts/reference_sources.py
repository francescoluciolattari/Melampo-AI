"""The reference library from primary, open sources (21 November 2026): points 3 and 4 of the library.

    python3 scripts/reference_sources.py --out out/reference.json [--budget 5] [--resume previous]

**Point 3: Radiopaedia as an index.** Radiopaedia is written by its users, like Wikipedia: by the owner's rule a
value rests on the publication it cites, not on the page. For every row of ``data/reference/values.yml`` taken
from Radiopaedia (and the rows in doubt), the page's reference list gives the publication its number cites (DOI,
PMID); Europe PMC gives its text (the full text when its licence is CC0, CC BY or CC BY-SA, otherwise the
abstract); the readers find the sentence with the value. Verdicts per row: ``confirmed`` (the primary states the
row's numbers), ``different`` (it states other numbers for the same measure: the row goes in doubt with them),
``not_found``, ``no_reference``.

**Point 4: every measure.** Every structure Melampo reads (the classes of ``data/linking/anatomy_lexicon.json``,
sides and numbered parts joined) and the structures reports measure that the lexicon does not have yet
(``EXTRA``: appendix, bile duct, lymph nodes...); for each, the open reviews and guidelines of Europe PMC on its
normal size, read for every reference value they state.

The method is the one of the reports and of the disease profiles (owner's choice, 20 November 2026): Nemotron 3
Ultra and Muse Spark Open, both with their own reasoning, uncapped, 60,000 tokens of room. Fixed before the run:

* a value is kept only when its sentence is in the text, word for word (a table row counts as a sentence), its
  numbers are in that sentence, its unit is one the library compares (mm, cm, mL, cm3, mL/m2), and it is a limit,
  a range or a threshold: a mean with its deviation is not a limit and is recorded as such, never as a row;
* two judges independent of the readers (Claude Sonnet 5, then Gemma 4 when the first says yes) must both say the
  sentence states that limit for that structure, technique and population; no person;
* every candidate keeps its source (title, DOI, PMCID, licence, section) and the readers that proposed it.

The run writes as it goes, stops before the budget and at once when OpenRouter has no credit (402), and resumes
(``--resume``): what was read is not read again, what was judged is not judged again. Nothing is written into
``values.yml`` here: the candidates and the primary checks are the artifact; ``--merge`` (next step) puts them in.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from melampo.memory import disease_profiles as dp  # noqa: E402
from melampo.memory import reference_values as rv  # noqa: E402

READERS = {"nemotron": "nvidia/nemotron-3-ultra-550b-a55b", "muse": "meta/muse-spark-open"}
READER_ROOM = 60000
JUDGES = (("claude-sonnet-5", "anthropic/claude-sonnet-5"), ("gemma-4-31b", "google/gemma-4-31b-it"))
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
TEXTS_PER_STRUCTURE = 2
TEXT_CHARACTERS = 40000
WORKERS = 4
LIMIT_KINDS = ("upper_limit", "lower_limit", "range", "threshold", "percentile")
UNITS = {"mm": ("mm", 1.0), "cm": ("mm", 10.0), "ml": ("mL", 1.0), "cm3": ("mL", 1.0), "cm³": ("mL", 1.0),
         "cc": ("mL", 1.0), "ml/m2": ("mL/m2", 1.0), "ml/m²": ("mL/m2", 1.0)}

# structures reports measure that the lexicon does not have yet: their rows wait for the lexicon to know them
EXTRA = ("appendix", "common bile duct", "pancreatic duct", "lymph node", "pulmonary artery", "endometrium",
         "uterus", "ovary", "testis", "renal pelvis", "ureter", "thymus", "pericardium", "optic nerve sheath",
         "lateral ventricle", "pituitary gland", "spinal canal", "seminal vesicle", "parotid gland",
         "submandibular gland", "carotid intima-media", "pylorus", "bowel wall", "urinary bladder wall")
SKIP = re.compile(r"^(kidney_cyst|liver_segment)")
NAMES = {"portal_vein_and_splenic_vein": ("portal vein", "splenic vein"), "autochthon": ("paraspinal muscles",),
         "iliopsoas": ("psoas muscle",), "iliac_vena": ("iliac vein",), "clavicula": ("clavicle",),
         "hip": ("hip joint",), "atrial_appendage": ("left atrial appendage",), "costal_cartilages": ("costal cartilage",),
         "heart": ("heart", "left ventricle", "right ventricle", "left atrium", "right atrium"),
         "aorta": ("thoracic aorta", "abdominal aorta", "aortic root"), "brain": ("brain",),
         "vertebrae_C": ("cervical spine",), "vertebrae_T": ("thoracic spine",), "vertebrae_L": ("lumbar spine",),
         "vertebrae_S": ("sacrum",)}


def structures(lexicon: Path) -> list[dict]:
    """The structures to search: the lexicon's classes with sides and numbers joined, then ``EXTRA``."""
    classes = json.loads(lexicon.read_text("utf-8"))["classes"]
    groups: dict[str, list[str]] = {}
    for cid in sorted(classes):
        if SKIP.match(cid):
            continue
        key = re.sub(r"_(left|right)(_\d+)?$", "", cid)
        key = re.sub(r"^(vertebrae_[CTLS])\d+$", r"\1", key)
        key = re.sub(r"^lung_.*", "lung", key)
        groups.setdefault(key, []).append(cid)
    out = [{"key": k, "cids": v, "names": list(NAMES.get(k, (k.replace("_", " "),)))} for k, v in groups.items()]
    out += [{"key": x.replace(" ", "_"), "cids": [], "names": [x]} for x in EXTRA]
    return out


class OverBudget(RuntimeError):
    pass


class Router:
    def __init__(self, key: str, budget: float):
        self.key, self.budget, self.cost, self.lock = key, budget, 0.0, threading.Lock()

    def chat(self, model: str, prompt: str, *, reasoning: bool, max_tokens: int) -> tuple[str, str]:
        with self.lock:
            if self.cost >= self.budget:
                raise OverBudget(f"{self.cost:.4f} USD spent, budget {self.budget}")
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                "max_tokens": max_tokens, "usage": {"include": True},
                "reasoning": {"enabled": True, "effort": "high"} if reasoning else {"enabled": False}}
        got: dict = {}
        for attempt in range(3):
            req = urllib.request.Request(OPENROUTER, data=json.dumps(body).encode(), method="POST",
                                         headers={"Authorization": f"Bearer {self.key}",
                                                  "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=900) as r:
                    got = json.loads(r.read())
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
                code = getattr(error, "code", 0)
                if code in (401, 402):
                    raise OverBudget(f"OpenRouter answered {code}: no credit or no key") from error
                if code == 400 and "reasoning" in body:
                    body.pop("reasoning")
                    continue
                if attempt == 2:
                    print(f"  {model}: {error}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
        with self.lock:
            self.cost += float((got.get("usage") or {}).get("cost") or 0)
        choice = (got.get("choices") or [{}])[0]
        return str((choice.get("message") or {}).get("content") or ""), str(choice.get("finish_reason") or "")


GET_PAUSES = (0, 5, 20, 60)   # seconds before each attempt when a server does not answer, or asks to wait
FAILED: dict[str, int] = {}   # host and reason -> pages that could not be read, for the summary of a run


def get(url: str, accept: str = "application/json", pauses=GET_PAUSES, wait=time.sleep) -> str:
    """A page, or "" when it is not there. A server that does not answer (a timeout, a connection broken, 429, a
    5xx) is asked again before giving up; every page given up is counted in ``FAILED`` (24 November 2026,
    declared after the third and fourth runs of the review: a page not answered once was taken for no page, and
    whole layers and the sources of the index gave nothing without a trace)."""
    host = urllib.parse.urlsplit(url).netloc
    why = ""
    for pause in pauses:
        wait(max(pause, 0.2))
        req = urllib.request.Request(url, headers={"User-Agent": "melampo-reference-sources/1.0", "Accept": accept})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as error:
            why = f"http_{error.code}"
            if error.code != 429 and error.code < 500:
                break                      # not there: asking again changes nothing
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            why = type(error).__name__
    FAILED[f"{host} {why}"] = FAILED.get(f"{host} {why}", 0) + 1
    return ""


# -- texts ----------------------------------------------------------------------------------------------------------

def _plain(el) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def text_from_jats(xml: str, limit: int = TEXT_CHARACTERS) -> list[tuple[str, str]]:
    """The sections of an article and its tables, a table row as one line of cells ("a | b | c"), in order."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    out, used = [], 0
    abstract = root.find(".//abstract")
    if abstract is not None:
        out.append(("Abstract", _plain(abstract)))
        used += len(out[-1][1])
    body = root.find(".//body")
    for sec in body.iter("sec") if body is not None else []:
        paras = [_plain(p) for p in sec.findall("p")]
        if paras:
            title = _plain(sec.find("title"))
            out.append((title, " ".join(paras)))
            used += len(out[-1][1])
        if used > limit:
            break
    for table in root.iter("table-wrap"):
        caption = _plain(table.find(".//caption")) or _plain(table.find("label"))
        rows = [" | ".join(_plain(c) for c in tr if c.tag in ("td", "th")) for tr in table.iter("tr")]
        if rows and used < limit:
            out.append((f"Table: {caption}", "\n".join(rows)))
            used += len(out[-1][1])
    return out


def epmc(query: str, size: int = 15) -> list[dict]:
    q = {"query": query, "format": "json", "resultType": "core", "pageSize": size, "sort": "CITED desc"}
    return json.loads(get(f"{EPMC}/search?{urllib.parse.urlencode(q)}") or "{}").get("resultList", {}).get("result", [])


def text_of(record: dict) -> dict | None:
    """The full text when its licence allows (CC0, CC BY, CC BY-SA), otherwise the abstract."""
    pmcid, licence = record.get("pmcid", ""), str(record.get("license", ""))
    source = {"title": record.get("title", ""), "doi": record.get("doi", ""), "pmid": record.get("pmid", ""),
              "pmcid": pmcid, "licence": licence or "abstract only", "year": record.get("pubYear", "")}
    if pmcid and dp.open_for_use(licence):
        sections = text_from_jats(get(f"{EPMC}/{pmcid}/fullTextXML", "application/xml"))
        if sum(len(t) for _, t in sections) > 800:
            return {"id": pmcid, "read": "full_text", "source": source, "sections": sections}
    abstract = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", record.get("abstractText", ""))).split())
    if abstract:
        return {"id": record.get("pmid") or record.get("doi") or pmcid, "read": "abstract", "source": source,
                "sections": [("Abstract", abstract)]}
    return None


def reviews_for(name: str, count: int) -> list[dict]:
    words = ('("normal size" OR "normal dimensions" OR "normal values" OR "reference values" OR '
             '"reference ranges" OR "normal measurements" OR "upper limit of normal" OR "normal diameter" OR '
             '"normal thickness")')
    base = f'TITLE_ABS:"{name}" AND {words} AND OPEN_ACCESS:y AND IN_EPMC:y'
    out, seen = [], set()
    for query in (f'{base} AND (PUB_TYPE:"review" OR TITLE:"guideline" OR TITLE:"recommendations")', base):
        for r in epmc(query):
            if len(out) >= count:
                return out
            pmcid = r.get("pmcid", "")
            if not pmcid or pmcid in seen or not dp.open_for_use(str(r.get("license", ""))):
                continue
            seen.add(pmcid)
            t = text_of(r)
            if t and t["read"] == "full_text":
                out.append(t)
    return out


def radiopaedia_references(url: str) -> dict[str, dict]:
    """The numbered references of a Radiopaedia page: number -> DOI, PMID."""
    page = get(url, "text/html")
    out: dict[str, dict] = {}
    block = re.search(r'(?is)<(ol|div)[^>]*class="[^"]*references[^"]*"[^>]*>(.*?)</\1>', page)
    items = re.findall(r"(?is)<li[^>]*>(.*?)</li>", block.group(2) if block else page)
    for n, item in enumerate(items, 1):
        doi = re.search(r"doi\.org/(10\.[^\s\"'<>]+)", item)
        pmid = re.search(r"pubmed(?:\.ncbi\.nlm\.nih\.gov)?/(\d+)", item)
        if doi or pmid:
            out[str(n)] = {"doi": html.unescape(doi.group(1)).rstrip(".") if doi else "",
                           "pmid": pmid.group(1) if pmid else "",
                           "words": " ".join(html.unescape(re.sub(r"<[^>]+>", " ", item)).split())[:300]}
    return out


def primary_text(ref: dict) -> dict | None:
    query = f'EXT_ID:{ref["pmid"]} AND SRC:MED' if ref.get("pmid") else f'DOI:"{ref["doi"]}"'
    found = epmc(query, 1)
    return text_of(found[0]) if found else None


# -- reading --------------------------------------------------------------------------------------------------------

def read_prompt(structure: str, sections: list[tuple[str, str]], focus: str = "") -> str:
    text = "\n\n".join(f"## {t}\n{x}" for t, x in sections)
    ask = f" Look in particular for: {focus}." if focus else ""
    return (f"You read a medical publication. List every REFERENCE VALUE it states for the {structure}: an upper or "
            "lower limit of normal, a normal range, a percentile used as a limit, or a threshold that defines an "
            f"abnormality (for example \"a diameter > 6 mm defines ...\").{ask}\n"
            "For each value answer an object with: aspect (length, diameter, thickness, width, depth, volume, "
            "index...), detail (how and where it is measured), technique (US, CT, MR, X-ray, echo, any), "
            "population {age: adult|child|any, sex: male|female|any}, kind (upper_limit, lower_limit, range, "
            "threshold, percentile, mean_sd), low, high (numbers as written, null when absent), unit (as written), "
            "above_means and below_means (the words the text uses for a value beyond the limit, else \"\"), and "
            "quote: ONE sentence, or ONE row of a table, copied word for word from the text, containing the numbers.\n"
            "Do not convert units. Do not add a value the text does not state. A mean with its standard deviation "
            "is kind mean_sd, not a limit. Answer only a JSON list, [] when there is none.\n\n" + text)


def read_items(answer: str) -> list[dict]:
    m = re.search(r"\[.*\]", answer or "", re.DOTALL)
    try:
        items = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []
    return [i for i in items if isinstance(i, dict) and str(i.get("quote", "")).strip()]


def _numbers(text: str) -> list[float]:
    return [float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", text or "")]


def as_value(item: dict, sections: list[tuple[str, str]]) -> tuple[dict | None, str]:
    """The item as a value of the library, or None and why not (fixed before the run)."""
    if not dp.quote_in(str(item["quote"]), sections):
        return None, "quote_not_in_text"
    kind = str(item.get("kind", "")).strip().lower()
    if kind not in LIMIT_KINDS:
        return None, "mean_sd" if kind == "mean_sd" else f"not_a_limit:{kind}"
    unit = re.sub(r"\s+", "", str(item.get("unit", "")).lower()).replace("milliliters", "ml").replace("millimetres", "mm")
    if unit not in UNITS:
        return None, f"unit_not_compared:{unit}"
    row_unit, factor = UNITS[unit]
    numbers = _numbers(item["quote"])
    values = {}
    for side in ("low", "high"):
        v = item.get(side)
        if v in (None, "", "null"):
            continue
        try:
            v = float(str(v).replace(",", "."))
        except ValueError:
            return None, "number_not_read"
        if not any(abs(v - n) < 1e-9 for n in numbers):
            return None, "number_not_in_quote"
        values[side] = round(v * factor, 4)
    if not values:
        return None, "no_number"
    pop = item.get("population") if isinstance(item.get("population"), dict) else {}
    return {"aspect": str(item.get("aspect", "")).strip().lower(), "detail": str(item.get("detail", "")).strip(),
            "technique": str(item.get("technique", "any")).strip() or "any",
            "population": {"age": str(pop.get("age", "any")), "sex": str(pop.get("sex", "any"))},
            "kind": kind, **values, "unit": row_unit, "above_means": str(item.get("above_means", "")),
            "below_means": str(item.get("below_means", "")), "quote": " ".join(str(item["quote"]).split())}, ""


def judge_prompt(structure: str, v: dict) -> str:
    limit = " to ".join(str(v[s]) for s in ("low", "high") if s in v)
    return (f"Structure: {structure}\nSentence (or table row) from a published text: \"{v['quote']}\"\n"
            f"Claim: the sentence states a reference value for the {structure}: {v['aspect']} ({v['detail']}), "
            f"{v['kind'].replace('_', ' ')} {limit} {v['unit']}, by {v['technique']}, in {v['population']['age']} "
            f"patients ({v['population']['sex']}).\nAnswer no when the numbers are a mean or a measurement of a "
            "group of patients rather than a limit of normal or a defining threshold, when they belong to another "
            "structure, technique or population, or when the sentence does not state them.\n"
            'Answer only JSON: {"states": "yes"} or {"states": "no"}.')


def key_of(v: dict, structure: str) -> str:
    return json.dumps([structure, v["aspect"], v["technique"].lower(), v["population"], v.get("low"), v.get("high"),
                       v["unit"]], sort_keys=True)


# -- run ------------------------------------------------------------------------------------------------------------

def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()] if path.exists() else []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/reference.json")
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--values", default=str(ROOT / "data" / "reference" / "values.yml"))
    ap.add_argument("--lexicon", default=str(ROOT / "data" / "linking" / "anatomy_lexicon.json"))
    ap.add_argument("--only", default="", help="primary | structures (default both)")
    args = ap.parse_args(argv)
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    readings_log, verdicts_log = out.with_name("readings.jsonl"), out.with_name("verdicts.jsonl")
    previous = Path(args.resume) if args.resume else None
    if previous:
        for name in ("readings.jsonl", "verdicts.jsonl", "texts.jsonl"):
            src = previous / name
            if src.exists() and src.resolve() != out.with_name(name).resolve():
                out.with_name(name).write_text(src.read_text("utf-8"), "utf-8")
    done = {(r["task"], r["text"], r["reader"]) for r in jsonl(readings_log)}
    judged = {r["key"]: r["votes"] for r in jsonl(verdicts_log)}
    texts_log = out.with_name("texts.jsonl")
    texts = {r["task"]: r for r in jsonl(texts_log)}
    router = Router(key, args.budget)
    lock = threading.Lock()

    def log(path: Path, row: dict) -> None:
        with lock, path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # the tasks: one per row to check in its primary (point 3), one per structure (point 4)
    import yaml

    rows = yaml.safe_load(Path(args.values).read_text("utf-8"))["rows"]
    tasks = []
    if args.only in ("", "primary"):
        for r in rows:
            if r["source"].get("publisher") == "Radiopaedia" or r.get("status") == "doubt":
                tasks.append({"task": f"primary:{r['id']}", "kind": "primary", "row": r,
                              "structure": " ".join(r["structures"]).replace("_", " ")})
    if args.only in ("", "structures"):
        for s in structures(Path(args.lexicon)):
            tasks.append({"task": f"structure:{s['key']}", "kind": "structure", "group": s,
                          "structure": " / ".join(s["names"])})

    def gather(task: dict) -> dict:
        if task["task"] in texts:
            return texts[task["task"]]
        found = {"task": task["task"], "texts": [], "notes": []}
        if task["kind"] == "primary":
            r = task["row"]
            refs = radiopaedia_references(r["source"].get("url", "")) if r["source"].get("url") else {}
            # check_ref: the reference a revised page cites now for this measure (21 November 2026)
            wanted = [x.strip() for x in str(r.get("check_ref") or r["source"].get("ref", "")).split(",") if x.strip()]
            found["refs"] = {n: refs.get(n) for n in wanted}
            for n in wanted:
                ref = refs.get(n)
                t = primary_text(ref) if ref else None
                if t:
                    t["ref"] = n
                    found["texts"].append(t)
                else:
                    found["notes"].append(f"reference {n}: " + ("no DOI or PMID on the page" if not ref
                                                                else "not in Europe PMC"))
            if not wanted:
                found["notes"].append("the row cites no reference number")
        else:
            for name in task["group"]["names"]:
                found["texts"] += reviews_for(name, TEXTS_PER_STRUCTURE)
                if len(found["texts"]) >= TEXTS_PER_STRUCTURE * max(1, len(task["group"]["names"]) // 2):
                    break
        log(texts_log, found)
        texts[task["task"]] = found
        return found

    def read(task: dict, text: dict, reader: str) -> None:
        if (task["task"], text["id"], reader) in done:
            return
        r = task.get("row") or {}
        focus = (f"{r['aspect']} of the {task['structure']} ({r.get('detail', '')}), techniques "
                 f"{', '.join(r.get('techniques', []))}" if r else "")
        sections = [tuple(s) for s in text["sections"]]
        answer, finish = router.chat(READERS[reader], read_prompt(task["structure"], sections, focus),
                                     reasoning=True, max_tokens=READER_ROOM)
        if not finish:
            return                                       # an answer cut by an error is not a reading
        log(readings_log, {"task": task["task"], "text": text["id"], "reader": reader, "finish": finish,
                           "items": read_items(answer)})

    stopped = ""
    try:
        for task in tasks:
            found = gather(task)
            jobs = [(task, t, reader) for t in found["texts"] for reader in READERS]
            with ThreadPoolExecutor(WORKERS) as pool:
                for f in [pool.submit(read, *j) for j in jobs]:
                    f.result()
            print(json.dumps({"task": task["task"], "texts": len(found["texts"]), "spent": round(router.cost, 4)}),
                  flush=True)
    except OverBudget as error:
        stopped = str(error)
        print(json.dumps({"reading_stopped": stopped}), flush=True)

    # the values: kept by the fixed rules, then judged
    sections_of = {(t["task"], x["id"]): [tuple(s) for s in x["sections"]] for t in texts.values() for x in t["texts"]}
    source_of = {(t["task"], x["id"]): {**x["source"], "read": x["read"], "ref": x.get("ref", "")}
                 for t in texts.values() for x in t["texts"]}
    structure_of = {t["task"]: t["structure"] for t in tasks}
    candidates: dict[str, dict] = {}
    refused: dict[str, int] = {}
    for reading in jsonl(readings_log):
        sections = sections_of.get((reading["task"], reading["text"]))
        if sections is None:
            continue
        for item in reading["items"]:
            v, why = as_value(item, sections)
            if v is None:
                refused[why.split(":")[0]] = refused.get(why.split(":")[0], 0) + 1
                continue
            k = key_of(v, structure_of.get(reading["task"], ""))
            c = candidates.setdefault(k, {**v, "task": reading["task"], "structure": structure_of.get(reading["task"]),
                                          "sources": [], "readers": []})
            src = source_of[(reading["task"], reading["text"])]
            if src not in c["sources"]:
                c["sources"].append(src)
            if reading["reader"] not in c["readers"]:
                c["readers"].append(reading["reader"])
    order = sorted(candidates, key=lambda k: hashlib.sha1(k.encode()).hexdigest())
    if not stopped:
        try:
            for k in order:
                if k in judged:
                    continue
                c, votes = candidates[k], []
                for _, model in JUDGES:
                    answer, _ = router.chat(model, judge_prompt(c["structure"], c), reasoning=False, max_tokens=60)
                    votes.append(dp.read_judgment(answer))
                    if votes[-1] != "yes":
                        break
                if votes and votes[-1] == "":
                    continue
                judged[k] = votes
                log(verdicts_log, {"key": k, "votes": votes})
        except OverBudget as error:
            stopped = str(error)
    for k, c in candidates.items():
        votes = judged.get(k)
        c["votes"] = votes
        c["accepted"] = votes == ["yes"] * len(JUDGES)

    # point 3: per row, what its primary says
    primary = []
    for task in (t for t in tasks if t["kind"] == "primary"):
        r, found = task["row"], texts.get(task["task"])
        if found is None:
            primary.append({"row": r["id"], "verdict": "not_read"})
            continue
        mine = [c for c in candidates.values() if c["task"] == task["task"] and c["accepted"]]
        want = {s: float(v) for s, v in (r.get("normal") or {}).items() if v is not None}
        same = [c for c in mine if c["unit"] == r["unit"] and all(abs(c.get(s, -1) - v) < 1e-6 for s, v in want.items())]
        verdict = ("no_reference" if not found["texts"] else "confirmed" if same else
                   "different" if mine else "not_found")
        primary.append({"row": r["id"], "status_now": r.get("status"), "verdict": verdict, "notes": found["notes"],
                        "refs": found.get("refs"), "read": [x["read"] for x in found["texts"]],
                        "confirming": same[:3], "other_values": [c for c in mine if c not in same][:5]})
    accepted = [c for c in candidates.values() if c["accepted"] and c["task"].startswith("structure:")]
    summary = {"stopped": stopped, "cost_usd": round(router.cost, 4), "tasks": len(tasks),
               "tasks_read": len(texts), "candidates": len(candidates), "judged": len(judged),
               "accepted_new_values": len(accepted), "refused_by_rule": refused,
               "primary": {v: sum(1 for p in primary if p["verdict"] == v)
                           for v in ("confirmed", "different", "not_found", "no_reference", "not_read")},
               "structures_with_values": sorted({c["structure"] for c in accepted}),
               "structures_without_value": sorted({t["structure"] for t in tasks if t["kind"] == "structure"
                                                   and t["task"] in texts}
                                                  - {c["structure"] for c in accepted})}
    out.write_text(json.dumps({"summary": summary, "primary": primary, "accepted": accepted,
                               "candidates": list(candidates.values())}, indent=1, ensure_ascii=False), "utf-8")
    out.with_name("summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), "utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
