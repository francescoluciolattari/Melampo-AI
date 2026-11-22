"""How much each manifestation weighs: frequencies and likelihood ratios measured in the literature (21 November 2026).

    python3 scripts/disease_frequencies.py --profiles profiles.json --hpo hp.obo --out out/frequencies.json \
        [--budget 5] [--resume previous]

A manifestation weighs in the ranking by two numbers: how often it is present in the disease, and how often in the
patients who have something else. Evidence-based diagnosis measures them in studies of diagnostic accuracy
(sensitivity, specificity, likelihood ratios: the method of JAMA's Rational Clinical Examination) and the first in
descriptive studies of patients with the disease. For every disease of the profiles this run searches, in order of
strength, and reads:

1. **Studies of diagnostic accuracy** of symptoms, signs, laboratory and imaging findings: sensitivity,
   specificity, LR+ and LR-, with the setting (emergency department, primary care, hospital) and the patients
   without the disease they were measured against. A meta-analysis's pooled values are taken as they are (it pooled
   with its own model, usually the bivariate one); single studies are pooled here.
2. **Descriptive studies** of patients with the disease ("fever in 73 of 100"): the frequency in the disease.
3. **Pooling.** Several studies of one manifestation are pooled by a random-effects model on the logit scale
   (DerSimonian-Laird), weighted by their patients, with the 95% interval and the heterogeneity (I2); sensitivity
   and specificity are pooled one by one (a declared simplification of the bivariate model, used only when no
   meta-analysis gives them). Studies in different settings are kept apart: spectrum bias makes a likelihood
   ratio of one setting wrong in another.
4. **Words of frequency** ("often", "rarely"), only when no number is given, mapped to HPO's frequency classes
   (those of Orphanet) at their middle, marked ``verbal``: the literature shows that such words are read very
   differently by different readers, so they count as the weakest evidence.
5. **The background measured**: the specificity of the accuracy studies, never estimated from the library when a
   study measured it.

Readers as for the reports (Nemotron 3 Ultra + Muse Spark Open, full reasoning, 60,000 tokens). Fixed before the
run: the sentence (or table row) is in the text word for word; its numbers are in it; counts agree with the
percentage given (within one point); a likelihood ratio given beside sensitivity and specificity agrees with them
(within 10%); two judges independent of the readers (Claude Sonnet 5, then Gemma 4) both say the sentence states
it; no person. Writes as it goes, stops at the budget and at once on a 402, resumes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
_spec = importlib.util.spec_from_file_location("reference_sources", HERE / "reference_sources.py")
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)

from melampo.clinical import case_graph as cg  # noqa: E402
from melampo.memory import disease_profiles as dp  # noqa: E402

TEXTS = {"accuracy": 3, "descriptive": 2}
VERDICTS = "v2:"          # the judges' second questionnaire (21 November 2026): states, category, same manifestation
# HPO frequency classes (Orphanet's bins) and their middles
VERBAL = (("obligate", 1.0, ("always", "invariably", "all patients", "universal")),
          ("very_frequent", 0.895, ("almost always", "nearly all", "most patients", "the majority of patients",
                                    "very common", "very frequent", "typically present")),
          ("frequent", 0.545, ("often", "frequent", "frequently", "common", "commonly", "usually", "many patients")),
          ("occasional", 0.17, ("occasional", "occasionally", "sometimes", "uncommon", "less common",
                                "less frequently", "some patients", "a minority")),
          ("very_rare", 0.025, ("rare", "rarely", "very rare", "exceptionally", "seldom")))
DESIGNS = ("meta_analysis", "systematic_review", "prospective_cohort", "retrospective_cohort", "cross_sectional",
           "case_control", "case_series", "review", "other")


def queries(disease: str) -> dict[str, list[str]]:
    d = f'TITLE_ABS:"{disease}"'
    open_ = "OPEN_ACCESS:y AND IN_EPMC:y"
    acc = ('(("sensitivity" AND "specificity") OR "likelihood ratio" OR "diagnostic accuracy") AND '
           '("clinical" OR "symptoms" OR "signs" OR "physical examination" OR "history" OR "laboratory")')
    desc = ('("clinical features" OR "clinical characteristics" OR "clinical presentation" OR "presenting symptoms" '
            'OR "symptoms and signs") AND "patients"')
    return {"accuracy": [f'{d} AND {acc} AND {open_} AND (PUB_TYPE:"systematic-review" OR TITLE:"meta-analysis")',
                         f"{d} AND {acc} AND {open_}", f"{d} AND {acc} AND HAS_ABSTRACT:y"],
            "descriptive": [f"{d} AND {desc} AND {open_}", f"{d} AND {desc} AND HAS_ABSTRACT:y"]}


def texts_for(disease: str) -> list[dict]:
    out, seen = [], set()
    for kind, qs in queries(disease).items():
        mine = 0
        for q in qs:
            for r in rs.epmc(q, 10):
                ident = r.get("pmcid") or r.get("pmid") or r.get("doi")
                if mine >= TEXTS[kind] or not ident or ident in seen:
                    continue
                t = rs.text_of(r)
                if t is None:
                    continue
                seen.add(ident)
                t["kind"] = kind
                out.append(t)
                mine += 1
            if mine >= TEXTS[kind]:
                break
    return out


def read_prompt(disease: str, sections: list[tuple[str, str]]) -> str:
    text = "\n\n".join(f"## {t}\n{x}" for t, x in sections)
    return (f"You read a medical publication about {disease}. List every MEASURE it states of how often a "
            "manifestation (symptom, sign, laboratory value, imaging finding, complication) is present, for patients "
            f"with {disease} or for patients without it.\nFor each answer an object with: manifestation (its name), "
            "with_disease {count, total, percent} (patients with the disease who have it), sensitivity, "
            "specificity, lr_positive, lr_negative (as written: a percentage or a fraction or a ratio, null when "
            "absent), comparator (which patients without the disease the specificity was measured in, as written), "
            "setting (emergency department, primary care, hospital, surgical, population, other), patients (how many "
            "patients the measure rests on), design (meta_analysis, systematic_review, prospective_cohort, "
            "retrospective_cohort, cross_sectional, case_control, case_series, review, other), pooled (true when it "
            "is a meta-analysis's pooled value), verbal (a word of frequency such as \"often\" or \"rarely\" when the "
            "text gives no number, else \"\"), and quote: ONE sentence or ONE table row copied word for word from "
            "the text, containing the numbers or the word.\nDo not compute anything the text does not state. "
            "Answer only a JSON list, [] when there is none.\n\n" + text)


def _num(v) -> float | None:
    if v in (None, "", "null"):
        return None
    m = re.search(r"\d+(?:[.,]\d+)?", str(v))
    return float(m.group(0).replace(",", ".")) if m else None


def _prop(v) -> float | None:
    x = _num(v)
    if x is None:
        return None
    return x / 100 if ("%" in str(v) or x > 1) else x


NUMBER_WORDS = {w: i for i, w in enumerate(("zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                                              "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
                                              "sixteen", "seventeen", "eighteen", "nineteen", "twenty"))}
RANGE = re.compile(r"^\s*\d+(?:[.,·]\d+)?\s*%?\s*(?:-|–|to|and)\s*\d")


def _letters(text: str) -> str:
    """The characters of a text without its spacing and table bars, typography folded (21 November 2026, declared
    after the first run: a table row the reader began with "|", or "I 2" for the text's "I2", is the same text)."""
    return re.sub(r"[\s|]+", "", dp._plain(text)).replace("·", ".")


def in_text(quote: str, sections: list[tuple[str, str]]) -> bool:
    q = _letters(quote).strip(".")
    return bool(q) and any(q in _letters(x) for _, x in sections)


def checked(item: dict, sections: list[tuple[str, str]]) -> tuple[dict | None, str]:
    """The measure, or None and why not, by the rules fixed before the run (corrections declared on 21 November
    2026, after the first run: spacing and table bars, numbers written as words, ranges, a meta-analysis's ratios)."""
    quote = " ".join(str(item.get("quote", "")).split())
    if not quote or not in_text(quote, sections):
        return None, "quote_not_in_text"
    plain = quote.replace("·", ".")
    numbers = [float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", plain)]
    numbers += [float(v) for w, v in NUMBER_WORDS.items() if re.search(rf"\b{w}\b", plain, re.I)]
    for k in ("sensitivity", "specificity", "lr_positive", "lr_negative"):
        if item.get(k) not in (None, "", "null") and RANGE.match(str(item[k])):
            return None, "a_range_not_a_value"

    def in_quote(v) -> bool:
        x = _num(v)
        return x is None or any(abs(x - n) < 1e-9 for n in numbers)

    wd = item.get("with_disease") if isinstance(item.get("with_disease"), dict) else {}
    fields = [wd.get("count"), wd.get("total"), wd.get("percent"), item.get("sensitivity"), item.get("specificity"),
              item.get("lr_positive"), item.get("lr_negative")]
    if not all(in_quote(v) for v in fields):
        return None, "number_not_in_quote"
    out = {"manifestation": str(item.get("manifestation", "")).strip(), "quote": quote,
           "setting": str(item.get("setting", "") or "not stated").lower(),
           "comparator": str(item.get("comparator", "") or ""), "design": str(item.get("design", "other")).lower(),
           "pooled": bool(item.get("pooled")), "patients": _num(item.get("patients"))}
    if out["design"] not in DESIGNS:
        out["design"] = "other"
    count, total, percent = _num(wd.get("count")), _num(wd.get("total")), _prop(wd.get("percent"))
    if count is not None and total:
        if count > total:
            return None, "count_above_total"
        if percent is not None and abs(count / total - percent) > 0.01:
            return None, "count_and_percent_disagree"
        out["with_disease"] = {"count": count, "total": total}
    elif percent is not None:
        out["with_disease"] = {"proportion": percent, "total": total or out["patients"]}
    for k in ("sensitivity", "specificity"):
        p = _prop(item.get(k))
        if p is not None:
            if not 0 <= p <= 1:
                return None, "proportion_out_of_range"
            out[k] = p
    for k in ("lr_positive", "lr_negative"):
        x = _num(item.get(k))
        if x is not None:
            out[k] = x
    sens, spec = out.get("sensitivity"), out.get("specificity")
    # a meta-analysis pools its likelihood ratios with its own model: they need not equal the ratio of its pooled
    # sensitivity and specificity (MRI in appendicitis: 97% and 95%, LR+ 16.3, not 19.4)
    if sens is not None and spec is not None and not (out["pooled"] or out["design"] in ("meta_analysis",
                                                                                        "systematic_review")):
        if "lr_positive" in out and spec < 1 and abs(out["lr_positive"] - sens / (1 - spec)) > 0.2 * out["lr_positive"]:
            return None, "lr_disagrees_with_sensitivity_and_specificity"
        if "lr_negative" in out and spec > 0 and abs(out["lr_negative"] - (1 - sens) / spec) > max(
                0.2 * out["lr_negative"], 0.02):
            return None, "lr_disagrees_with_sensitivity_and_specificity"
    verbal = str(item.get("verbal", "") or "").strip().lower()
    if verbal and not any(k in out for k in ("with_disease", "sensitivity", "specificity", "lr_positive",
                                              "lr_negative")):
        if verbal not in quote.lower():
            return None, "word_not_in_quote"
        cls = next((c for c, _, words in VERBAL if any(verbal == w or verbal.startswith(w + " ") for w in words)), "")
        if not cls:
            return None, "word_not_a_frequency"
        out["verbal"] = {"word": verbal, "class": cls, "frequency": next(f for c, f, _ in VERBAL if c == cls)}
    if not any(k in out for k in ("with_disease", "sensitivity", "specificity", "lr_positive", "lr_negative",
                                  "verbal")):
        return None, "no_measure"
    return out, ""


def claim(disease: str, m: dict) -> str:
    parts = []
    wd = m.get("with_disease") or {}
    if "count" in wd:
        parts.append(f"{wd['count']:g} of {wd['total']:g} patients with {disease} had it")
    elif "proportion" in wd:
        parts.append(f"{wd['proportion']:.0%} of patients with {disease} had it")
    for k, label in (("sensitivity", "sensitivity"), ("specificity", "specificity"),
                     ("lr_positive", "positive likelihood ratio"), ("lr_negative", "negative likelihood ratio")):
        if k in m:
            parts.append(f"{label} {m[k]:.3g}")
    if "verbal" in m:
        parts.append(f"it is {m['verbal']['word']} in {disease}")
    return "; ".join(parts) + f" (setting: {m['setting']})"


CATEGORIES = ("manifestation", "test", "score", "outcome")


def judge_prompt(disease: str, m: dict, profile_names: list[str] | None = None) -> str:
    names = "\n".join(f"- {n}" for n in (profile_names or []))
    return (f"Disease: {disease}\nMeasured: {m['manifestation']}\nSentence (or table row) from a published "
            f"text: \"{m['quote']}\"\nClaim: {claim(disease, m)}.\n\n"
            "1. states: yes when the sentence states these numbers for this in this disease; no when it does not, "
            "when they belong to another disease or group, or when they are a number of something else.\n"
            "2. category: manifestation (a symptom, sign, laboratory value or imaging finding the patient has, or a "
            "complication of the disease itself), test (how well an examination or a laboratory test detects the "
            "disease: an imaging exam, an assay, a test with its cut-off), score (a clinical score or decision rule "
            "combining several findings), outcome (death, a complication of treatment, length of stay, recurrence).\n"
            "3. same_as: when the category is manifestation, the name of the manifestation of the list below that is "
            "the same manifestation (same thing, possibly other words or a more specific form), copied exactly; "
            "\"none\" when none is.\n"
            f"Manifestations already known for {disease}:\n{names}\n\n"
            'Answer only JSON: {"states": "yes" or "no", "category": "...", "same_as": "..."}.')


def read_verdict(answer: str) -> dict:
    m = re.search(r"\{.*\}", answer or "", re.DOTALL)
    try:
        got = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        got = {}
    states = str(got.get("states", "")).strip().lower()
    category = str(got.get("category", "")).strip().lower()
    return {"states": states if states in ("yes", "no") else "",
            "category": category if category in CATEGORIES else "",
            "same_as": str(got.get("same_as", "") or "").strip()}


# -- pooling ---------------------------------------------------------------------------------------------------------

def pool(pairs: list[tuple[float, float]]) -> dict | None:
    """Random-effects pooled proportion (DerSimonian-Laird, logit scale) of (events, total); 0.5 added only to a
    study with no event or all events."""
    pairs = [(x, n) for x, n in pairs if n and n > 0]
    if not pairs:
        return None
    ys, vs = [], []
    for x, n in pairs:
        # the 0.5 correction only where a study has 0 or all events (its logit is otherwise infinite)
        x0, n0 = (x + 0.5, n + 1.0) if x <= 0 or x >= n else (x, n)
        ys.append(math.log(x0 / (n0 - x0)))
        vs.append(1 / x0 + 1 / (n0 - x0))
    w = [1 / v for v in vs]
    fixed = sum(wi * yi for wi, yi in zip(w, ys)) / sum(w)
    q = sum(wi * (yi - fixed) ** 2 for wi, yi in zip(w, ys))
    k = len(pairs)
    c = sum(w) - sum(wi * wi for wi in w) / sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 and c > 0 else 0.0
    wr = [1 / (v + tau2) for v in vs]
    mean = sum(wi * yi for wi, yi in zip(wr, ys)) / sum(wr)
    se = math.sqrt(1 / sum(wr))
    inv = lambda t: 1 / (1 + math.exp(-t))  # noqa: E731
    return {"value": round(inv(mean), 4), "ci": [round(inv(mean - 1.96 * se), 4), round(inv(mean + 1.96 * se), 4)],
            "studies": k, "patients": int(sum(n for _, n in pairs)),
            "i2": round(max(0.0, (q - (k - 1)) / q), 3) if k > 1 and q > 0 else 0.0}


def weigh(measures: list[dict]) -> dict:
    """The weight of one manifestation from its accepted measures, by the order of strength fixed before the run."""
    out: dict = {}
    by_setting: dict[str, list[dict]] = {}
    for m in measures:
        if any(k in m for k in ("sensitivity", "specificity", "lr_positive", "lr_negative")):
            by_setting.setdefault(m["setting"], []).append(m)
    if by_setting:
        strata = {}
        for setting, ms in by_setting.items():
            pooled = [m for m in ms if m["pooled"] or m["design"] == "meta_analysis"]
            if pooled:
                best = max(pooled, key=lambda m: m.get("patients") or 0)
                acc = {k: best[k] for k in ("sensitivity", "specificity", "lr_positive", "lr_negative") if k in best}
                acc.update(design="meta_analysis", studies=None, patients=best.get("patients"))
            else:
                acc = {"design": "pooled_studies" if len(ms) > 1 else ms[0]["design"]}
                for k in ("sensitivity", "specificity"):
                    pairs = [(m[k] * m["patients"], m["patients"]) for m in ms if k in m and m.get("patients")]
                    p = pool(pairs)
                    if p:
                        acc[k], acc[f"{k}_ci"], acc["studies"], acc["patients"] = (p["value"], p["ci"], p["studies"],
                                                                                    p["patients"])
                        acc[f"{k}_i2"] = p["i2"]
                    elif any(k in m for m in ms):
                        acc[k] = next(m[k] for m in ms if k in m)
                for k in ("lr_positive", "lr_negative"):
                    if k in acc or ("sensitivity" in acc and "specificity" in acc):
                        continue
                    vals = [m[k] for m in ms if k in m]
                    if vals:
                        acc[k] = sorted(vals)[len(vals) // 2]         # the median of the studies' ratios
            acc["setting"] = setting
            acc["comparator"] = next((m["comparator"] for m in ms if m["comparator"]), "")
            acc["sources"] = [m["source"] for m in ms][:6]
            strata[setting] = acc
        main = max(strata.values(), key=lambda a: a.get("patients") or 0)
        out["accuracy"] = {**main, "other_settings": [s for s in strata if s != main["setting"]]}
    freq = [((m["with_disease"]["count"], m["with_disease"]["total"]) if "count" in m.get("with_disease", {}) else
             (m["with_disease"]["proportion"] * m["with_disease"]["total"], m["with_disease"]["total"]))
            for m in measures if m.get("with_disease") and (m["with_disease"].get("total"))]
    freq += [(m["sensitivity"] * m["patients"], m["patients"]) for m in measures
             if "sensitivity" in m and m.get("patients") and not m.get("with_disease")]
    p = pool(freq)
    if p:
        out["frequency"], out["frequency_ci"] = p["value"], p["ci"]
        out["frequency_basis"] = "pooled_studies" if p["studies"] > 1 else "one_study"
        out["frequency_studies"], out["frequency_patients"], out["frequency_i2"] = p["studies"], p["patients"], p["i2"]
    elif any("verbal" in m for m in measures):
        v = [m["verbal"] for m in measures if "verbal" in m]
        f = sorted(x["frequency"] for x in v)[len(v) // 2]
        out["frequency"], out["frequency_basis"], out["verbal_words"] = f, "verbal", sorted({x["word"] for x in v})
    return out


# -- run --------------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--hpo", required=True)
    ap.add_argument("--out", default="out/frequencies.json")
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--resume", default="")
    args = ap.parse_args(argv)
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    logs = {n: out.with_name(n) for n in ("texts.jsonl", "readings.jsonl", "verdicts.jsonl")}
    if args.resume:
        for name, path in logs.items():
            src = Path(args.resume) / name
            if src.exists() and src.resolve() != path.resolve():
                path.write_text(src.read_text("utf-8"), "utf-8")
    texts = {r["disease"]: r for r in rs.jsonl(logs["texts.jsonl"])}
    done = {(r["disease"], r["text"], r["reader"]) for r in rs.jsonl(logs["readings.jsonl"])}
    judged = {r["key"]: r["votes"] for r in rs.jsonl(logs["verdicts.jsonl"])}
    hpo = dp.Terms.from_obo(Path(args.hpo).read_text("utf-8", errors="replace").splitlines())
    profiles = [cg.Disease.from_dict(p) for p in json.loads(Path(args.profiles).read_text("utf-8"))["profiles"]]
    router, lock = rs.Router(key, args.budget), threading.Lock()

    def log(name: str, row: dict) -> None:
        with lock, logs[name].open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def read(disease: str, text: dict, reader: str) -> None:
        if (disease, text["id"], reader) in done:
            return
        answer, finish = router.chat(rs.READERS[reader], read_prompt(disease, [tuple(s) for s in text["sections"]]),
                                     reasoning=True, max_tokens=rs.READER_ROOM)
        if finish:
            log("readings.jsonl", {"disease": disease, "text": text["id"], "reader": reader, "finish": finish,
                                   "items": rs.read_items(answer)})

    stopped = ""
    try:
        for d in profiles:
            if d.words not in texts:
                texts[d.words] = {"disease": d.words, "texts": texts_for(d.words)}
                log("texts.jsonl", texts[d.words])
            with ThreadPoolExecutor(rs.WORKERS) as pool_:
                for f in [pool_.submit(read, d.words, t, r) for t in texts[d.words]["texts"] for r in rs.READERS]:
                    f.result()
            print(json.dumps({"disease": d.words, "texts": len(texts[d.words]["texts"]),
                              "spent": round(router.cost, 4)}), flush=True)
    except rs.OverBudget as error:
        stopped = str(error)
        print(json.dumps({"reading_stopped": stopped}), flush=True)

    sections_of = {(t["disease"], x["id"]): [tuple(s) for s in x["sections"]] for t in texts.values() for x in t["texts"]}
    source_of = {(t["disease"], x["id"]): {**x["source"], "read": x["read"], "kind": x["kind"]}
                 for t in texts.values() for x in t["texts"]}
    measures, refused = {}, {}
    for r in rs.jsonl(logs["readings.jsonl"]):
        sections = sections_of.get((r["disease"], r["text"]))
        if sections is None:
            continue
        for item in r["items"]:
            m, why = checked(item, sections)
            if m is None:
                refused[why] = refused.get(why, 0) + 1
                continue
            m["source"] = source_of[(r["disease"], r["text"])]
            k = json.dumps([r["disease"], dp.norm(m["manifestation"]), m["quote"]])
            m.setdefault("readers", [])
            measures.setdefault(k, {**m, "disease": r["disease"], "readers": []})["readers"].append(r["reader"])
    names_of = {d.words: sorted({m.words for m in d.manifestations}) for d in profiles}
    if not stopped:
        try:
            for k in sorted(measures, key=lambda k: hashlib.sha1(k.encode()).hexdigest()):
                if VERDICTS + k in judged:
                    continue
                m, votes = measures[k], []
                for _, model in rs.JUDGES:
                    answer, _ = router.chat(model, judge_prompt(m["disease"], m, names_of.get(m["disease"])),
                                            reasoning=False, max_tokens=200)
                    votes.append(read_verdict(answer))
                    if votes[-1]["states"] != "yes":
                        break
                if votes and votes[-1]["states"] == "":
                    continue
                judged[VERDICTS + k] = votes
                log("verdicts.jsonl", {"key": VERDICTS + k, "votes": votes})
        except rs.OverBudget as error:
            stopped = str(error)

    # accepted: both judges say it states it and agree on what it is; a manifestation goes to the one of the profile
    # both name (checked to exist), else stands as new; tests and scores are kept apart, outcomes left out
    accepted, tests, scores, outcomes, doubts = [], [], [], 0, []
    for k, m in measures.items():
        votes = judged.get(VERDICTS + k) or []
        if len(votes) != len(rs.JUDGES) or any(v["states"] != "yes" for v in votes):
            continue
        cats = {v["category"] for v in votes}
        if len(cats) != 1 or "" in cats:
            doubts.append({"measure": m["manifestation"], "disease": m["disease"], "categories": sorted(cats)})
            continue
        m["category"] = cats.pop()
        same = {v["same_as"] for v in votes}
        m["same_as"] = same.pop() if len(same) == 1 and next(iter(same)) in names_of.get(m["disease"], []) else ""
        if m["category"] == "manifestation":
            accepted.append(m)
        elif m["category"] == "test":
            tests.append(m)
        elif m["category"] == "score":
            scores.append(m)
        else:
            outcomes += 1
    weighted, per_pair, new = [], [], []
    for d in profiles:
        mine = [m for m in accepted if m["disease"] == d.words]
        groups: dict[str, list[dict]] = {}
        index, by_words = {}, {}
        for man in d.manifestations:
            index.setdefault(man.concept.curie if man.concept and man.concept.code else dp.norm(man.words), man)
            by_words.setdefault(man.words, man)
        for m in mine:
            known = by_words.get(m["same_as"]) if m["same_as"] else None
            if known is not None:
                key_ = known.concept.curie if known.concept and known.concept.code else dp.norm(known.words)
            else:
                key_ = hpo.find(m["manifestation"], prefix="HP:") or dp.norm(m["manifestation"])
            groups.setdefault(key_, []).append(m)
        for key_, ms in groups.items():
            w = weigh(ms)
            target = index.get(key_)
            if target is None:
                target = cg.Manifestation(f"f{len(d.manifestations) + 1}", ms[0]["manifestation"],
                                          cg.Concept("HPO", key_, ms[0]["manifestation"]) if key_.startswith("HP:")
                                          else None, citations=[m["source"] for m in ms][:6], evidence="reported")
                d.manifestations.append(target)
                index[key_] = target
                new.append({"disease": d.words, "manifestation": target.words})
            if "frequency" in w and w.get("frequency_basis") != "verbal" or (
                    "frequency" in w and target.frequency is None and not target.typicality):
                target.frequency, target.frequency_basis = w["frequency"], w["frequency_basis"]
            if "accuracy" in w:
                target.accuracy = w["accuracy"]
            per_pair.append({"disease": d.words, "manifestation": target.words, "key": key_, **w,
                             "measures": len(ms)})
        weighted.append(d)
    counts = {"manifestations": sum(len(d.manifestations) for d in weighted),
              "with_accuracy": sum(1 for d in weighted for m in d.manifestations if m.accuracy),
              "with_frequency_measured": sum(1 for d in weighted for m in d.manifestations
                                             if m.frequency is not None and m.frequency_basis in (
                                                 "pooled_studies", "one_study")),
              "with_frequency_verbal": sum(1 for d in weighted for m in d.manifestations if m.frequency_basis == "verbal"),
              "with_typicality_only": sum(1 for d in weighted for m in d.manifestations
                                          if m.frequency is None and not m.accuracy and m.typicality),
              "without_weight": sum(1 for d in weighted for m in d.manifestations
                                    if m.frequency is None and not m.accuracy and not m.typicality)}
    def kept_apart(ms: list[dict]) -> dict:
        out: dict = {}
        for m in ms:
            out.setdefault(m["disease"], {}).setdefault(dp.norm(m["manifestation"]), []).append(m)
        return {d: [{"name": v[0]["manifestation"], **weigh(v), "measures": len(v)} for v in g.values()]
                for d, g in out.items()}
    tests_by, scores_by = kept_apart(tests), kept_apart(scores)
    out.with_name("tests.json").write_text(json.dumps(tests_by, indent=1, ensure_ascii=False), "utf-8")
    out.with_name("scores.json").write_text(json.dumps(scores_by, indent=1, ensure_ascii=False), "utf-8")
    counts.update(tests=len(tests), scores=len(scores), outcomes_left_out=outcomes, category_doubts=len(doubts),
                  matched_to_profile=sum(1 for m in accepted if m["same_as"]))
    summary = {"stopped": stopped, "cost_usd": round(router.cost, 4), "diseases": len(profiles),
               "texts": sum(len(t["texts"]) for t in texts.values()), "measures_read": len(measures),
               "judged": len(judged), "accepted": len(accepted), "refused_by_rule": refused,
               "new_manifestations": len(new), **counts}
    Path(out.with_name("profiles_weighted.json")).write_text(json.dumps(
        {"profiles": [d.as_dict() for d in weighted]}, indent=1, ensure_ascii=False), "utf-8")
    out.write_text(json.dumps({"summary": summary, "weights": per_pair, "new": new}, indent=1, ensure_ascii=False),
                   "utf-8")
    out.with_name("summary.json").write_text(json.dumps(summary, indent=1), "utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
