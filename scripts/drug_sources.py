"""Build the local cache of the drug branch from the open sources (15 November 2026). No language model is asked,
no patient's datum is sent: every ingredient AIFA lists is resolved and its US label fetched, whatever case will
later read it.

    python3 scripts/drug_sources.py --out drug_cache [--limit 400] [--aifa file.csv] [--ema file.xlsx]

1. AIFA "Anagrafica farmaci" (confezioni_fornitura.csv): the products, their ingredient as AIFA writes it, ATC.
2. ATC -> international name in English: Wikidata (P267, CC0) and the EMA medicines table. For each AIFA
   ingredient: the names its ATC codes carry; one distinct name -> taken; several -> those that share the
   ingredient's stem; the two sources agree -> ``wikidata+ema``; one source only -> that source; they disagree
   -> the label decides if exactly one of the names has a US label (``label``), else ``doubt``. A combination
   ("A/B") is resolved part by part by the stem among all the names (``stem``), a weaker basis, stated.
3. openFDA: the label of each international name (``--limit`` labels per run; the cache resumes: a name already
   asked is not asked again). Without OPENFDA_API_KEY openFDA allows 1,000 requests a day.

Writes ``products.json``, ``ingredients.json``, ``labels/*.json``, ``manifest.json`` (sources, dates, licences,
the columns recognised) and prints the counts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import urllib.parse
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from melampo.connectors import drug_sources as ds

_COMBINED = re.compile(r"/|;|\+|\band\b", re.IGNORECASE)   # a comma may only list a salt
_SPLIT = re.compile(r"\s*(?:/|\+|;|,)\s*")


def _spelling(word: str) -> str:
    """Italian and English spellings of an international name brought together for the stem only: th -> t,
    ph -> f, y -> i, ae -> e, k -> c ("claritromicina" / "clarithromycin")."""
    word = word.replace(" ", "")
    for a, b in (("th", "t"), ("ph", "f"), ("ae", "e"), ("y", "i"), ("k", "c")):
        word = word.replace(a, b)
    return word


def stem_matches(ingredient: str, name: str) -> bool:
    """The Italian ingredient and the international name share their stem: the common beginning is at least five
    letters and all but three of the shorter ("atorvastatina" / "atorvastatin", "acarbosio" / "acarbose")."""
    a, b = _spelling(ds.ingredient_core(ingredient)), _spelling(ds.base_name(name))
    if not a or not b:
        return False
    if abs(len(a) - len(b)) > 4:          # "cedazuridina" is not "cedazuridine / decitabine"
        return False
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return common >= max(5, min(len(a), len(b)) - 3)


def _one(names: list[str]) -> list[str]:
    """Distinct international names by their base (salts aside), the shortest spelling of each."""
    by_base: dict[str, str] = {}
    for n in sorted(names, key=lambda x: (len(x), ds.norm(x))):
        by_base.setdefault(ds.base_name(n), n)
    return [ds.base_name(n) for n in by_base.values() if ds.base_name(n)]


def resolve_one(ingredient: str, pairs: list[dict[str, str]], ema_names: list[str]) -> dict:
    """The international name of one Italian ingredient from the names its ATC codes carry. A name is taken only
    when it is the ingredient's own (15 November 2026: the second run took the single EMA name of B01AC06,
    "clopidogrel", for "acetilsalicilato di lisina", whose ATC an EMA combination product also carries):

    1. ``wikidata_it``: an item whose Italian label is the ingredient (salts aside) gives its English label;
    2. ``stem``: an English name (Wikidata or EMA) shares the ingredient's stem;
    3. ``wikidata+ema``: the two sources give one and the same name;
    otherwise ``doubt`` with the candidates, or ``unresolved`` when there is none."""
    core = ds.ingredient_core(ingredient)
    italian = _one([p["en"] for p in pairs if p["en"] and p["it"] and ds.ingredient_core(p["it"]) == core])
    if len(italian) == 1:
        return {"inn": italian, "basis": "wikidata_it"}
    # a combination's name ("cedazuridine / decitabine") is never the name of a single ingredient
    english = [n for n in [p["en"] for p in pairs if p["en"]] + list(ema_names) if not _COMBINED.search(n)]
    stemmed = _one([n for n in english if stem_matches(ingredient, n)])
    if len(stemmed) == 1:
        return {"inn": stemmed, "basis": "stem"}
    w, e = _one([p["en"] for p in pairs if p["en"]]), _one(list(ema_names))
    if len(w) == 1 and w == e:
        return {"inn": w, "basis": "wikidata+ema"}
    candidates = sorted(set(italian + stemmed + w + e))
    return {"inn": [], "basis": "doubt", "candidates": candidates} if candidates else {"inn": [],
                                                                                        "basis": "unresolved"}


def resolve(ingredients: dict[str, set[str]], wikidata: dict[str, list[dict[str, str]]],
            ema: dict[str, list[str]]) -> dict:
    """Each AIFA ingredient -> its international name(s) with the basis; a combination ("A/B") part by part,
    each part against all the names (its ATC is the combination's)."""
    all_pairs = [p for pairs in wikidata.values() for p in pairs]
    all_ema = sorted({n for names in ema.values() for n in names})
    out = {}
    for ingredient, codes in sorted(ingredients.items()):
        parts = list(dict.fromkeys(p for p in _SPLIT.split(ingredient) if p.strip() and ds.norm(p) not in ("n d", "nd")))
        if len(parts) == 1 and parts[0] != ingredient:
            pairs = [p for c in codes for p in wikidata.get(c, [])]
            names = [n for c in codes for n in ema.get(c, [])]
            out[ingredient] = {"atc": sorted(codes), **resolve_one(parts[0], pairs, names)}
            continue
        if len(parts) > 1:
            got = [resolve_one(part, all_pairs, all_ema) for part in parts]
            ok = all(g["inn"] and g["basis"] in ("wikidata_it", "stem") for g in got)
            out[ingredient] = {"atc": sorted(codes), "inn": [g["inn"][0] for g in got] if ok else [],
                               "basis": "combination" if ok else "unresolved", "parts": parts}
            continue
        pairs = [p for c in codes for p in wikidata.get(c, [])]
        names = [n for c in codes for n in ema.get(c, [])]
        out[ingredient] = {"atc": sorted(codes), **resolve_one(ingredient, pairs, names)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="drug_cache")
    ap.add_argument("--limit", type=int, default=400, help="labels to ask openFDA in this run")
    ap.add_argument("--aifa", default="", help="a local copy of confezioni_fornitura.csv")
    ap.add_argument("--aifa-pa", default="", help="a local copy of PA_confezioni.csv")
    ap.add_argument("--ema", default="", help="a local copy of the EMA medicines table (.xlsx)")
    ap.add_argument("--wikidata", default="", help="a local copy of the Wikidata ATC answer (.json)")
    args = ap.parse_args(argv)
    cache, http = ds.DrugCache(Path(args.out)), ds.Http()
    today = dt.datetime.now(dt.UTC).date().isoformat()
    manifest = cache.read_json("manifest.json", {})

    raw = Path(args.aifa).read_bytes() if args.aifa else http.get(ds.AIFA_LISTS["anagrafica"], accept="text/csv")
    anagrafica = ds.decode(raw)
    products, aifa_report = ds.read_aifa(anagrafica)
    if not products and "ingredient" not in aifa_report.get("columns", {}):
        # the "Anagrafica farmaci" names no ingredient: they are in "Principi attivi per confezione"
        pa_raw = Path(args.aifa_pa).read_bytes() if args.aifa_pa else http.get(ds.AIFA_LISTS["principi_attivi"],
                                                                               accept="text/csv")
        by_package, pa_report = ds.read_aifa_ingredients(ds.decode(pa_raw))
        manifest["aifa_principi_attivi"] = {"url": ds.AIFA_LISTS["principi_attivi"], "taken": today,
                                            "licence": ds.AIFA_LICENCE, **pa_report}
        products, aifa_report = ds.read_aifa(anagrafica, by_package)
    manifest["aifa"] = {"url": ds.AIFA_LISTS["anagrafica"], "taken": today, "licence": ds.AIFA_LICENCE,
                        **aifa_report}
    if not products:
        cache.write_json("manifest.json", manifest)
        print(json.dumps({"problem": aifa_report, "principi_attivi": manifest.get("aifa_principi_attivi")},
                         indent=2, ensure_ascii=False))
        return 1
    cache.write_json("products.json", [ds.asdict(p) for p in products])

    # each source may fail on its own (15 November 2026: the second run stopped after AIFA and EMA without saying
    # why); a source that fails is reported and the others go on
    wikidata: dict = {}
    try:
        if args.wikidata:
            wikidata = ds.read_wikidata_atc(json.loads(Path(args.wikidata).read_text("utf-8")))
        else:
            for group in ds.ATC_GROUPS:
                url = (f"{ds.WIKIDATA_SPARQL}?"
                       f"{urllib.parse.urlencode({'query': ds.wikidata_atc_query(group), 'format': 'json'})}")
                part = ds.read_wikidata_atc(json.loads(http.get(url, accept="application/sparql-results+json",
                                                                timeout=120)))
                for code, pairs in part.items():
                    wikidata.setdefault(code, []).extend(p for p in pairs if p not in wikidata.get(code, []))
        manifest["wikidata"] = {"url": ds.WIKIDATA_SPARQL, "taken": today, "licence": ds.WIKIDATA_LICENCE,
                                "codes": len(wikidata)}
    except Exception as error:  # noqa: BLE001
        manifest["wikidata"] = {"url": ds.WIKIDATA_SPARQL, "problem": f"{type(error).__name__}: {error}"[:500]}
    cache.write_json("manifest.json", manifest)
    ema: dict = {}
    ema_report: dict = {}
    try:
        ema_path = Path(args.ema) if args.ema else Path(args.out) / "ema_medicines.xlsx"
        if not args.ema:
            ema_path.write_bytes(http.get(ds.EMA_MEDICINES, accept="*/*", timeout=180))
        ema, ema_report = ds.read_ema_medicines(ema_path)
        manifest["ema"] = {"url": ds.EMA_MEDICINES, "taken": today, "licence": ds.EMA_LICENCE, **ema_report}
    except Exception as error:  # noqa: BLE001
        manifest["ema"] = {"url": ds.EMA_MEDICINES, "problem": f"{type(error).__name__}: {error}"[:500]}
    cache.write_json("manifest.json", manifest)

    ingredients = resolve(ds.ingredients_of(products), wikidata, ema)
    key = os.environ.get("OPENFDA_API_KEY", "")
    names = sorted({n for e in ingredients.values() for n in e["inn"]}, key=ds.norm)
    asked, failures = 0, {}
    for name in names:
        if cache.has_label_entry(name):
            continue
        if asked >= args.limit:
            break
        try:
            cache.put_label(name, ds.fetch_label(http, name, key))
        except Exception as error:  # noqa: BLE001 -- asked again next run
            failures[name] = f"{type(error).__name__}: {error}"[:200]
            if len(failures) >= 25:
                break                      # openFDA refuses repeatedly (limit reached): the next run resumes
        asked += 1
    # the label decides between two names that disagree, when exactly one of them has a label
    for entry in ingredients.values():
        if entry["basis"] == "doubt":
            with_label = [c for c in entry.get("candidates", []) if cache.label(c) is not None]
            if len(with_label) == 1:
                entry.update(inn=with_label, basis="label")
    cache.write_json("ingredients.json", ingredients)
    labels = {n: cache.label(n) for n in names if cache.has_label_entry(n)}
    manifest["openfda"] = {"url": ds.OPENFDA_LABEL, "taken": today, "licence": ds.OPENFDA_LICENCE,
                           "asked_this_run": asked, "with_key": bool(key),
                           "failures": dict(list(failures.items())[:25])}
    cache.write_json("manifest.json", manifest)
    summary = {
        "products": len(products), "ingredients": len(ingredients),
        "resolution": dict(Counter(e["basis"] for e in ingredients.values())),
        "international_names": len(names), "labels_asked": len(labels),
        "labels_found": sum(1 for v in labels.values() if v is not None),
        "labels_with_interactions": sum(1 for v in labels.values() if v and v.sections.get("drug_interactions")),
        "labels_still_to_ask": len(names) - len(labels),
        "aifa_columns": aifa_report.get("columns"), "ema_columns": ema_report.get("columns"),
        "wikidata_codes": len(wikidata), "ema_codes": len(ema), "label_failures": len(failures),
        "http_calls": http.calls,
    }
    cache.write_json("summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
