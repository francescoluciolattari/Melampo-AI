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

_SPLIT = re.compile(r"\s*(?:/|\+|;|,)\s*")


def stem_matches(ingredient: str, name: str) -> bool:
    a, b = ds.ingredient_core(ingredient).replace(" ", ""), ds.norm(name).replace(" ", "")
    if not a or not b:
        return False
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return common >= max(5, min(len(a), len(b)) - 3)


def choose(ingredient: str, names: list[str]) -> list[str]:
    distinct = sorted({n for n in names}, key=ds.norm)
    by_norm = {}
    for n in distinct:
        by_norm.setdefault(ds.norm(n), n)
    distinct = list(by_norm.values())
    if len(distinct) <= 1:
        return distinct
    stemmed = [n for n in distinct if stem_matches(ingredient, n)]
    return stemmed or distinct


def resolve(ingredients: dict[str, set[str]], wikidata: dict[str, list[str]], ema: dict[str, list[str]]) -> dict:
    all_names = sorted({n for names in list(wikidata.values()) + list(ema.values()) for n in names}, key=ds.norm)
    out = {}
    for ingredient, codes in sorted(ingredients.items()):
        parts = [p for p in _SPLIT.split(ingredient) if p.strip()]
        if len(parts) > 1:
            inns = []
            for part in parts:
                hits = [n for n in all_names if stem_matches(part, n)]
                hits = list({ds.norm(h): h for h in hits}.values())
                inns.append(hits[0] if len(hits) == 1 else "")
            ok = all(inns)
            out[ingredient] = {"atc": sorted(codes), "inn": inns if ok else [], "basis": "stem" if ok else "unresolved",
                               "parts": parts}
            continue
        w = choose(ingredient, [n for c in codes for n in wikidata.get(c, [])])
        e = choose(ingredient, [n for c in codes for n in ema.get(c, [])])
        if len(w) == 1 and len(e) == 1 and ds.norm(w[0]) == ds.norm(e[0]):
            entry = {"inn": [w[0]], "basis": "wikidata+ema"}
        elif len(w) == 1 and not e:
            entry = {"inn": [w[0]], "basis": "wikidata"}
        elif len(e) == 1 and not w:
            entry = {"inn": [e[0]], "basis": "ema"}
        elif w or e:
            entry = {"inn": [], "basis": "doubt", "candidates": sorted(set(w + e), key=ds.norm)}
        else:
            entry = {"inn": [], "basis": "unresolved"}
        out[ingredient] = {"atc": sorted(codes), **entry}
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

    if args.wikidata:
        payload = json.loads(Path(args.wikidata).read_text("utf-8"))
    else:
        url = f"{ds.WIKIDATA_SPARQL}?{urllib.parse.urlencode({'query': ds.wikidata_atc_query(), 'format': 'json'})}"
        payload = json.loads(http.get(url, accept="application/sparql-results+json", timeout=180))
    wikidata = ds.read_wikidata_atc(payload)
    manifest["wikidata"] = {"url": ds.WIKIDATA_SPARQL, "taken": today, "licence": ds.WIKIDATA_LICENCE,
                            "codes": len(wikidata)}
    ema_path = Path(args.ema) if args.ema else Path(args.out) / "ema_medicines.xlsx"
    if not args.ema:
        ema_path.write_bytes(http.get(ds.EMA_MEDICINES, accept="*/*", timeout=180))
    ema, ema_report = ds.read_ema_medicines(ema_path)
    manifest["ema"] = {"url": ds.EMA_MEDICINES, "taken": today, "licence": ds.EMA_LICENCE, **ema_report}

    ingredients = resolve(ds.ingredients_of(products), wikidata, ema)
    key = os.environ.get("OPENFDA_API_KEY", "")
    names = sorted({n for e in ingredients.values() for n in e["inn"]}, key=ds.norm)
    asked = 0
    for name in names:
        if cache.has_label_entry(name):
            continue
        if asked >= args.limit:
            break
        cache.put_label(name, ds.fetch_label(http, name, key))
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
                           "asked_this_run": asked, "with_key": bool(key)}
    cache.write_json("manifest.json", manifest)
    summary = {
        "products": len(products), "ingredients": len(ingredients),
        "resolution": dict(Counter(e["basis"] for e in ingredients.values())),
        "international_names": len(names), "labels_asked": len(labels),
        "labels_found": sum(1 for v in labels.values() if v is not None),
        "labels_with_interactions": sum(1 for v in labels.values() if v and v.sections.get("drug_interactions")),
        "labels_still_to_ask": len(names) - len(labels),
        "aifa_columns": aifa_report.get("columns"), "ema_columns": ema_report.get("columns"),
        "http_calls": http.calls,
    }
    cache.write_json("summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
