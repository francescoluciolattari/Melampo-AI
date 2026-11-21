"""The names of the profiled diseases, in English and in Italian (20 November 2026).

    python3 scripts/disease_names.py --profiles profiles.json --mondo mondo.obo --out names.json

A disease of an Italian case is written in Italian ("sospetta appendicite acuta"); its profile is named in English,
by its MONDO id. This writes, for every MONDO id of the profiles, the names that find it
(``memory.profile_library``, ``MELAMPO_DISEASE_NAMES``):

* MONDO's own name and its exact synonyms (CC BY 4.0);
* Wikidata's Italian and English labels and aliases of the item that carries the MONDO id (P5270; CC0).

Only public data are asked: MONDO ids, never a patient's words. A name shared by two diseases is kept for both,
and the library then chooses neither (``ambiguous_name``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from melampo.memory import disease_profiles as dp  # noqa: E402

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
BATCH = 40


# Wikidata properties by which an item is found, from the MONDO id or its cross-references (21 November 2026: the
# first run found no item by the MONDO id alone; the Disease Ontology, MeSH and UMLS ids of the same disease are
# carried by more items)
PROPERTIES = {"MONDO": "P5270", "DOID": "P699", "MESH": "P486", "UMLS": "P2892"}


def identifiers(mondo_id: str, xrefs: list[str]) -> list[tuple[str, str]]:
    """(property, value as Wikidata writes it) for a disease: MONDO with and without its prefix, DOID with it,
    MeSH and UMLS without."""
    out = [("P5270", mondo_id), ("P5270", mondo_id.split(":", 1)[1])]
    for x in xrefs:
        prefix, _, local = x.partition(":")
        prefix = prefix.upper()
        if prefix == "DOID":
            out.append(("P699", x))
        elif prefix in ("MESH", "UMLS", "UMLS_CUI") and local:
            out.append((PROPERTIES["MESH" if prefix == "MESH" else "UMLS"], local))
    return out


def query(ids: list[str], xrefs: dict[str, list[str]] | None = None) -> str:
    """The Wikidata items carrying these diseases' identifiers, with their Italian and English labels and aliases;
    each answer says which disease it is for (?mondo)."""
    rows = []
    for i in ids:
        for prop, value in identifiers(i, (xrefs or {}).get(i, [])):
            rows.append(f'("{i}" wdt:{prop} "{value}")')
    return ("SELECT ?mondo ?name WHERE { VALUES (?mondo ?p ?v) { " + " ".join(rows) + " } "
            "?item ?p ?v . { ?item rdfs:label ?name } UNION { ?item skos:altLabel ?name } "
            "FILTER(LANG(?name) = 'it' || LANG(?name) = 'en') }")


def read(payload: dict) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in payload.get("results", {}).get("bindings", []):
        mondo = row.get("mondo", {}).get("value", "").strip()
        name = row.get("name", {}).get("value", "").strip()
        if mondo and name:
            out.setdefault(mondo if mondo.startswith("MONDO:") else f"MONDO:{mondo}", set()).add(name)
    return out


def ask(sparql: str) -> dict:
    url = f"{WIKIDATA_SPARQL}?{urllib.parse.urlencode({'query': sparql, 'format': 'json'})}"
    req = urllib.request.Request(url, headers={"User-Agent": "melampo-disease-names/1.0 (open data)",
                                               "Accept": "application/sparql-results+json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = r.read()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                print(f"wikidata: not JSON ({len(raw)} bytes): {raw[:200]!r}", flush=True)
                return {}
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            print(f"wikidata: attempt {attempt + 1}: {error}", flush=True)
            time.sleep(5 * (attempt + 1))
    return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profiles", required=True, nargs="+")
    ap.add_argument("--mondo", required=True)
    ap.add_argument("--out", default="names.json")
    ap.add_argument("--no-wikidata", action="store_true")
    args = ap.parse_args(argv)
    ids: set[str] = set()
    for path in args.profiles:
        data = json.loads(Path(path).read_text("utf-8"))
        ids |= {p["disease_id"] for p in data.get("profiles", []) if str(p.get("disease_id", "")).startswith("MONDO:")}
    mondo = dp.Terms.from_obo(Path(args.mondo).read_text("utf-8", errors="replace").splitlines())
    names: dict[str, set[str]] = {i: {mondo.names.get(i, "")} | set(mondo.synonyms.get(i, [])) for i in ids}
    found = 0
    if not args.no_wikidata:
        ordered = sorted(ids)
        for start in range(0, len(ordered), BATCH):
            got = ask(query(ordered[start:start + BATCH], mondo.xrefs))
            print(json.dumps({"wikidata_answers": len(got.get("results", {}).get("bindings", []))}), flush=True)
            for mondo_id, words in read(got).items():
                if mondo_id in names:
                    names[mondo_id] |= words
                    found += 1
    out = {i: sorted(w for w in words if w) for i, words in sorted(names.items())}
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), "utf-8")
    print(json.dumps({"diseases": len(out), "with_wikidata": found,
                      "names": sum(len(v) for v in out.values())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
